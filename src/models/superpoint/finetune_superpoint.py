import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, SuperPointForKeypointDetection, get_linear_schedule_with_warmup
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path

from .superpoint_model import SuperPoint
from .homographic_adaptation import generate_homography, warp_points

class SuperPointDataset(Dataset):
    def __init__(self, data_dir, stage='magicpoint', transform=None):
        """
        Dataset for SuperPoint training.
        
        Args:
            data_dir: Directory containing training data
            stage: One of ['magicpoint', 'superpoint']
            transform: Optional transform
        """
        self.data_dir = Path(data_dir)
        self.stage = stage
        self.transform = transform
        
        # Get image paths
        if stage == 'magicpoint':
            self.image_files = list(self.data_dir.glob('*.png'))
        else:
            self.image_files = list(self.data_dir.glob('*_image.png'))
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        
        # Load or generate keypoint labels
        if self.stage == 'magicpoint':
            # Generate synthetic labels using corner detection
            corners = cv2.goodFeaturesToTrack(img, maxCorners=1000, qualityLevel=0.01, minDistance=8)
            if corners is not None:
                corners = corners.squeeze()
            else:
                corners = np.zeros((0, 2))
                
            # Convert to cell format
            keypoint_map = np.zeros((H//8, W//8), dtype=np.float32)
            for corner in corners:
                x, y = corner.astype(int)
                cell_x, cell_y = x // 8, y // 8
                if 0 <= cell_x < W//8 and 0 <= cell_y < H//8:
                    keypoint_map[cell_y, cell_x] = 1.0
        else:
            # Load pre-computed pseudo ground truth
            keypoint_map = np.load(str(img_path).replace('_image.png', '_points.npy'))
        
        # Convert image to tensor
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        
        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return {
            'image': img_tensor,
            'keypoint_map': torch.from_numpy(keypoint_map)
        }

def collate_fn(batch):
    # Stack images and keypoint maps
    images = torch.stack([item['image'] for item in batch])
    keypoint_maps = torch.stack([item['keypoint_map'] for item in batch])
    
    return {
        'image': images,
        'keypoint_map': keypoint_maps
    }

class HomographicPairDataset(Dataset):
    """Dataset for training descriptors with homographic pairs."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_files = list(self.data_dir.glob('*_image.png'))
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        
        # Load keypoint map
        keypoint_map = np.load(str(img_path).replace('_image.png', '_points.npy'))
        
        # Generate random homography
        H_mat = generate_homography((H, W))
        
        # Warp image and keypoint map
        warped_img = cv2.warpPerspective(img, H_mat, (W, H), flags=cv2.INTER_LINEAR)
        warped_kp = cv2.warpPerspective(keypoint_map, H_mat, (W//8, H//8), flags=cv2.INTER_LINEAR)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img).float() / 255.0
        warped_tensor = torch.from_numpy(warped_img).float() / 255.0
        
        # Add channel dimension
        img_tensor = img_tensor.unsqueeze(0)
        warped_tensor = warped_tensor.unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)
            warped_tensor = self.transform(warped_tensor)
            
        return {
            'image1': img_tensor,
            'image2': warped_tensor,
            'keypoints1': torch.from_numpy(keypoint_map),
            'keypoints2': torch.from_numpy(warped_kp),
            'H': torch.from_numpy(H_mat.astype(np.float32))
        }

def collate_homographic_pairs(batch):
    # Stack all tensors
    images1 = torch.stack([item['image1'] for item in batch])
    images2 = torch.stack([item['image2'] for item in batch])
    keypoints1 = torch.stack([item['keypoints1'] for item in batch])
    keypoints2 = torch.stack([item['keypoints2'] for item in batch])
    H_mats = torch.stack([item['H'] for item in batch])
    
    return {
        'image1': images1,
        'image2': images2,
        'keypoints1': keypoints1,
        'keypoints2': keypoints2,
        'H': H_mats
    }

def descriptor_loss(desc1, desc2, H, keypoints1, keypoints2, positive_margin=1, negative_margin=0.2):
    """
    Compute descriptor loss using contrastive learning.
    Match keypoints using homography and enforce descriptor similarity.
    """
    B = desc1.shape[0]
    loss = 0
    
    for b in range(B):
        # Get valid keypoints
        kp1 = keypoints1[b].nonzero(as_tuple=False).float()  # (N1,2)
        kp2 = keypoints2[b].nonzero(as_tuple=False).float()  # (N2,2)
        
        if len(kp1) == 0 or len(kp2) == 0:
            continue
            
        # Scale keypoints to original image space
        kp1 = kp1 * 8
        kp2 = kp2 * 8
        
        # Warp keypoints1 to image2
        kp1_warped = warp_points(kp1.cpu().numpy(), H[b].cpu().numpy())
        kp1_warped = torch.from_numpy(kp1_warped).to(kp1.device)
        
        # Find matching keypoints (within 3 pixels)
        dists = torch.cdist(kp1_warped, kp2)  # (N1,N2)
        matches = dists < 3  # (N1,N2) boolean mask
        
        if not matches.any():
            continue
            
        # Get descriptors for keypoints
        d1 = F.grid_sample(desc1[b:b+1], kp1.view(1,-1,1,2) / 8, align_corners=True)  # (1,256,1,N1)
        d2 = F.grid_sample(desc2[b:b+1], kp2.view(1,-1,1,2) / 8, align_corners=True)  # (1,256,1,N2)
        
        d1 = d1.squeeze().t()  # (N1,256)
        d2 = d2.squeeze().t()  # (N2,256)
        
        # Compute positive and negative pairs
        sim = torch.mm(d1, d2.t())  # (N1,N2) cosine similarity
        
        # Positive loss
        pos_loss = F.relu(positive_margin - sim[matches]).mean()
        
        # Negative loss (hardest negative mining)
        sim_neg = sim.clone()
        sim_neg[matches] = -1  # Exclude positive pairs
        hardest_neg = sim_neg.max(dim=1)[0]  # Hardest negative for each keypoint in image1
        neg_loss = F.relu(hardest_neg - negative_margin).mean()
        
        loss += pos_loss + neg_loss
        
    return loss / B if B > 0 else torch.tensor(0.0).to(desc1.device)

def train_superpoint(
    data_dir,
    output_dir,
    stage='magicpoint',
    epochs=50,
    batch_size=32,
    lr=1e-4,
    device='cuda',
    start_epoch=0,
    save_every=5
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = SuperPoint().to(device)
    model.train()
    
    # Setup data
    if stage == 'magicpoint':
        dataset = SuperPointDataset(data_dir, stage='magicpoint')
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, collate_fn=collate_fn)
    else:
        dataset = HomographicPairDataset(data_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, collate_fn=collate_homographic_pairs)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100,
        num_training_steps=epochs*len(loader)
    )
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            
            if stage == 'magicpoint':
                # Train detector only
                outputs = model(batch['image'], batch['keypoint_map'])
                loss = outputs['loss']
            else:
                # Train both detector and descriptor
                # Forward pass on both images
                outputs1 = model(batch['image1'], batch['keypoints1'])
                outputs2 = model(batch['image2'], batch['keypoints2'])
                
                # Detector loss
                det_loss1 = outputs1['loss']
                det_loss2 = outputs2['loss']
                
                # Descriptor loss
                desc_loss = descriptor_loss(
                    outputs1['descriptors'],
                    outputs2['descriptors'],
                    batch['H'],
                    batch['keypoints1'],
                    batch['keypoints2']
                )
                
                # Total loss
                loss = det_loss1 + det_loss2 + desc_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Log epoch metrics
        print(f"\nEpoch {epoch+1} average loss: {epoch_loss/len(loader):.4f}")
        
        # Save checkpoint
        if (epoch+1) % save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss/len(loader)
            }
            torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pth"))
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': epoch_loss/len(loader)
    }
    torch.save(checkpoint, os.path.join(output_dir, "final.pth"))
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save checkpoints')
    parser.add_argument('--stage', type=str, default='magicpoint', choices=['magicpoint', 'superpoint'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    train_superpoint(
        args.data_dir,
        args.output_dir,
        args.stage,
        args.epochs,
        args.batch_size,
        args.lr,
        args.device
    ) 