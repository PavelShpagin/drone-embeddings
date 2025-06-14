#!/usr/bin/env python3
"""
Fixed SuperPoint fine-tuning on UAV data using homographic adaptation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import argparse
from simple_superpoint import SuperPoint, SuperPointNet

class UAVDataset(Dataset):
    """Dataset for UAV images with homographic augmentation."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.image_files = list(self.data_dir.glob('*.png'))
        self.transform = transform
        
        if not self.image_files:
            raise ValueError(f"No PNG files found in {data_dir}")
        
        print(f"Found {len(self.image_files)} UAV images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            # Return a random image if this one fails
            idx = random.randint(0, len(self.image_files) - 1)
            img_path = self.image_files[idx]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Ensure image is 256x256
        if img.shape != (256, 256):
            img = cv2.resize(img, (256, 256))
        
        # Generate homographic pair
        img1 = img.copy()
        img2, H = self._apply_homography(img)
        
        # Convert to tensors
        img1_tensor = torch.from_numpy(img1.astype(np.float32) / 255.0).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2.astype(np.float32) / 255.0).unsqueeze(0)
        H_tensor = torch.from_numpy(H.astype(np.float32))
        
        return {
            'image1': img1_tensor,
            'image2': img2_tensor,
            'homography': H_tensor
        }
    
    def _apply_homography(self, img):
        """Apply random homography to image with better validation."""
        h, w = img.shape
        
        # Define corners
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        
        # FIXED: Use smaller, more reasonable perturbations
        max_perturbation = min(w, h) * 0.1  # 10% of image size
        perturbation = np.random.uniform(-max_perturbation, max_perturbation, (4, 2)).astype(np.float32)
        perturbed_corners = corners + perturbation
        
        # FIXED: Ensure corners stay within reasonable bounds (with margin)
        margin = 10
        perturbed_corners[:, 0] = np.clip(perturbed_corners[:, 0], margin, w - margin - 1)
        perturbed_corners[:, 1] = np.clip(perturbed_corners[:, 1], margin, h - margin - 1)
        
        # Compute homography
        H = cv2.getPerspectiveTransform(corners, perturbed_corners)
        
        # FIXED: Validate homography is not degenerate
        det = np.linalg.det(H[:2, :2])
        if abs(det) < 1e-6:  # Degenerate homography
            # Return identity transformation
            H = np.eye(3, dtype=np.float32)
            warped = img.copy()
        else:
            # Warp image
            warped = cv2.warpPerspective(img, H, (w, h))
        
        return warped, H

def descriptor_loss_fixed(desc1, desc2, kpts1, kpts2, H, margin=1.0):
    """
    FIXED: Compute descriptor loss for corresponding keypoints.
    """
    if len(kpts1) == 0 or len(kpts2) == 0:
        return torch.tensor(0.0, device=desc1.device)
    
    # Convert keypoints to homogeneous coordinates
    kpts1_h = np.concatenate([kpts1, np.ones((len(kpts1), 1))], axis=1)
    
    # Warp keypoints from image1 to image2
    kpts1_warped = (H @ kpts1_h.T).T
    kpts1_warped = kpts1_warped[:, :2] / (kpts1_warped[:, 2:3] + 1e-8)  # Add epsilon
    
    # Find correspondences (within 3 pixels)
    dists = np.linalg.norm(kpts1_warped[:, None, :] - kpts2[None, :, :], axis=2)
    matches = np.where(dists < 3.0)
    
    if len(matches[0]) == 0:
        return torch.tensor(0.0, device=desc1.device)
    
    # Get matched descriptors
    desc1_matched = desc1[matches[0]]
    desc2_matched = desc2[matches[1]]
    
    # FIXED: Positive loss - minimize distance for matches
    pos_dists = torch.norm(desc1_matched - desc2_matched, dim=1)
    pos_loss = torch.mean(pos_dists)  # Want to minimize this
    
    # FIXED: Better negative sampling - ensure true negatives
    if len(desc1) > len(matches[0]) and len(desc2) > len(matches[1]):
        # Create mask of matched pairs
        matched_pairs = set(zip(matches[0], matches[1]))
        
        # Sample negatives that are NOT in matched pairs
        n_neg = min(len(matches[0]) * 2, 200)  # More negatives
        neg_pairs = []
        
        for _ in range(n_neg):
            idx1 = np.random.randint(0, len(desc1))
            idx2 = np.random.randint(0, len(desc2))
            
            # Ensure this is not a matched pair
            if (idx1, idx2) not in matched_pairs:
                neg_pairs.append((idx1, idx2))
        
        if neg_pairs:
            neg_idx1, neg_idx2 = zip(*neg_pairs)
            desc1_neg = desc1[list(neg_idx1)]
            desc2_neg = desc2[list(neg_idx2)]
            
            neg_dists = torch.norm(desc1_neg - desc2_neg, dim=1)
            neg_loss = torch.mean(torch.clamp(margin - neg_dists, min=0))  # Want to maximize distance
        else:
            neg_loss = torch.tensor(0.0, device=desc1.device)
    else:
        neg_loss = torch.tensor(0.0, device=desc1.device)
    
    return pos_loss + neg_loss

def get_keypoints_from_heatmap(semi, conf_thresh=0.015, nms_dist=4, cell=8):
    """Extract keypoints from detector output efficiently."""
    # Convert to numpy
    semi_np = semi.detach().cpu().numpy()  # [B, 65, H/8, W/8]
    
    batch_keypoints = []
    
    for b in range(semi_np.shape[0]):
        # Extract keypoints for this batch item
        dense = np.exp(semi_np[b])  # Softmax
        dense = dense / (np.sum(dense, axis=0) + 1e-6)  # Normalize
        
        # Remove dustbin (no-keypoint class)
        nodust = dense[:-1, :, :]  # [64, H/8, W/8]
        
        # Reshape to get keypoint probabilities per cell
        Hc, Wc = nodust.shape[1], nodust.shape[2]
        nodust = nodust.transpose(1, 2, 0)  # [H/8, W/8, 64]
        heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * cell, Wc * cell])  # [H, W]
        
        # Find keypoints above threshold
        xs, ys = np.where(heatmap >= conf_thresh)
        if len(xs) == 0:
            batch_keypoints.append(np.zeros((0, 2)))
            continue
            
        pts = np.zeros((len(xs), 2))
        pts[:, 0] = ys  # x coordinates
        pts[:, 1] = xs  # y coordinates
        scores = heatmap[xs, ys]
        
        # Apply NMS
        if nms_dist > 0 and len(pts) > 1:
            keep = nms_fast(pts, scores, nms_dist)
            pts = pts[keep]
        
        batch_keypoints.append(pts)
    
    return batch_keypoints

def nms_fast(pts, scores, nms_dist):
    """Fast NMS implementation."""
    if len(pts) == 0:
        return []
        
    # Sort by score (descending)
    order = np.argsort(scores)[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
            
        # Compute distances to remaining points
        dists = np.sqrt(np.sum((pts[order[1:]] - pts[i]) ** 2, axis=1))
        
        # Keep points that are far enough
        inds = np.where(dists >= nms_dist)[0]
        order = order[inds + 1]
        
    return keep

def train_superpoint(data_dir, pretrained_weights, output_dir, epochs=20, batch_size=4, lr=1e-4):
    """FIXED: Train SuperPoint on UAV data."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained SuperPoint
    superpoint = SuperPoint(pretrained_weights, device)
    model = superpoint.net
    model.train()
    
    # Setup dataset and dataloader
    dataset = UAVDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Reduced workers
    
    # Setup optimizer (only train descriptor head)
    # Freeze encoder and detector, only train descriptor head
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze descriptor head
    for param in model.convDa.parameters():
        param.requires_grad = True
    for param in model.convDb.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    # Training loop
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            H_batch = batch['homography'].cpu().numpy()
            
            # Forward pass
            semi1, desc1 = model(img1)
            semi2, desc2 = model(img2)
            
            # FIXED: More efficient keypoint extraction
            with torch.no_grad():
                kpts1_batch = get_keypoints_from_heatmap(semi1)
                kpts2_batch = get_keypoints_from_heatmap(semi2)
            
            batch_loss = 0.0
            valid_samples = 0
            
            # Process each sample in the batch
            for i in range(len(img1)):
                kpts1 = kpts1_batch[i]
                kpts2 = kpts2_batch[i]
                
                if len(kpts1) > 0 and len(kpts2) > 0:
                    # Sample descriptors at keypoint locations
                    desc1_sample = sample_descriptors(desc1[i], kpts1)
                    desc2_sample = sample_descriptors(desc2[i], kpts2)
                    
                    # FIXED: Use corrected descriptor loss
                    loss = descriptor_loss_fixed(desc1_sample, desc2_sample, kpts1, kpts2, H_batch[i])
                    batch_loss += loss
                    valid_samples += 1
            
            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': batch_loss.item()})
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / f"superpoint_uav_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "superpoint_uav_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training complete! Final model saved: {final_path}")

def sample_descriptors(desc_map, keypoints):
    """Sample descriptors at keypoint locations."""
    if len(keypoints) == 0:
        return torch.zeros((0, 256), device=desc_map.device)
    
    # Convert keypoints to descriptor map coordinates
    kpts_scaled = keypoints / 8.0  # SuperPoint uses 8x downsampling
    
    # Clamp to valid range
    H, W = desc_map.shape[1], desc_map.shape[2]
    kpts_scaled[:, 0] = np.clip(kpts_scaled[:, 0], 0, W - 1)
    kpts_scaled[:, 1] = np.clip(kpts_scaled[:, 1], 0, H - 1)
    
    # Sample using bilinear interpolation
    descriptors = []
    for kpt in kpts_scaled:
        x, y = kpt
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
        
        # Bilinear weights
        wx = x - x0
        wy = y - y0
        
        # Sample descriptor
        d = (1 - wx) * (1 - wy) * desc_map[:, y0, x0] + \
            wx * (1 - wy) * desc_map[:, y0, x1] + \
            (1 - wx) * wy * desc_map[:, y1, x0] + \
            wx * wy * desc_map[:, y1, x1]
        
        descriptors.append(d)
    
    return torch.stack(descriptors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SuperPoint on UAV data")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing UAV training images')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                       help='Path to pretrained SuperPoint weights')
    parser.add_argument('--output_dir', type=str, default='superpoint_uav_trained',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    train_superpoint(
        args.data_dir,
        args.pretrained_weights,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.lr
    ) 