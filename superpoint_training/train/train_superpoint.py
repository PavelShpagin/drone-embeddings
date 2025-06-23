import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast # For mixed precision
import torch.nn.functional as F
from torchvision import transforms

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import math
import time

# Import SuperPointNet and helper functions
from superpoint_training.superpoint_model import SuperPointNet, get_keypoints_from_heatmap, nms_fast_pytorch

# --- Configuration and Hyperparameters ---
# Data paths
CROPS_DIR = "superpoint_training/crops"
# PSEUDOLABELS_DIR = "superpoint_training/pseudolabels" # Removed as pseudolabels are generated dynamically
CHECKPOINTS_DIR = "superpoint_training/checkpoints"
FINAL_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "superpoint_uav_final.pth") # Path to the supervisor model

# Training Hyperparameters
IMAGE_SIZE = 256
CELL_SIZE = 8
BATCH_SIZE = 512 # Set for H100 parallelism
LEARNING_RATE = 0.001
EPOCHS = 5000 # Large number, monitor for early stopping
LOG_INTERVAL = 10 # Log every N batches
SAVE_INTERVAL = 50 # Save checkpoint every N epochs

# Loss Weights (from SuperPoint paper)
LAMBDA_DESC = 0.0001 # λ for balancing descriptor loss

# Descriptor Hinge Loss Margins (from SuperPoint paper)
MARGIN_POS = 1.0
MARGIN_NEG = 0.2

# Homography Augmentation Parameters (for on-the-fly data augmentation during training)
# These are typically more restrictive than for pseudo-label generation to mimic real camera motion
TRAIN_HOMOGRAPHY_PARAMS = {
    'translation': True, 'rotation': True, 'scaling': True, 'perspective': True,
    'max_scale': 0.2, 'min_scale': 0.8, 'max_angle': 15, 'perspective_amplitude_x': 0.1,
    'perspective_amplitude_y': 0.1, 'allow_artifacts': True,
    'patch_ratio': 0.9 # Slightly smaller patch ratio for training homographies
}

# Keypoint detection parameters (for extracting kpts for descriptor loss)
CONF_THRESH = 0.015
NMS_DIST = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Augmentations (on-the-fly) ---
# Random transformations for robust training
class RandomPhotometricAugmentations(object):
    def __call__(self, img_np):
        # img_np is a numpy array (H, W), [0, 1]
        img = Image.fromarray((img_np * 255).astype(np.uint8))

        # Brightness, Contrast
        if random.random() < 0.5:
            img = transforms.ColorJitter(brightness=0.3, contrast=0.3)(img)
        
        # Gaussian Noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.05, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 1)
            img = Image.fromarray((img_np * 255).astype(np.uint8))

        # Gaussian Blur
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Convert back to numpy and normalize
        return np.array(img, dtype=np.float32) / 255.0

photometric_augmentations = RandomPhotometricAugmentations()

# --- Helper Functions for Homography Generation (GPU-accelerated and Batched) ---
def generate_homography_batch_torch(image_size, params, batch_size, device):
    """
    Generates a batch of random homography matrices H and their inverses H_inv (torch tensors).
    Adapted from rpautrat's SuperPoint implementation, using PyTorch operations (DLT).
    """
    height, width = image_size, image_size

    # Source points (corners of the image) for the batch
    # [0, 0], [W-1, 0], [W-1, H-1], [0, H-1]
    pts1_template = torch.tensor([
        [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
    ], dtype=torch.float32, device=device) # [4, 2]
    pts1 = pts1_template.unsqueeze(0).repeat(batch_size, 1, 1) # [B, 4, 2]

    # Perturb the corners to create destination points for the batch
    s = width * params['patch_ratio']
    center = torch.tensor([width/2., height/2.], dtype=torch.float32, device=device) # [2]
    
    perturbations = torch.randn(batch_size, 4, 2, device=device) * (s / 2) # [B, 4, 2]
    pts2 = center.unsqueeze(0).unsqueeze(0) + perturbations # [B, 4, 2]

    # Apply random transformations (translation, rotation, scale, perspective) for the batch
    # Translation
    if params['translation']:
        t = torch.randn(batch_size, 2, device=device) * 0.1 * torch.tensor([width, height], device=device) # [B, 2]
        pts2 += t.unsqueeze(1) # [B, 1, 2] -> [B, 4, 2]

    # Rotation
    if params['rotation']:
        angle = torch.deg2rad(torch.rand(batch_size, device=device) * (params['max_angle'] * 2) - params['max_angle']) # [B]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        R = torch.stack([
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1)
        ], dim=1) # [B, 2, 2]
        
        pts2_centered = pts2 - center.unsqueeze(0).unsqueeze(0) # [B, 4, 2]
        pts2 = torch.bmm(pts2_centered, R.transpose(-1, -2)) + center.unsqueeze(0).unsqueeze(0)

    # Scaling
    if params['scaling']:
        s_factor = torch.exp(torch.rand(batch_size, device=device) * (torch.log(torch.tensor(params['max_scale'], device=device)) - torch.log(torch.tensor(params['min_scale'], device=device))) + torch.log(torch.tensor(params['min_scale'], device=device))) # [B]
        pts2_centered = pts2 - center.unsqueeze(0).unsqueeze(0) # [B, 4, 2]
        pts2 = pts2_centered * s_factor.unsqueeze(1).unsqueeze(2) + center.unsqueeze(0).unsqueeze(0)

    # Perspective
    if params['perspective']:
        h_persp = torch.randn(batch_size, 2, device=device) * params['perspective_amplitude_x'] * width # [B, 2]
        w_persp = torch.randn(batch_size, 2, device=device) * params['perspective_amplitude_y'] * height # [B, 2]

        pts2[:, 0, 0] += w_persp[:, 0]
        pts2[:, 0, 1] += h_persp[:, 0]
        pts2[:, 1, 0] -= w_persp[:, 1]
        pts2[:, 1, 1] += h_persp[:, 1]
        pts2[:, 2, 0] += w_persp[:, 1]
        pts2[:, 2, 1] -= h_persp[:, 1]
        pts2[:, 3, 0] -= w_persp[:, 0]
        pts2[:, 3, 1] -= h_persp[:, 0]
    
    # DLT (Direct Linear Transform) for batch
    # Construct A matrix for each homography in the batch
    A_batch = torch.zeros(batch_size, 8, 9, device=device, dtype=torch.float32)

    # We need to construct the A matrix for DLT. This can be done more efficiently
    # without an explicit Python loop per batch item by using tensor operations.
    # However, given the complexity of the A matrix construction for 8 points and 9 unknowns,
    # and that pts1 and pts2 are already batched, we will use a small loop over the 4 points
    # for each batch item for clarity and correctness with the DLT formulation. The main
    # performance gain will come from batching SVD.
    for j in range(4):
        x, y = pts1[:, j, 0], pts1[:, j, 1] # [B]
        x_prime, y_prime = pts2[:, j, 0], pts2[:, j, 1] # [B]

        # Fill A_batch for current point j across all batch items
        A_batch[:, 2*j, 0] = -x
        A_batch[:, 2*j, 1] = -y
        A_batch[:, 2*j, 2] = -1
        A_batch[:, 2*j, 6] = x*x_prime
        A_batch[:, 2*j, 7] = y*x_prime
        A_batch[:, 2*j, 8] = x_prime

        A_batch[:, 2*j + 1, 3] = -x
        A_batch[:, 2*j + 1, 4] = -y
        A_batch[:, 2*j + 1, 5] = -1
        A_batch[:, 2*j + 1, 6] = x*y_prime
        A_batch[:, 2*j + 1, 7] = y*y_prime
        A_batch[:, 2*j + 1, 8] = y_prime
    
    # Solve Ah = 0 using SVD for the batch
    U, S, V = torch.linalg.svd(A_batch)
    h = V[:, :, -1] # Last column of V is the null space vector (h_vec) for each batch item
    H_batch = h.reshape(batch_size, 3, 3)

    # Batched inverse for H_batch
    H_inv_batch = torch.linalg.inv(H_batch)

    return H_batch, H_inv_batch

# --- Dataset Class ---
class SuperPointDataset(Dataset):
    def __init__(self, crops_dir, image_size, cell_size, augmentations=None):
        self.image_paths = list(Path(crops_dir).glob('*.png'))

        self.image_size = image_size
        self.cell_size = cell_size
        # self.homography_params is no longer needed here as homographies are generated in training loop
        self.augmentations = augmentations

        print(f"Found {len(self.image_paths)} images for training.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load original image
        img = Image.open(img_path).convert("L") # Grayscale
        img = img.resize((self.image_size, self.image_size))
        img_np = np.array(img, dtype=np.float32) / 255.0 # Normalize to [0,1]

        # Apply photometric augmentations to original image
        if self.augmentations:
            img_np = self.augmentations(img_np)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float() # [1, H, W]
        
        return img_tensor # Return only the image

# --- Loss Functions ---
def detector_loss(pred_logits, true_heatmap):
    """
    Compute the keypoint detector loss (cross-entropy).
    pred_logits: [B, 65, Hc, Wc] (raw output from convDb)
    true_heatmap: [B, 65, Hc, Wc] (pseudo-ground-truth heatmap)
    """
    # Reshape logits to [B*Hc*Wc, 65] and true_heatmap to [B*Hc*Wc]
    # For cross_entropy, true_heatmap should be long type and represent class indices
    # However, pseudo_label_semi is a one-hot like distribution across 65 channels.
    # So, we should use F.log_softmax and then negative dot product or MSE/KLDivLoss

    # Option 1: Treat as multi-class classification and convert pseudo_label to class index
    # This assumes true_heatmap is essentially a one-hot with 64 for a keypoint, and 64 for dustbin.
    # Given pseudo_label_semi is averaged from many heatmaps, it's more like probabilities.
    # A better approach is to use F.kl_div with F.log_softmax.

    log_softmax = F.log_softmax(pred_logits, dim=1)
    loss = F.kl_div(log_softmax, true_heatmap, reduction='batchmean', log_target=False)
    return loss

def descriptor_loss(desc1, desc2, kpts1_list, kpts2_list, H_batch, margin_pos, margin_neg, lambda_d):
    """
    Compute the descriptor loss (hinge loss) for a batch of images.
    desc1, desc2: [B, D, Hc, Wc] - descriptor maps
    kpts1_list, kpts2_list: list of lists of keypoints for each image in batch
    H_batch: [B, 3, 3] - homography matrices
    """
    batch_loss = torch.tensor(0.0, device=desc1.device)
    valid_samples = 0

    # Need to process each item in the batch individually for keypoint extraction
    # and then aggregate loss. This will be slow on CPU for dataloader, but inside train step, it's fine.
    
    for i in range(desc1.shape[0]): # Iterate over batch items
        desc1_i = desc1[i].unsqueeze(0) # [1, D, Hc, Wc]
        desc2_i = desc2[i].unsqueeze(0) # [1, D, Hc, Wc]
        kpts1 = kpts1_list[i]
        kpts2 = kpts2_list[i]
        H = H_batch[i]

        if len(kpts1) == 0 or len(kpts2) == 0:
            continue

        # Convert keypoints to homogeneous coordinates
        kpts1_h = torch.cat([kpts1, torch.ones(len(kpts1), 1, device=kpts1.device)], dim=1)
        
        # Warp keypoints from image1 to image2 using ground truth homography
        kpts1_warped_h = (H @ kpts1_h.T).T
        kpts1_warped = kpts1_warped_h[:, :2] / (kpts1_warped_h[:, 2:3] + 1e-8) # Dehomogenize
        
        # Find correspondences (within 3 pixels - this needs to be consistent with NMS_DIST if used as correspondence threshold)
        # For descriptor loss, typically use a more exact match for positives
        # Here, we use the correspondence as defined by the homography plus a small tolerance
        # A better way is to use `warp_points_pytorch` from rpautrat's code.

        # This part should be optimized for GPU in a real H100 setup, possibly using kornia or custom CUDA op.
        # Removed kpts_warped_np and kpts2_np conversion to numpy.

        # Grid coordinates for sampling descriptors
        grid_kpts1 = kpts1 / (IMAGE_SIZE / CELL_SIZE - 1) * 2 - 1 # Normalize to [-1, 1]
        grid_kpts1 = grid_kpts1.unsqueeze(0).unsqueeze(0) # [1, 1, N, 2]
        
        grid_kpts2 = kpts2 / (IMAGE_SIZE / CELL_SIZE - 1) * 2 - 1 # Normalize to [-1, 1]
        grid_kpts2 = grid_kpts2.unsqueeze(0).unsqueeze(0) # [1, 1, N, 2]

        # Sample descriptors at keypoint locations
        desc1_sampled = F.grid_sample(desc1_i, grid_kpts1, mode='bilinear', align_corners=True).squeeze().T # [N, D]
        desc2_sampled = F.grid_sample(desc2_i, grid_kpts2, mode='bilinear', align_corners=True).squeeze().T # [N, D]

        if desc1_sampled.dim() == 1: desc1_sampled = desc1_sampled.unsqueeze(0)
        if desc2_sampled.dim() == 1: desc2_sampled = desc2_sampled.unsqueeze(0)

        # Compute all-pairs Euclidean distance between desc1 and desc2
        # Correspondence based on homography and pixel distance (PyTorch-native)
        # Using broadcasting to compute all-pairs squared Euclidean distance
        dists = torch.cdist(kpts1_warped, kpts2) # [N1, N2]
        matches = (dists < 3.0).nonzero(as_tuple=True) # (row_indices, col_indices)

        if matches[0].numel() == 0:
            continue
        
        # Positive pairs
        pos_desc1 = desc1_sampled[matches[0]]
        pos_desc2 = desc2_sampled[matches[1]]
        pos_dists = torch.norm(pos_desc1 - pos_desc2, dim=1)
        pos_loss = torch.mean(torch.clamp(pos_dists - margin_pos, min=0))

        # Negative pairs
        # Generate a mask for non-matches
        non_matches_mask = (dists >= 3.0)
        # Get indices of negative candidates
        neg_candidate_idx1, neg_candidate_idx2 = non_matches_mask.nonzero(as_tuple=True)

        if neg_candidate_idx1.numel() > 0:
            # Sample a subset of negative pairs to balance with positives
            num_neg_samples = min(neg_candidate_idx1.numel(), pos_dists.numel() * 5) # 5x positives
            
            # Randomly sample indices for negative pairs
            perm = torch.randperm(neg_candidate_idx1.numel(), device=DEVICE)[:num_neg_samples]
            neg_idx1 = neg_candidate_idx1[perm]
            neg_idx2 = neg_candidate_idx2[perm]

            neg_desc1 = desc1_sampled[neg_idx1]
            neg_desc2 = desc2_sampled[neg_idx2]
            neg_dists = torch.norm(neg_desc1 - neg_desc2, dim=1)
            neg_loss = torch.mean(torch.clamp(margin_neg - neg_dists, min=0))
        else:
            neg_loss = torch.tensor(0.0, device=DEVICE)

        batch_loss += (pos_loss + neg_loss) * lambda_d
        valid_samples += 1
    
    return batch_loss / max(valid_samples, 1)

# --- Training Function ---
def train_superpoint(args):
    # Initialize models
    model = SuperPointNet(pred_init=True).to(DEVICE) # Model for training
    
    # Load supervisor model (frozen)
    supervisor_model = SuperPointNet(pred_init=True).to(DEVICE)
    if os.path.exists(FINAL_MODEL_PATH):
        supervisor_model.load_state_dict(torch.load(FINAL_MODEL_PATH))
        print(f"Loaded supervisor model from {FINAL_MODEL_PATH}")
    else:
        print(f"WARNING: Supervisor model not found at {FINAL_MODEL_PATH}. Initializing with random weights.")
    supervisor_model.eval() # Set to evaluation mode
    for param in supervisor_model.parameters():
        param.requires_grad = False # Freeze supervisor model

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() # For mixed precision training

    # Dataset and DataLoader
    dataset = SuperPointDataset(
        crops_dir=CROPS_DIR,
        image_size=IMAGE_SIZE,
        cell_size=CELL_SIZE,
        augmentations=photometric_augmentations
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print(f"Starting training on {len(dataset)} images.")

    # Training Loop
    for epoch in range(EPOCHS):
        model.train() # Set current model to training mode
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_detector_loss = 0.0
        total_descriptor_loss = 0.0

        for batch_idx, img1_orig in enumerate(pbar):
            # img1_orig is [B, 1, H, W] from the dataset
            
            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16): # Enable mixed precision for H100
                # Generate Homography for original image (for pseudolabel generation)
                # We need homographies that are more aggressive for IHA.
                # These homographies are used to warp the input images for the supervisor model
                # to generate robust pseudo-labels.
                H_pseudo_label, H_pseudo_label_inv = generate_homography_batch_torch(
                    IMAGE_SIZE, TRAIN_HOMOGRAPHY_PARAMS, img1_orig.shape[0], DEVICE
                )

                # Warp original image to create img_pseudo_source for supervisor
                # This image is fed to the (frozen) supervisor model to get high-quality keypoint
                # and descriptor pseudo-labels.
                grid_pseudo_label = F.affine_grid(H_pseudo_label[:, :2, :], img1_orig.size(), align_corners=True)
                img_pseudo_source = F.grid_sample(img1_orig, grid_pseudo_label, mode='bilinear', padding_mode='zeros', align_corners=True) # [B, 1, H, W]

                # Generate pseudolabels using the frozen supervisor model
                with torch.no_grad():
                    # Output from supervisor model (raw logits and descriptors)
                    heatmap_pseudo_logits, desc_pseudo = supervisor_model(img_pseudo_source)
                    
                    # Apply softmax to detector logits to get a probability distribution over 65 classes
                    pseudo_label_semi = F.softmax(heatmap_pseudo_logits, dim=1) # [B, 65, Hc, Wc]

                # Unwarp pseudolabels back to the original image's coordinate system using the inverse homography.
                # This creates the ground-truth heatmap (true_heatmap_pseudo) for the detector loss.
                # The homography transformation for features (Hc, Wc) is the same as for image (H, W)
                # after scaling the coordinates appropriately. affine_grid handles this if the output size
                # is correctly specified as the feature map size.
                grid_pseudo_label_inv = F.affine_grid(H_pseudo_label_inv[:, :2, :], pseudo_label_semi.size(), align_corners=True)
                true_heatmap_pseudo = F.grid_sample(pseudo_label_semi, grid_pseudo_label_inv, mode='bilinear', padding_mode='zeros', align_corners=True)

                # Generate new homographies for the current training pair (img1_orig, img2).
                # These homographies are typically less aggressive than those used for pseudo-labeling
                # to simulate more realistic camera motion between consecutive frames.
                H_train, H_train_inv = generate_homography_batch_torch(
                    IMAGE_SIZE, TRAIN_HOMOGRAPHY_PARAMS, img1_orig.shape[0], DEVICE
                )

                # Warp original image to create img2 for the training pair
                grid_train = F.affine_grid(H_train[:, :2, :], img1_orig.size(), align_corners=True)
                img2 = F.grid_sample(img1_orig, grid_train, mode='bilinear', padding_mode='zeros', align_corners=True) # [B, 1, H, W]

                # Forward pass for both images through the current model
                # This generates the predicted keypoint logits and descriptors for the training pair.
                pred_logits1, desc1 = model(img1_orig)
                pred_logits2, desc2 = model(img2)
                
                # Detector Loss: Compares the current model's detector output for img1_orig
                # against the unwarped pseudo-ground-truth heatmap generated by the supervisor.
                d_loss = detector_loss(pred_logits1, true_heatmap_pseudo)

                # Descriptor Loss: Computes the triplet margin loss for descriptors.
                # This requires extracting keypoints from the current model's output (pred_logits1, pred_logits2)
                # and using the homography (H_train) to establish positive matches between img1 and img2.
                
                # Get keypoints for descriptor loss from current model's output
                # The get_keypoints_from_heatmap function expects raw logits and applies softmax internally.
                # So, we pass pred_logits1 and pred_logits2 directly.
                # heatmap1 = F.softmax(pred_logits1, dim=1)
                # heatmap2 = F.softmax(pred_logits2, dim=1)

                # Extract keypoints in a loop (can be a bottleneck, but get_keypoints_from_heatmap is not easily batched due to NMS).
                kpts1_list = []
                kpts2_list = []
                for i in range(img1_orig.shape[0]):
                    # Pass raw logits to get_keypoints_from_heatmap
                    kpts1, _ = get_keypoints_from_heatmap(pred_logits1[i:i+1], conf_thresh=CONF_THRESH, nms_dist=NMS_DIST, cell=CELL_SIZE)
                    kpts2, _ = get_keypoints_from_heatmap(pred_logits2[i:i+1], conf_thresh=CONF_THRESH, nms_dist=NMS_DIST, cell=CELL_SIZE)
                    kpts1_list.append(kpts1)
                    kpts2_list.append(kpts2)

                # The descriptor loss function expects [B,D,Hc,Wc] and list of keypoints.
                # H_train is used to compute the correct matches, establishing which keypoints in img1 should
                # correspond to which keypoints in img2 after the homography.
                des_loss = descriptor_loss(desc1, desc2, kpts1_list, kpts2_list, H_train, MARGIN_POS, MARGIN_NEG, LAMBDA_DESC)

                # Total loss is a weighted sum of detector and descriptor losses.
                loss = d_loss + des_loss
            
            # Backpropagation and optimization step.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_detector_loss += d_loss.item()
            total_descriptor_loss += des_loss.item()

            # Update tqdm progress bar with current average losses.
            pbar.set_postfix({'det_loss': total_detector_loss / (batch_idx + 1), 'desc_loss': total_descriptor_loss / (batch_idx + 1)})

        # Save checkpoint at specified intervals.
        if (epoch + 1) % SAVE_INTERVAL == 0:
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True) # Ensure checkpoint directory exists
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'superpoint_uav_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training finished.")

if __name__ == '__main__':
    # Argument parser for command-line arguments.
    parser = argparse.ArgumentParser(description='Train SuperPoint model with Iterative Homographic Adaptation.')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2 or 1, 
                        help='Number of data loading workers (default: half of CPU cores).')
    args = parser.parse_args()
    train_superpoint(args) 