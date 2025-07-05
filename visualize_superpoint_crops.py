#!/usr/bin/env python3
"""
Clean SuperPoint visualization on 6 random 100x100 crops from large map images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys
import torch
import torch.nn as nn

# Add the pytorch-superpoint models to the path
sys.path.append('third_party/pytorch-superpoint')
from models.SuperPointNet_gauss2 import SuperPointNet_gauss2

def sample_crop(image, crop_size=100):
    """Sample a random crop from an image."""
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return None
    
    # Random top-left corner
    max_y = h - crop_size
    max_x = w - crop_size
    
    y = random.randint(0, max_y)
    x = random.randint(0, max_x)
    
    # Extract crop
    if len(image.shape) == 3:
        crop = image[y:y+crop_size, x:x+crop_size]
    else:
        crop = image[y:y+crop_size, x:x+crop_size]
    
    return crop

def process_superpoint_output(semi, desc, conf_thresh=0.01, nms_dist=4):
    """Process SuperPoint raw outputs to get keypoints and descriptors."""
    # Convert semi to heatmap
    semi = semi.squeeze().cpu().numpy()  # [65, H/8, W/8]
    desc = desc.squeeze().cpu().numpy()  # [256, H/8, W/8]
    
    # Get softmax probabilities
    dense = np.exp(semi)
    dense = dense / (np.sum(dense, axis=0) + 1e-6)
    
    # Remove dustbin (no-keypoint class)
    nodust = dense[:-1, :, :]  # [64, H/8, W/8]
    
    # Reshape to get keypoint probabilities per cell
    Hc, Wc = nodust.shape[1], nodust.shape[2]
    nodust = nodust.transpose(1, 2, 0)  # [H/8, W/8, 64]
    heatmap = np.reshape(nodust, [Hc, Wc, 8, 8])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc * 8, Wc * 8])  # [H, W]
    
    # Find keypoints above threshold
    xs, ys = np.where(heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 256))
        
    pts = np.zeros((len(xs), 2))
    pts[:, 0] = ys  # x coordinates
    pts[:, 1] = xs  # y coordinates
    scores = heatmap[xs, ys]
    
    # Apply simple NMS
    if nms_dist > 0 and len(pts) > 0:
        keep = nms_fast(pts, scores, nms_dist)
        pts = pts[keep]
        scores = scores[keep]
    
    # Extract descriptors at keypoint locations
    if len(pts) > 0:
        # Convert keypoint coordinates to descriptor map coordinates
        samp_pts = pts / 8  # Scale to descriptor map resolution
        samp_pts[:, [0, 1]] = samp_pts[:, [1, 0]]  # Swap x,y to y,x for indexing
        
        # Bilinear interpolation to get descriptors
        descriptors = sample_descriptors(desc, samp_pts)
    else:
        descriptors = np.zeros((0, 256))
        
    return pts, scores, descriptors

def nms_fast(pts, scores, nms_dist):
    """Fast non-maximum suppression."""
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

def sample_descriptors(desc, pts):
    """Sample descriptors using bilinear interpolation."""
    D, H, W = desc.shape
    descriptors = np.zeros((len(pts), D))
    
    for i, pt in enumerate(pts):
        y, x = pt[0], pt[1]
        
        # Clamp coordinates
        y = max(0, min(H-1, y))
        x = max(0, min(W-1, x))
        
        # Get integer coordinates
        y0, x0 = int(y), int(x)
        y1, x1 = min(H-1, y0+1), min(W-1, x0+1)
        
        # Bilinear weights
        wy = y - y0
        wx = x - x0
        
        # Interpolate
        desc_interp = (1-wy)*(1-wx)*desc[:, y0, x0] + \
                      (1-wy)*wx*desc[:, y0, x1] + \
                      wy*(1-wx)*desc[:, y1, x0] + \
                      wy*wx*desc[:, y1, x1]
        
        descriptors[i] = desc_interp
    
    return descriptors

def visualize_keypoints_on_crop(crop, keypoints, scores, title="SuperPoint"):
    """Visualize keypoints on a single crop."""
    if len(crop.shape) == 3:
        vis_crop = crop.copy()
    else:
        vis_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    
    # Draw keypoints with different colors based on confidence
    for i, (kpt, score) in enumerate(zip(keypoints, scores)):
        x, y = int(kpt[0]), int(kpt[1])
        
        # Color based on confidence: red for high confidence, blue for low
        confidence_normalized = min(1.0, score / 0.05)  # Normalize assuming max ~0.05
        color = (
            int(255 * confidence_normalized),  # R
            int(255 * (1 - confidence_normalized)),  # G  
            0  # B
        )
        
        cv2.circle(vis_crop, (x, y), 2, color, -1)
    
    return vis_crop

def main():
    """Main function to run SuperPoint visualization on crops."""
    
    # Paths
    weights_path = "third_party/pytorch-superpoint/logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar"
    imagery_dir = Path("imagery")
    
    # Check if weights exist
    if not Path(weights_path).exists():
        print(f"Error: SuperPoint weights not found at {weights_path}")
        return
    
    # Load SuperPoint model
    print(f"Loading SuperPoint with COCO weights...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = SuperPointNet_gauss2()
    model = model.to(device)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ SuperPoint weights loaded successfully")
    
    # Get image files
    image_files = list(imagery_dir.glob('*.png'))
    if len(image_files) == 0:
        print(f"No PNG images found in {imagery_dir}")
        return
    
    print(f"Found {len(image_files)} images in {imagery_dir}")
    
    # Sample 6 crops from random images
    crops = []
    source_files = []
    
    for i in range(6):
        attempts = 0
        crop = None
        
        while crop is None and attempts < 10:
            # Pick random image
            img_file = random.choice(image_files)
            img = cv2.imread(str(img_file))
            
            if img is not None:
                crop = sample_crop(img, crop_size=100)
                if crop is not None:
                    crops.append(crop)
                    source_files.append(img_file.name)
                    print(f"Crop {i+1}: sampled from {img_file.name}")
            
            attempts += 1
        
        if crop is None:
            print(f"Failed to sample crop {i+1} after 10 attempts")
            return
    
    # Process each crop with SuperPoint
    results = []
    
    for i, crop in enumerate(crops):
        print(f"Processing crop {i+1}/6...")
        
        # Convert to grayscale for SuperPoint
        if len(crop.shape) == 3:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = crop
        
        # Prepare input tensor
        inp = torch.from_numpy(gray_crop.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(inp)
            semi = output['semi']
            desc = output['desc']
        
        # Process outputs to get keypoints
        keypoints, scores, descriptors = process_superpoint_output(semi, desc, conf_thresh=0.01)
        
        print(f"  Found {len(keypoints)} keypoints")
        
        # Create visualization
        vis_crop = visualize_keypoints_on_crop(crop, keypoints, scores)
        
        results.append({
            'crop': crop,
            'vis_crop': vis_crop,
            'keypoints': keypoints,
            'scores': scores,
            'source_file': source_files[i],
            'n_keypoints': len(keypoints)
        })
    
    # Create final visualization with all 6 crops
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SuperPoint Keypoint Detection on 100x100 Crops (COCO Weights)', fontsize=16)
    
    for i, result in enumerate(results):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Convert BGR to RGB for matplotlib
        if len(result['vis_crop'].shape) == 3:
            vis_rgb = cv2.cvtColor(result['vis_crop'], cv2.COLOR_BGR2RGB)
        else:
            vis_rgb = result['vis_crop']
        
        ax.imshow(vis_rgb)
        ax.set_title(f"Crop {i+1}: {result['n_keypoints']} keypoints\n{result['source_file']}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = "superpoint_crops_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print summary statistics
    total_keypoints = sum(r['n_keypoints'] for r in results)
    avg_keypoints = total_keypoints / len(results)
    
    print(f"\n--- Summary ---")
    print(f"Total crops processed: {len(results)}")
    print(f"Total keypoints detected: {total_keypoints}")
    print(f"Average keypoints per crop: {avg_keypoints:.1f}")
    
    # Show individual crop statistics
    for i, result in enumerate(results):
        avg_score = np.mean(result['scores']) if len(result['scores']) > 0 else 0
        print(f"Crop {i+1}: {result['n_keypoints']} keypoints, avg confidence: {avg_score:.4f}")

if __name__ == "__main__":
    main() 