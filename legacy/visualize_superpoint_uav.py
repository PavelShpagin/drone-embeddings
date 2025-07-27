#!/usr/bin/env python3
"""
SuperPoint visualization using UAV-trained weights on 256x256 grayscale crops.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from simple_superpoint import SuperPoint

def sample_crop(image, crop_size=256):
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

def visualize_keypoints_on_crop(crop, keypoints, scores, title="SuperPoint UAV"):
    """Visualize keypoints on a single crop."""
    if len(crop.shape) == 3:
        vis_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    else:
        vis_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    
    # Draw keypoints with different colors based on confidence
    for i, (kpt, score) in enumerate(zip(keypoints, scores)):
        x, y = int(kpt[0]), int(kpt[1])
        
        # Color based on confidence: red for high confidence, green for medium, blue for low
        if score > 0.03:
            color = (255, 0, 0)  # Red for high confidence
        elif score > 0.02:
            color = (255, 165, 0)  # Orange for medium confidence  
        else:
            color = (0, 100, 255)  # Blue for low confidence
        
        cv2.circle(vis_crop, (x, y), 3, color, -1)
        # Add small circle outline for better visibility
        cv2.circle(vis_crop, (x, y), 3, (255, 255, 255), 1)
    
    return vis_crop

def main():
    """Main function to run SuperPoint UAV visualization on crops."""
    
    # Paths
    weights_path = "superpoint_uav_trained/superpoint_uav_final.pth"
    imagery_dir = Path("imagery")
    
    # Check if weights exist
    if not Path(weights_path).exists():
        print(f"Error: SuperPoint UAV weights not found at {weights_path}")
        return
    
    # Load SuperPoint with UAV weights
    print(f"Loading SuperPoint with UAV-trained weights...")
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    superpoint = SuperPoint(weights_path, device=device)
    
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
                crop = sample_crop(img, crop_size=256)
                if crop is not None:
                    crops.append(crop)
                    source_files.append(img_file.name)
                    print(f"Crop {i+1}: sampled 256x256 from {img_file.name}")
            
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
        
        # Detect keypoints with SuperPoint
        keypoints, scores, descriptors = superpoint.detect(
            gray_crop, 
            conf_thresh=0.01,  # Lower threshold to get more keypoints
            nms_dist=4
        )
        
        print(f"  Found {len(keypoints)} keypoints")
        
        # Create visualization
        vis_crop = visualize_keypoints_on_crop(gray_crop, keypoints, scores)
        
        results.append({
            'crop': gray_crop,
            'vis_crop': vis_crop,
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'source_file': source_files[i],
            'n_keypoints': len(keypoints)
        })
    
    # Create final visualization with all 6 crops
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SuperPoint Keypoint Detection - UAV Trained Model (256x256 Grayscale Crops)', fontsize=16)
    
    for i, result in enumerate(results):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.imshow(result['vis_crop'])
        
        # Calculate confidence statistics
        scores = result['scores']
        if len(scores) > 0:
            avg_conf = np.mean(scores)
            max_conf = np.max(scores)
            title_text = f"Crop {i+1}: {result['n_keypoints']} keypoints\nAvg: {avg_conf:.3f}, Max: {max_conf:.3f}\n{result['source_file'][:20]}..."
        else:
            title_text = f"Crop {i+1}: {result['n_keypoints']} keypoints\n{result['source_file'][:20]}..."
            
        ax.set_title(title_text, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = "superpoint_uav_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print detailed summary statistics
    total_keypoints = sum(r['n_keypoints'] for r in results)
    avg_keypoints = total_keypoints / len(results)
    
    print(f"\n--- SuperPoint UAV Model Summary ---")
    print(f"Model: simple_superpoint.py (Original SuperPointNet)")
    print(f"Weights: UAV-trained ({weights_path})")
    print(f"Crop size: 256x256 pixels (grayscale)")
    print(f"Total crops processed: {len(results)}")
    print(f"Total keypoints detected: {total_keypoints}")
    print(f"Average keypoints per crop: {avg_keypoints:.1f}")
    
    # Show individual crop statistics
    print(f"\n--- Individual Crop Results ---")
    for i, result in enumerate(results):
        scores = result['scores']
        if len(scores) > 0:
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            std_score = np.std(scores)
            print(f"Crop {i+1}: {result['n_keypoints']:3d} keypoints | "
                  f"Conf - avg: {avg_score:.4f}, max: {max_score:.4f}, min: {min_score:.4f}, std: {std_score:.4f}")
        else:
            print(f"Crop {i+1}: {result['n_keypoints']:3d} keypoints | No confidence scores")
    
    # Comparison info
    print(f"\n--- Legend ---")
    print(f"🔴 Red circles: High confidence (> 0.03)")
    print(f"🟠 Orange circles: Medium confidence (0.02-0.03)")
    print(f"🔵 Blue circles: Low confidence (< 0.02)")

if __name__ == "__main__":
    main() 