#!/usr/bin/env python3
"""
Generate UAV training data from earth imagery for SuperPoint fine-tuning.
FIXED: Better quality checks and augmentations.
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import argparse

def random_crop_with_augmentation(image, crop_size=256):
    """Extract a random augmented crop from a large aerial image."""
    H, W = image.shape[:2]
    
    if H < crop_size or W < crop_size:
        # Resize if image is too small
        scale = max(crop_size / H, crop_size / W) * 1.1  # Add 10% margin
        new_H, new_W = int(H * scale), int(W * scale)
        image = cv2.resize(image, (new_W, new_H))
        H, W = new_H, new_W
    
    # Random crop location
    max_x = W - crop_size
    max_y = H - crop_size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    crop = image[y:y+crop_size, x:x+crop_size]
    
    # FIXED: More diverse augmentations
    # 1. Random rotation (smaller range for aerial imagery)
    if random.random() < 0.6:  # Increased probability
        angle = random.uniform(-10, 10)  # Smaller range for aerial
        center = (crop_size // 2, crop_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        crop = cv2.warpAffine(crop, M, (crop_size, crop_size))
    
    # 2. Random brightness/contrast (more conservative)
    if random.random() < 0.7:
        alpha = random.uniform(0.85, 1.15)  # Contrast (more conservative)
        beta = random.uniform(-15, 15)      # Brightness (more conservative)
        crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
    
    # 3. Random horizontal flip
    if random.random() < 0.5:
        crop = cv2.flip(crop, 1)
    
    # 4. ADDED: Random vertical flip (valid for aerial imagery)
    if random.random() < 0.3:
        crop = cv2.flip(crop, 0)
    
    # 5. ADDED: Random Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(2, 8), crop.shape).astype(np.int16)
        crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 6. ADDED: Random blur (simulate altitude/atmospheric effects)
    if random.random() < 0.2:
        kernel_size = random.choice([3, 5])
        crop = cv2.GaussianBlur(crop, (kernel_size, kernel_size), 0)
    
    return crop

def is_good_crop(crop, min_std=15.0, min_gradient=10.0):
    """
    FIXED: Better quality assessment for crops.
    
    Args:
        crop: Image crop
        min_std: Minimum standard deviation (texture)
        min_gradient: Minimum gradient magnitude (edges/features)
    
    Returns:
        bool: True if crop has sufficient quality
    """
    # Check standard deviation (texture)
    if np.std(crop) < min_std:
        return False
    
    # Check gradient magnitude (edges/features)
    grad_x = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    if np.mean(grad_mag) < min_gradient:
        return False
    
    # Check for too much saturation (all white/black regions)
    hist = cv2.calcHist([crop], [0], None, [256], [0, 256])
    if hist[0] > crop.size * 0.5 or hist[255] > crop.size * 0.5:  # >50% pure black or white
        return False
    
    return True

def generate_uav_dataset(earth_imagery_dir, output_dir, n_crops_per_location=1000, crop_size=256):
    """Generate UAV dataset from earth imagery with better quality control."""
    
    earth_imagery_dir = Path(earth_imagery_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all location directories
    loc_dirs = sorted([d for d in earth_imagery_dir.iterdir() if d.is_dir() and d.name.startswith('loc')])
    
    if not loc_dirs:
        print(f"No location directories found in {earth_imagery_dir}")
        return
    
    print(f"Found {len(loc_dirs)} location directories")
    
    total_crops = 0
    total_rejected = 0
    
    for loc_dir in loc_dirs:
        print(f"\nProcessing {loc_dir.name}...")
        
        # Find all image files in this location
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
            image_files.extend(loc_dir.glob(ext))
        
        if not image_files:
            print(f"No images found in {loc_dir}")
            continue
        
        print(f"Found {len(image_files)} images")
        
        # Generate crops for this location
        crop_count = 0
        attempts = 0
        max_attempts = n_crops_per_location * 3  # Allow more attempts
        
        pbar = tqdm(total=n_crops_per_location, desc=f"Generating crops for {loc_dir.name}")
        
        while crop_count < n_crops_per_location and attempts < max_attempts:
            attempts += 1
            
            # Randomly select an image
            img_path = random.choice(image_files)
            
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Generate augmented crop
                crop = random_crop_with_augmentation(img, crop_size)
                
                # FIXED: Better quality check
                if not is_good_crop(crop):
                    total_rejected += 1
                    continue
                
                # Save the crop
                crop_filename = f"{loc_dir.name}_crop_{crop_count:06d}.png"
                crop_path = output_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)
                
                crop_count += 1
                total_crops += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        pbar.close()
        
        if crop_count < n_crops_per_location:
            print(f"⚠️ Only generated {crop_count}/{n_crops_per_location} crops for {loc_dir.name}")
    
    print(f"\n✓ Generated {total_crops} UAV crops")
    print(f"✓ Rejected {total_rejected} low-quality crops")
    print(f"✓ Quality rate: {total_crops/(total_crops + total_rejected)*100:.1f}%")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate UAV training data")
    parser.add_argument('--earth_imagery_dir', type=str, default='data/earth_imagery',
                       help='Directory containing earth imagery (loc1, loc2, etc.)')
    parser.add_argument('--output_dir', type=str, default='uav_data',
                       help='Output directory for UAV crops')
    parser.add_argument('--n_crops_per_location', type=int, default=1000,
                       help='Number of crops to generate per location')
    parser.add_argument('--crop_size', type=int, default=256,
                       help='Size of crops (width=height)')
    
    args = parser.parse_args()
    
    generate_uav_dataset(
        args.earth_imagery_dir,
        args.output_dir,
        args.n_crops_per_location,
        args.crop_size
    ) 