#!/usr/bin/env python3
"""
Generate UAV training data from earth imagery for SuperPoint fine-tuning.
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
        scale = max(crop_size / H, crop_size / W)
        new_H, new_W = int(H * scale), int(W * scale)
        image = cv2.resize(image, (new_W, new_H))
        H, W = new_H, new_W
    
    # Random crop location
    max_x = W - crop_size
    max_y = H - crop_size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    crop = image[y:y+crop_size, x:x+crop_size]
    
    # Random augmentations
    # 1. Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        center = (crop_size // 2, crop_size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        crop = cv2.warpAffine(crop, M, (crop_size, crop_size))
    
    # 2. Random brightness/contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.uniform(-20, 20)    # Brightness
        crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
    
    # 3. Random horizontal flip
    if random.random() < 0.5:
        crop = cv2.flip(crop, 1)
    
    return crop

def generate_uav_dataset(earth_imagery_dir, output_dir, n_crops_per_location=1000, crop_size=256):
    """Generate UAV dataset from earth imagery."""
    
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
        pbar = tqdm(total=n_crops_per_location, desc=f"Generating crops for {loc_dir.name}")
        
        while crop_count < n_crops_per_location:
            # Randomly select an image
            img_path = random.choice(image_files)
            
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Generate augmented crop
                crop = random_crop_with_augmentation(img, crop_size)
                
                # Check if crop has enough variation (not just empty space)
                if np.std(crop) < 10.0:  # Skip very uniform crops
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
    
    print(f"\n✓ Generated {total_crops} UAV crops")
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