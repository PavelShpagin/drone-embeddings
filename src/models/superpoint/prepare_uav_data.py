import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import argparse
from glob import glob

def random_crop_with_augmentation(
    image, 
    crop_size=256, 
    angle_range=(-180, 180),
    scale_range=(0.8, 1.2),
    brightness_range=(0.8, 1.2),
    p_flip=0.5
):
    """Extract a random augmented crop from a large aerial image."""
    H, W = image.shape[:2]
    
    # Add padding to ensure we can rotate without losing corners
    pad_size = int(np.ceil(crop_size * np.sqrt(2) * scale_range[1]))
    pad = (pad_size - crop_size) // 2
    
    # Get valid crop center points (accounting for padding)
    valid_h = range(pad_size, H - pad_size)
    valid_w = range(pad_size, W - pad_size)
    
    if not valid_h or not valid_w:
        return None
        
    # Random center point
    center_h = random.choice(valid_h)
    center_w = random.choice(valid_w)
    
    # Extract larger patch to allow for rotation and scaling
    patch = image[center_h-pad_size:center_h+pad_size, center_w-pad_size:center_w+pad_size]
    if patch.shape[0] != 2*pad_size or patch.shape[1] != 2*pad_size:
        return None
    
    # Random rotation
    angle = random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D((pad_size, pad_size), angle, 1.0)
    rotated = cv2.warpAffine(patch, M, (2*pad_size, 2*pad_size))
    
    # Random scaling
    scale = random.uniform(*scale_range)
    scaled_size = int(crop_size * scale)
    scaled = cv2.resize(rotated, (scaled_size, scaled_size))
    
    # Center crop to target size
    if scaled_size > crop_size:
        start = (scaled_size - crop_size) // 2
        crop = scaled[start:start+crop_size, start:start+crop_size]
    else:
        # Pad if scaled size is smaller
        pad_width = (crop_size - scaled_size) // 2
        crop = cv2.copyMakeBorder(scaled, pad_width, pad_width, pad_width, pad_width, 
                                cv2.BORDER_CONSTANT, value=0)
        crop = crop[:crop_size, :crop_size]  # Ensure exact size
    
    # Random brightness adjustment
    brightness = random.uniform(*brightness_range)
    crop = cv2.multiply(crop, brightness, dtype=cv2.CV_8U)
    
    # Random horizontal/vertical flip
    if random.random() < p_flip:
        crop = cv2.flip(crop, 1)  # horizontal flip
    if random.random() < p_flip:
        crop = cv2.flip(crop, 0)  # vertical flip
        
    return crop

def prepare_uav_dataset(
    earth_imagery_dir,
    output_dir,
    n_crops_per_location=5000,
    crop_size=256,
    min_std=20.0  # Minimum standard deviation to ensure we don't get empty crops
):
    """
    Prepare UAV dataset by extracting random augmented crops from large aerial images.
    
    Args:
        earth_imagery_dir: Base directory containing loc1, loc2, etc.
        output_dir: Where to save the crops
        n_crops_per_location: Number of crops to generate per location
        crop_size: Size of crops (will be crop_size x crop_size)
        min_std: Minimum standard deviation for a crop to be considered valid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all location directories
    loc_dirs = sorted(glob(os.path.join(earth_imagery_dir, "loc*")))
    if not loc_dirs:
        raise ValueError(f"No location directories found in {earth_imagery_dir}")
    
    print(f"Found {len(loc_dirs)} locations")
    
    total_crops = 0
    
    for loc_dir in loc_dirs:
        loc_name = os.path.basename(loc_dir)
        print(f"\nProcessing {loc_name}...")
        
        # Get all image files in this location
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_files.extend(glob(os.path.join(loc_dir, ext)))
        
        if not image_files:
            print(f"No images found in {loc_dir}, skipping...")
            continue
            
        print(f"Found {len(image_files)} source images")
        
        # Create output directory for this location
        loc_output_dir = output_dir / loc_name
        loc_output_dir.mkdir(exist_ok=True)
        
        # Generate crops for this location
        crop_count = 0
        pbar = tqdm(total=n_crops_per_location, desc=f"Generating crops for {loc_name}")
        
        while crop_count < n_crops_per_location:
            # Randomly select an image
            img_path = random.choice(image_files)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # Generate a random augmented crop
            crop = random_crop_with_augmentation(
                img,
                crop_size=crop_size,
                angle_range=(-180, 180),
                scale_range=(0.8, 1.2),
                brightness_range=(0.8, 1.2),
                p_flip=0.5
            )
            
            if crop is None:
                continue
                
            # Check if crop has enough variation (not just empty space)
            if np.std(crop) < min_std:
                continue
                
            # Save the crop
            out_path = loc_output_dir / f"crop_{crop_count:06d}.png"
            cv2.imwrite(str(out_path), crop)
            
            crop_count += 1
            total_crops += 1
            pbar.update(1)
        
        pbar.close()
    
    print(f"\nTotal crops generated: {total_crops}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--earth_imagery_dir', type=str, required=True,
                      help='Base directory containing loc1, loc2, etc.')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Where to save the crops')
    parser.add_argument('--n_crops_per_location', type=int, default=5000,
                      help='Number of crops to generate per location')
    parser.add_argument('--crop_size', type=int, default=256,
                      help='Size of crops (width=height)')
    args = parser.parse_args()
    
    prepare_uav_dataset(
        args.earth_imagery_dir,
        args.output_dir,
        args.n_crops_per_location,
        args.crop_size
    ) 