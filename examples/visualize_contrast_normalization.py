import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from pathlib import Path
import random

from src.contrast_normalize import normalize

def get_random_crop(image_path, crop_size=256):
    img = Image.open(image_path).convert("L") # Load as grayscale
    img_np = np.array(img)

    h, w = img_np.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image {image_path} is too small for crop size {crop_size}")

    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)

    crop = img_np[y:y+crop_size, x:x+crop_size]
    return crop

def visualize_normalization():
    # Define paths to the image directories
    loc_dirs = [
        "data/earth_imagery/loc1",
        "data/earth_imagery/loc2",
        "data/earth_imagery/loc3",
        "data/earth_imagery/loc4",
        "data/earth_imagery/loc5",
        "data/earth_imagery/loc6",
        "data/earth_imagery/loc7",
        "data/earth_imagery/loc8",
        "data/earth_imagery/loc9",
        "data/earth_imagery/loc10",
    ]

    # Get a list of all image paths
    all_image_paths = []
    for loc_dir in loc_dirs:
        all_image_paths.extend(list(Path(loc_dir).glob("*.jpg")))
    
    if not all_image_paths:
        print("No images found in the specified data directories.")
        return

    # Select a few random images to demonstrate
    num_samples = 3
    selected_image_paths = random.sample(all_image_paths, min(num_samples, len(all_image_paths)))

    fig, axes = plt.subplots(len(selected_image_paths), 3, figsize=(15, 5 * len(selected_image_paths)))
    if len(selected_image_paths) == 1:
        axes = [axes] # Ensure axes is always a 2D array for consistent indexing

    for i, img_path in enumerate(selected_image_paths):
        try:
            original_crop = get_random_crop(img_path, crop_size=256)
        except ValueError as e:
            print(f"Skipping {img_path}: {e}")
            continue

        # 1. Original Image
        axes[i, 0].imshow(original_crop, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f"Original ({img_path.name})")
        axes[i, 0].axis('off')

        # 2. Standard Normalization (scale to 0-255 for display)
        # Convert to float [0, 1], then normalize, then back to uint8 [0, 255]
        standard_normalized_crop = (original_crop.astype(np.float32) / 255.0)
        standard_normalized_crop = (standard_normalized_crop * 255).astype(np.uint8) # No actual contrast norm here, just scaling

        axes[i, 1].imshow(standard_normalized_crop, cmap='gray', vmin=0, vmax=255)
        axes[i, 1].set_title("Standard Normalized")
        axes[i, 1].axis('off')

        # 3. Custom Contrast Normalization
        contrast_normalized_crop = normalize(original_crop, contrasting=1, blur_map_iters=1)
        
        axes[i, 2].imshow(contrast_normalized_crop, cmap='gray', vmin=0, vmax=255)
        axes[i, 2].set_title("Contrast Normalized")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("contrast_normalization_comparison.png")
    print("Comparison saved to contrast_normalization_comparison.png")

if __name__ == '__main__':
    visualize_normalization() 