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

def get_drone_crop(image_path, crop_size=256):
    """Get a crop from drone image"""
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)
    
    h, w = img_np.shape
    if h < crop_size or w < crop_size:
        # If image is smaller, resize it first
        scale = max(crop_size / h, crop_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_pil = Image.fromarray(img_np).resize((new_w, new_h), Image.LANCZOS)
        img_np = np.array(img_pil)
        h, w = img_np.shape
    
    # Get center crop
    y = (h - crop_size) // 2
    x = (w - crop_size) // 2
    
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
    ]

    # Get a list of all Earth imagery paths
    earth_image_paths = []
    for loc_dir in loc_dirs:
        earth_image_paths.extend(list(Path(loc_dir).glob("*.jpg")))
    
    # Drone image path
    drone_image_path = Path("real_data/00204144.jpg")
    
    # Prepare samples: 1 drone image + 2 Earth imagery crops
    samples = []
    
    # Add drone image
    if drone_image_path.exists():
        try:
            drone_crop = get_drone_crop(drone_image_path, crop_size=256)
            samples.append(("Drone Image", drone_crop))
        except Exception as e:
            print(f"Error processing drone image: {e}")
    
    # Add Earth imagery samples
    if earth_image_paths:
        selected_earth_paths = random.sample(earth_image_paths, min(2, len(earth_image_paths)))
        for img_path in selected_earth_paths:
            try:
                earth_crop = get_random_crop(img_path, crop_size=256)
                samples.append((f"Earth ({img_path.parent.name})", earth_crop))
            except ValueError as e:
                print(f"Skipping {img_path}: {e}")
                continue

    if not samples:
        print("No valid images found to process.")
        return

    # Create visualization: 4 columns (Original, Contrast Norm, Standard Norm after Contrast, Direct Standard Norm)
    fig, axes = plt.subplots(len(samples), 4, figsize=(20, 5 * len(samples)))
    if len(samples) == 1:
        axes = [axes] # Ensure axes is always a 2D array for consistent indexing

    for i, (sample_name, original_crop) in enumerate(samples):
        # 1. Original Image
        axes[i, 0].imshow(original_crop, cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f"Original\n({sample_name})")
        axes[i, 0].axis('off')

        # 2. Contrast Normalization (blur_map_iters=0)
        contrast_normalized_crop = normalize(original_crop, contrasting=1, blur_map_iters=0)
        axes[i, 1].imshow(contrast_normalized_crop, cmap='gray', vmin=0, vmax=255)
        axes[i, 1].set_title("Contrast Normalized\n(blur_iters=0)")
        axes[i, 1].axis('off')

        # 3. Standard Normalization after Contrast (Pipeline 1: Original → Contrast → Standard)
        standard_after_contrast = (contrast_normalized_crop.astype(np.float32) / 255.0)
        standard_after_contrast = (standard_after_contrast * 255).astype(np.uint8)
        axes[i, 2].imshow(standard_after_contrast, cmap='gray', vmin=0, vmax=255)
        axes[i, 2].set_title("Pipeline 1:\nContrast → Standard")
        axes[i, 2].axis('off')

        # 4. Direct Standard Normalization (Pipeline 2: Original → Standard)
        direct_standard = (original_crop.astype(np.float32) / 255.0)
        direct_standard = (direct_standard * 255).astype(np.uint8)
        axes[i, 3].imshow(direct_standard, cmap='gray', vmin=0, vmax=255)
        axes[i, 3].set_title("Pipeline 2:\nDirect Standard")
        axes[i, 3].axis('off')

        # Print some statistics
        print(f"\n{sample_name} Statistics:")
        print(f"  Original: mean={np.mean(original_crop):.1f}, std={np.std(original_crop):.1f}")
        print(f"  Contrast Normalized: mean={np.mean(contrast_normalized_crop):.1f}, std={np.std(contrast_normalized_crop):.1f}")
        print(f"  Pipeline 1 (Contrast→Standard): mean={np.mean(standard_after_contrast):.1f}, std={np.std(standard_after_contrast):.1f}")
        print(f"  Pipeline 2 (Direct Standard): mean={np.mean(direct_standard):.1f}, std={np.std(direct_standard):.1f}")

    plt.tight_layout()
    plt.savefig("pipeline_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nPipeline comparison saved to pipeline_comparison.png")
    print("Comparing:")
    print("  Pipeline 1: Original → Contrast Normalization → Standard Normalization")
    print("  Pipeline 2: Original → Direct Standard Normalization")

if __name__ == '__main__':
    visualize_normalization() 