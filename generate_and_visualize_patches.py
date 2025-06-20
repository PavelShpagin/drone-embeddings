import cv2
import numpy as np
from pathlib import Path
import subprocess
import random
import argparse

# Define the base image path
base_img_path = "data/earth_imagery/loc1/45.3395, 29.6287.autumn.jpg"
output_dir = Path('./') # Save temporary images in current directory

# Define crop parameters
crop_size = 256 # pixels (representing 100x100m patch)
min_offset_pixels = 10
max_offset_pixels = 50

def generate_and_visualize(base_img, iteration, weights_path, rotation_angle):
    H, W = base_img.shape
    
    # Randomly select a top-left corner for the first crop
    max_x1 = W - crop_size - max_offset_pixels
    max_y1 = H - crop_size - max_offset_pixels
    
    if max_x1 < 0 or max_y1 < 0:
        print(f"Warning: Image too small for desired crops and offsets in iteration {iteration}. Skipping.")
        return

    x1 = random.randint(0, max_x1)
    y1 = random.randint(0, max_y1)
    
    # Extract the first patch
    patch1 = base_img[y1:y1 + crop_size, x1:x1 + crop_size]
    
    # Generate a random offset for the second patch (10-50 pixels)
    offset_x = random.randint(min_offset_pixels, max_offset_pixels)
    offset_y = random.randint(min_offset_pixels, max_offset_pixels)
    
    # Randomly decide direction of offset
    if random.random() < 0.5: offset_x *= -1
    if random.random() < 0.5: offset_y *= -1
    
    # Calculate top-left corner for the second patch
    x2 = x1 + offset_x
    y2 = y1 + offset_y
    
    # Ensure the second patch is within bounds, adjust if necessary
    x2 = np.clip(x2, 0, W - crop_size)
    y2 = np.clip(y2, 0, H - crop_size)
    
    # Re-extract patch2 to ensure it's within valid bounds
    patch2 = base_img[y2:y2 + crop_size, x2:x2 + crop_size]
    
    print(f"Iteration {iteration}: Patch 1 (top-left): ({x1}, {y1}), Patch 2 (top-left): ({x2}, {y2}), relative offset: ({x2-x1}, {y2-y1})")
    
    # Save temporary patches
    temp_patch1_path = output_dir / 'temp_patch1.png'
    temp_patch2_path = output_dir / 'temp_patch2.png'
    cv2.imwrite(str(temp_patch1_path), patch1)
    cv2.imwrite(str(temp_patch2_path), patch2)
    
    # Run visualization script
    output_filename = f"superpoint_patches_distant_matches_{iteration:02d}.png"
    command = [
        'python3', 'visualize_superpoint_clean.py',
        '--weights', weights_path,
        '--img1', str(temp_patch1_path),
        '--img2', str(temp_patch2_path),
        '--output', str(output_dir / output_filename),
        '--rotate_img2_angle', str(rotation_angle)
    ]
    print("Running visualization command:", ' '.join(command))
    subprocess.run(command, check=True)
    
    print(f"Visualization saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize SuperPoint matches on patches.")
    parser.add_argument('--n_examples', type=int, default=1, help='Number of examples to generate.')
    parser.add_argument('--weights', type=str, default='superpoint_uav_trained/superpoint_uav_final.pth',
                        help='Path to SuperPoint weights.')
    parser.add_argument('--rotation_angle', type=float, default=0,
                       help='Angle in degrees to rotate img2 for testing (0 for no rotation).')
    args = parser.parse_args()

    # Load the base image once
    print(f"Loading base image: {base_img_path}")
    img = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {base_img_path}")
        exit(1)
    
    H, W = img.shape
    print(f"Base image shape: {H}x{W}")
    
    # Ensure the image is large enough for cropping and max offset
    if H < crop_size + max_offset_pixels or W < crop_size + max_offset_pixels:
        print(f"Error: Image too small for desired crops and offsets. Image size: {H}x{W}, required minimum: {crop_size + max_offset_pixels}x{crop_size + max_offset_pixels}")
        exit(1)

    for i in range(args.n_examples):
        generate_and_visualize(img, i + 1, args.weights, args.rotation_angle)

    print(f"Generated {args.n_examples} visualizations.") 