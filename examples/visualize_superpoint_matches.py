import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from src.models.superpoint.superpoint_model import SuperPoint
from src.contrast_normalize import normalize

# --- CONFIG ---
IMAGE_PATH = "inference/46.6234, 32.7851.jpg"  # Fixed: space after comma
CROP_SIZE = 256  # pixels
NUM_EXAMPLES = 3
MAX_OFFSET = 64  # max offset in pixels for "close" crops

# Try different model paths in order of preference
import os
MODEL_PATHS = [
    "superpoint_training/checkpoints/superpoint_uav_final.pth",  # Fixed model (preferred)
]

# Find the first available model
MODEL_PATH = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    print("❌ No SuperPoint model found!")
    print("Available paths checked:")
    for path in MODEL_PATHS:
        print(f"  - {path}")
    print("\nPlease run the training pipeline first.")
    exit(1)

# --- Load SuperPoint model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SuperPoint model from: {MODEL_PATH}")
try:
    model = SuperPoint(pretrained_path=MODEL_PATH, device=device)
    print("✓ SuperPoint model loaded successfully")
except Exception as e:
    print(f"✗ Error loading SuperPoint model: {e}")
    exit(1)

# --- Load large location image ---
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}. Please check the file path and ensure the file exists.")
H, W = img.shape
print(f"Loaded image with dimensions: {H}x{W}")

# --- Visualization loop ---
for i in range(NUM_EXAMPLES):
    print(f"\n=== Processing example {i+1}/{NUM_EXAMPLES} ===")
    
    # Randomly sample top-left corners for two crops that are close
    x1 = random.randint(0, W - CROP_SIZE - MAX_OFFSET)
    y1 = random.randint(0, H - CROP_SIZE - MAX_OFFSET)
    dx = random.randint(-MAX_OFFSET, MAX_OFFSET)
    dy = random.randint(-MAX_OFFSET, MAX_OFFSET)
    x2 = np.clip(x1 + dx, 0, W - CROP_SIZE)
    y2 = np.clip(y1 + dy, 0, H - CROP_SIZE)

    crop1 = img[y1:y1+CROP_SIZE, x1:x1+CROP_SIZE]
    crop2 = img[y2:y2+CROP_SIZE, x2:x2+CROP_SIZE]

    print(f"Crop 1 location: ({x1}, {y1})")
    print(f"Crop 2 location: ({x2}, {y2})")
    print(f"Offset: ({dx}, {dy}) pixels")

    # Apply contrast normalization to crops
    normalized_crop1 = normalize(crop1)
    normalized_crop2 = normalize(crop2)

    # --- SuperPoint inference ---
    try:
        kpts1, scores1, desc1 = model.detect(normalized_crop1)
        kpts2, scores2, desc2 = model.detect(normalized_crop2)
        
        print(f"SuperPoint detected {len(kpts1)} keypoints in normalized crop 1")
        print(f"SuperPoint detected {len(kpts2)} keypoints in normalized crop 2")
        
        if len(kpts1) > 0:
            print(f"Normalized Crop 1 - Keypoint scores: min={scores1.min():.3f}, max={scores1.max():.3f}, mean={scores1.mean():.3f}")
        if len(kpts2) > 0:
            print(f"Normalized Crop 2 - Keypoint scores: min={scores2.min():.3f}, max={scores2.max():.3f}, mean={scores2.mean():.3f}")
        
    except Exception as e:
        print(f"✗ Error during SuperPoint inference: {e}")
        continue

    # --- Analyze descriptors ---
    matches = []
    if len(desc1) > 0 and len(desc2) > 0:
        print(f"\n--- SuperPoint Descriptor Analysis ---")
        print(f"Descriptor shapes: {desc1.shape} vs {desc2.shape}")
        print(f"Descriptor range crop1: [{desc1.min():.3f}, {desc1.max():.3f}]")
        print(f"Descriptor range crop2: [{desc2.min():.3f}, {desc2.max():.3f}]")
        print(f"Descriptor mean crop1: {desc1.mean():.3f}, std: {desc1.std():.3f}")
        print(f"Descriptor mean crop2: {desc2.mean():.3f}, std: {desc2.std():.3f}")
        
        # Check if descriptors are normalized
        desc1_norms = np.linalg.norm(desc1, axis=1)
        desc2_norms = np.linalg.norm(desc2, axis=1)
        print(f"Descriptor norms - Crop1: mean={desc1_norms.mean():.3f}, std={desc1_norms.std():.3f}")
        print(f"Descriptor norms - Crop2: mean={desc2_norms.mean():.3f}, std={desc2_norms.std():.3f}")
        
        # Check if all descriptors are zeros (the original problem)
        if np.allclose(desc1, 0) or np.allclose(desc2, 0):
            print("⚠️  WARNING: Descriptors are all zeros! Model not trained properly.")
        else:
            print("✓ Descriptors contain non-zero values")
        
        # Compute pairwise distances
        dists = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)
        print(f"Descriptor distances: min={dists.min():.3f}, max={dists.max():.3f}, mean={dists.mean():.3f}")
        
        # Find best matches using Hungarian algorithm for one-to-one matching
        if not np.allclose(desc1, 0) and not np.allclose(desc2, 0):
            row_indices, col_indices = linear_sum_assignment(dists)
            
            # Filter matches by distance threshold
            distance_threshold = 1.0
            for i1, i2 in zip(row_indices, col_indices):
                distance = dists[i1, i2]
                if distance < distance_threshold:
                    matches.append((i1, i2, distance))
            
            matches.sort(key=lambda x: x[2])  # Sort by distance
            print(f"Found {len(matches)} good matches (distance < {distance_threshold})")
            
            if matches:
                print(f"Best match distance: {matches[0][2]:.3f}")
                print(f"Worst match distance: {matches[-1][2]:.3f}")
        else:
            print("Skipping matching due to zero descriptors")
    
    elif len(desc1) == 0 and len(desc2) == 0:
        print("No keypoints detected in either crop")
    elif len(desc1) == 0:
        print("No keypoints detected in crop 1")
    elif len(desc2) == 0:
        print("No keypoints detected in crop 2")

    # --- Visualization ---
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Stack crops side by side
    vis = np.concatenate([normalized_crop1, normalized_crop2], axis=1)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    offset = np.array([CROP_SIZE, 0])

    # Draw keypoints
    for idx, kpt in enumerate(kpts1):
        pt = tuple(np.round(kpt).astype(int))
        is_matched = any(i1 == idx for i1, _, _ in matches)
        color = (0, 255, 0) if is_matched else (255, 255, 255)  # Green if matched, white if not
        cv2.circle(vis, pt, 4, color, 2)
        cv2.putText(vis, str(idx), (pt[0]-8, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    for idx, kpt in enumerate(kpts2):
        pt = tuple(np.round(kpt).astype(int) + offset)
        is_matched = any(i2 == idx for _, i2, _ in matches)
        color = (0, 255, 0) if is_matched else (255, 255, 255)  # Green if matched, white if not
        cv2.circle(vis, pt, 4, color, 2)
        cv2.putText(vis, str(idx), (pt[0]-8, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw matches
    for i1, i2, distance in matches:
        pt1 = tuple(np.round(kpts1[i1]).astype(int))
        pt2 = tuple(np.round(kpts2[i2]).astype(int) + offset)
        
        # Color code by match quality: green = good, yellow = medium, red = poor
        if distance < 0.5:
            color = (0, 255, 0)  # Green - excellent match
        elif distance < 0.8:
            color = (0, 255, 255)  # Yellow - good match
        else:
            color = (0, 0, 255)  # Red - poor match
        
        cv2.line(vis, pt1, pt2, color, 2)

    ax.imshow(vis[..., ::-1])  # Convert BGR to RGB for matplotlib
    ax.set_title(f"SuperPoint Results - Example {i+1}\n"
                f"Keypoints: {len(kpts1)} + {len(kpts2)} = {len(kpts1) + len(kpts2)} | "
                f"Matches: {len(matches)} | "
                f"Offset: ({dx}, {dy})px\n"
                f"Green=matched keypoints, White=unmatched | "
                f"Line colors: Green=excellent, Yellow=good, Red=poor")
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"superpoint_contrast_norm_example_{i+1}.png", dpi=150, bbox_inches='tight')
    plt.show()

print("\n=== SuperPoint Analysis Complete ===")
print("Check the generated images to see SuperPoint's actual performance with contrast normalization.")
print("If descriptors are all zeros, the model needs proper training with the fixed descriptor loss.")
