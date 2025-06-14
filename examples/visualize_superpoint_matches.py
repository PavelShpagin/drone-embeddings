import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from src.models.superpoint.superpoint_model import SuperPoint

# --- CONFIG ---
IMAGE_PATH = "inference/46.6234, 32.7851.jpg"  # Fixed: space after comma
MODEL_PATH = "superpoint_training/superpoint_fixed/final.pth"
CROP_SIZE = 256  # pixels
NUM_EXAMPLES = 3  # Reduced for debugging
MAX_OFFSET = 64  # max offset in pixels for "close" crops

# --- Load SuperPoint model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SuperPoint(pretrained_path=MODEL_PATH, device=device)

# --- Load large location image ---
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}. Please check the file path and ensure the file exists.")
H, W = img.shape
print(f"Loaded image with dimensions: {H}x{W}")

# --- Initialize SIFT for comparison ---
sift = cv2.SIFT_create()

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

    # --- SuperPoint inference ---
    kpts1, scores1, desc1 = model.detect(crop1)
    kpts2, scores2, desc2 = model.detect(crop2)
    
    print(f"SuperPoint - Crop 1: {len(kpts1)} keypoints, Crop 2: {len(kpts2)} keypoints")
    
    # --- SIFT inference for comparison ---
    sift_kpts1, sift_desc1 = sift.detectAndCompute(crop1, None)
    sift_kpts2, sift_desc2 = sift.detectAndCompute(crop2, None)
    
    print(f"SIFT - Crop 1: {len(sift_kpts1)} keypoints, Crop 2: {len(sift_kpts2)} keypoints")

    # --- Analyze SuperPoint descriptors ---
    if len(desc1) > 0 and len(desc2) > 0:
        print(f"\n--- SuperPoint Descriptor Analysis ---")
        print(f"Descriptor shape: {desc1.shape}")
        print(f"Descriptor range: [{desc1.min():.3f}, {desc1.max():.3f}]")
        print(f"Descriptor mean: {desc1.mean():.3f}, std: {desc1.std():.3f}")
        
        # Check if descriptors are normalized
        desc1_norms = np.linalg.norm(desc1, axis=1)
        desc2_norms = np.linalg.norm(desc2, axis=1)
        print(f"Descriptor norms - Crop1: mean={desc1_norms.mean():.3f}, std={desc1_norms.std():.3f}")
        print(f"Descriptor norms - Crop2: mean={desc2_norms.mean():.3f}, std={desc2_norms.std():.3f}")
        
        # Check descriptor diversity (are they all similar?)
        desc1_pairwise = np.dot(desc1, desc1.T)
        desc1_self_sim = desc1_pairwise[np.triu_indices_from(desc1_pairwise, k=1)]
        print(f"Descriptor self-similarity in Crop1: mean={desc1_self_sim.mean():.3f}, std={desc1_self_sim.std():.3f}")
        
        # Compute distances
        sp_dists = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)
        print(f"SuperPoint distances: min={sp_dists.min():.3f}, max={sp_dists.max():.3f}, mean={sp_dists.mean():.3f}")
        
        # SuperPoint matching
        row_indices, col_indices = linear_sum_assignment(sp_dists)
        sp_matches = [(i1, i2, sp_dists[i1, i2]) for i1, i2 in zip(row_indices, col_indices)]
        sp_matches = [m for m in sp_matches if m[2] < 1.0]  # Filter by threshold
        sp_matches.sort(key=lambda x: x[2])
        
        print(f"SuperPoint good matches: {len(sp_matches)}")
        if sp_matches:
            print(f"Best SuperPoint match distance: {sp_matches[0][2]:.3f}")
            print(f"Worst SuperPoint match distance: {sp_matches[-1][2]:.3f}")

    # --- SIFT matching for comparison ---
    sift_matches = []
    if sift_desc1 is not None and sift_desc2 is not None and len(sift_desc1) > 0 and len(sift_desc2) > 0:
        print(f"\n--- SIFT Comparison ---")
        
        # SIFT matching using Lowe's ratio test
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(sift_desc1, sift_desc2, k=2)
        
        good_sift_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                    good_sift_matches.append(m)
        
        print(f"SIFT good matches: {len(good_sift_matches)}")
        
        # Convert to our format
        for m in good_sift_matches[:len(sp_matches)]:  # Limit to same number as SuperPoint
            sift_matches.append((m.queryIdx, m.trainIdx, m.distance))

    # --- Visualization comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(18, 16))
    
    # SuperPoint visualization
    vis_sp = np.concatenate([crop1, crop2], axis=1)
    vis_sp = cv2.cvtColor(vis_sp, cv2.COLOR_GRAY2BGR)
    offset = np.array([CROP_SIZE, 0])

    # Draw SuperPoint keypoints and matches
    for idx, kpt in enumerate(kpts1):
        pt = tuple(np.round(kpt).astype(int))
        is_matched = any(i1 == idx for i1, _, _ in sp_matches)
        color = (255, 255, 255) if is_matched else (128, 128, 128)
        cv2.circle(vis_sp, pt, 3, color, 1)
        cv2.putText(vis_sp, str(idx), (pt[0]-5, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    for idx, kpt in enumerate(kpts2):
        pt = tuple(np.round(kpt).astype(int) + offset)
        is_matched = any(i2 == idx for _, i2, _ in sp_matches)
        color = (255, 255, 255) if is_matched else (128, 128, 128)
        cv2.circle(vis_sp, pt, 3, color, 1)
        cv2.putText(vis_sp, str(idx), (pt[0]-5, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Draw SuperPoint matches
    for i1, i2, distance in sp_matches:
        pt1 = tuple(np.round(kpts1[i1]).astype(int))
        pt2 = tuple(np.round(kpts2[i2]).astype(int) + offset)
        cv2.line(vis_sp, pt1, pt2, (0, 255, 0), 2)

    axes[0].imshow(vis_sp[..., ::-1])
    axes[0].set_title(f"SuperPoint Matches - {len(sp_matches)} matches")
    axes[0].axis('off')

    # SIFT visualization
    vis_sift = np.concatenate([crop1, crop2], axis=1)
    vis_sift = cv2.cvtColor(vis_sift, cv2.COLOR_GRAY2BGR)

    # Draw SIFT keypoints and matches
    if sift_kpts1 and sift_kpts2:
        for kpt in sift_kpts1:
            pt = tuple(np.round(kpt.pt).astype(int))
            cv2.circle(vis_sift, pt, 2, (255, 255, 255), 1)
        
        for kpt in sift_kpts2:
            pt = tuple(np.round(kpt.pt).astype(int) + offset)
            cv2.circle(vis_sift, pt, 2, (255, 255, 255), 1)

        # Draw SIFT matches
        for i1, i2, distance in sift_matches:
            pt1 = tuple(np.round(sift_kpts1[i1].pt).astype(int))
            pt2 = tuple(np.round(sift_kpts2[i2].pt).astype(int) + offset)
            cv2.line(vis_sift, pt1, pt2, (0, 255, 0), 2)

    axes[1].imshow(vis_sift[..., ::-1])
    axes[1].set_title(f"SIFT Matches - {len(sift_matches)} matches")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"superpoint_vs_sift_comparison_{i+1}.png", dpi=150, bbox_inches='tight')
    plt.show()

print("Analysis complete!")
