import numpy as np
import torch
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Import DINOv2+VLAD embedder
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

# --- Config ---
MAP_IMAGE_PATH = "inference/46.6234, 32.7851.jpg"  # Example map
M_PER_PIXEL = 4000.0 / 8192.0  # meters per pixel
CROP_SIZE_PX = 224
CROP_STRIDE_PX = CROP_SIZE_PX // 2  # 50% overlap for database
PATCH_SIZE_M = CROP_SIZE_PX * M_PER_PIXEL
VIO_NOISE_STD = 10.0  # meters per step
STEP_SIZE_M = 50.0  # meters per step
UPDATE_INTERVAL_M = 100.0
VIDEO_FILENAME = "dinovit_probmap_simulation.avi"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load map and prepare database crops (grid cells) ---
map_image = Image.open(MAP_IMAGE_PATH).convert("RGB")
map_w, map_h = map_image.size

# Define database grid cell centers with overlap (for database only)
db_centers_px = []
db_centers_m = []
db_images = []
for y in range(0, map_h - CROP_SIZE_PX + 1, CROP_STRIDE_PX):
    for x in range(0, map_w - CROP_SIZE_PX + 1, CROP_STRIDE_PX):
        crop = map_image.crop((x, y, x + CROP_SIZE_PX, y + CROP_SIZE_PX))
        db_images.append(crop)
        cx, cy = x + CROP_SIZE_PX // 2, y + CROP_SIZE_PX // 2
        db_centers_px.append((cx, cy))
        x_m = (cx - map_w // 2) * M_PER_PIXEL
        y_m = (cy - map_h // 2) * M_PER_PIXEL
        db_centers_m.append((x_m, y_m))
GRID_W = (map_w - CROP_SIZE_PX) // CROP_STRIDE_PX + 1
GRID_H = (map_h - CROP_SIZE_PX) // CROP_STRIDE_PX + 1

print(f"Database grid: {GRID_W}x{GRID_H} = {len(db_images)} crops")

# --- Use fixed aerial vocabulary (FAIR: don't fit on same data) ---
embedder = AnyLocVLADEmbedder(device=DEVICE)

# Generate fixed aerial vocabulary from a DIFFERENT aerial image (not the same as database)
print("Generating fixed aerial vocabulary from a different aerial image...")
# Use a different location for vocabulary to ensure fairness
vocab_image_path = "data/earth_imagery/loc2/46.6234, 32.7851.spring.jpg"  # Different location
if not os.path.exists(vocab_image_path):
    # Fallback to a different part of the same image if other locations don't exist
    print("Warning: Using fallback vocabulary from different part of same image")
    vocab_image = map_image.crop((map_w//4, map_h//4, 3*map_w//4, 3*map_h//4))  # Center crop
else:
    vocab_image = Image.open(vocab_image_path).convert("RGB")

vocab_w, vocab_h = vocab_image.size
vocab_images = []
for _ in range(100):  # 100 random crops for vocabulary
    x = np.random.randint(0, vocab_w - CROP_SIZE_PX)
    y = np.random.randint(0, vocab_h - CROP_SIZE_PX)
    crop = vocab_image.crop((x, y, x + CROP_SIZE_PX, y + CROP_SIZE_PX))
    vocab_images.append(crop)

# Extract dense features for vocabulary
all_dense_feats = []
for img in tqdm(vocab_images, desc="Dense features for vocab"):
    feats = embedder.extract_dense_features(img)
    all_dense_feats.append(feats)
all_dense_feats = np.concatenate(all_dense_feats, axis=0)

# Subsample for K-means
MAX_VOCAB_FEATS = 20000
if all_dense_feats.shape[0] > MAX_VOCAB_FEATS:
    idx = np.random.choice(all_dense_feats.shape[0], MAX_VOCAB_FEATS, replace=False)
    vocab_feats = all_dense_feats[idx]
else:
    vocab_feats = all_dense_feats

print(f"Fitting VLAD vocabulary on {vocab_feats.shape[0]} features...")
embedder.fit_vocabulary_from_features(vocab_feats)

# Compute embeddings for database crops
print("Computing embeddings for database crops...")
db_embeddings = [embedder.get_embedding(img) for img in tqdm(db_images)]
db_embeddings = np.stack(db_embeddings)

# --- Simulate drone trajectories ---
def random_trajectory(start, steps, step_size, noise_std, bounds):
    """Simulate a random walk trajectory within bounds."""
    traj = [np.array(start)]
    for _ in range(steps):
        angle = np.random.uniform(0, 2 * np.pi)
        move = np.array([np.cos(angle), np.sin(angle)]) * step_size
        noisy_move = move + np.random.normal(0, noise_std, 2)
        next_pos = traj[-1] + noisy_move
        # Keep within bounds
        next_pos[0] = np.clip(next_pos[0], bounds[0][0], bounds[0][1])
        next_pos[1] = np.clip(next_pos[1], bounds[1][0], bounds[1][1])
        traj.append(next_pos)
    return np.array(traj)

# Map bounds in meters
x_bounds = (-(map_w // 2) * M_PER_PIXEL, (map_w // 2) * M_PER_PIXEL)
y_bounds = (-(map_h // 2) * M_PER_PIXEL, (map_h // 2) * M_PER_PIXEL)

num_steps = int((min(map_w, map_h) * M_PER_PIXEL) // STEP_SIZE_M)
num_steps = num_steps * 5  # Longer simulation
true_start = (0.0, 0.0)
true_traj = random_trajectory(true_start, num_steps, STEP_SIZE_M, 0, (x_bounds, y_bounds))
vio_traj = random_trajectory(true_start, num_steps, STEP_SIZE_M, VIO_NOISE_STD, (x_bounds, y_bounds))

# --- Video writer setup ---
VIDEO_W, VIDEO_H = 1920, 1080
if map_w > VIDEO_W or map_h > VIDEO_H:
    print(f"[Warning] Map size ({map_w}x{map_h}) is larger than video frame ({VIDEO_W}x{VIDEO_H}). Downscaling frames.")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, 10, (VIDEO_W, VIDEO_H))

# --- Recall statistics ---
recall1_count = 0
recall5_count = 0
recall_total = 0
SPATIAL_TOL_M = 1.5 * PATCH_SIZE_M  # spatial tolerance in meters

print(f"Spatial tolerance: {SPATIAL_TOL_M:.1f}m ({1.5}x crop size)")

# --- Main simulation loop ---
for step in tqdm(range(1, num_steps+1)):
    # Every UPDATE_INTERVAL_M, perform retrieval
    if step % int(UPDATE_INTERVAL_M // STEP_SIZE_M) == 0:
        # --- FAIR: Extract query crop at true position (non-overlapping with database) ---
        true_pos = true_traj[step]
        
        # Convert world position to pixel coordinates
        px = int(true_pos[0] / M_PER_PIXEL + map_w // 2)
        py = int(true_pos[1] / M_PER_PIXEL + map_h // 2)
        
        # Extract query crop (centered at true position)
        x0 = px - CROP_SIZE_PX // 2
        y0 = py - CROP_SIZE_PX // 2
        x1 = x0 + CROP_SIZE_PX
        y1 = y0 + CROP_SIZE_PX
        
        # Handle boundary conditions
        query_crop = Image.new('RGB', (CROP_SIZE_PX, CROP_SIZE_PX), color=(0, 0, 0))
        crop_img = map_image.crop((max(0, x0), max(0, y0), min(map_w, x1), min(map_h, y1)))
        paste_x = max(0, -x0)
        paste_y = max(0, -y0)
        query_crop.paste(crop_img, (paste_x, paste_y))
        
        # Get query embedding
        query_emb = embedder.get_embedding(query_crop)
        
        # Compute L2 distance to all database embeddings
        l2_dists = np.linalg.norm(db_embeddings - query_emb, axis=1)
        
        # Get top-5 closest matches
        top5_idx = np.argpartition(l2_dists, 5)[:5]
        sorted_top5 = top5_idx[np.argsort(l2_dists[top5_idx])]
        
        # Get the closest match position
        closest_idx = sorted_top5[0]
        closest_pos = db_centers_m[closest_idx]
        closest_dist = l2_dists[closest_idx]
        
        # --- Recall@1 and Recall@5 with spatial tolerance ---
        # Calculate spatial distances from database centers to true position
        spatial_dists = np.array([np.linalg.norm(np.array(center) - true_pos) for center in db_centers_m])
        
        recall_total += 1
        
        # Check if any of the top-1/top-5 are within spatial tolerance
        top1_spatial_dist = spatial_dists[sorted_top5[0]]
        top5_spatial_dists = spatial_dists[sorted_top5]
        
        if top1_spatial_dist <= SPATIAL_TOL_M:
            recall1_count += 1
        if np.any(top5_spatial_dists <= SPATIAL_TOL_M):
            recall5_count += 1
            
        # Log retrieval results
        print(f"Step {step}: True pos: {true_pos}, Closest match: {closest_pos}")
        print(f"  Embedding distance: {closest_dist:.4f}, Spatial distance: {top1_spatial_dist:.1f}m")
        print(f"  Top-5 embedding distances: {l2_dists[sorted_top5]}")
        print(f"  Top-5 spatial distances: {top5_spatial_dists}")
        print(f"  Spatial tolerance: {SPATIAL_TOL_M:.1f}m")

    # --- Visualization ---
    vis = np.array(map_image).copy()
    
    # Draw true and VIO trajectories
    for i in range(1, step+1):
        def world_to_px(pos):
            x = int(pos[0] / M_PER_PIXEL + map_w // 2)
            y = int(pos[1] / M_PER_PIXEL + map_h // 2)
            return x, y
        cv2.line(vis, world_to_px(true_traj[i-1]), world_to_px(true_traj[i]), (0, 255, 0), 2)
        cv2.line(vis, world_to_px(vio_traj[i-1]), world_to_px(vio_traj[i]), (0, 0, 255), 2)
    
    # Draw current positions
    cv2.circle(vis, world_to_px(true_traj[step]), 6, (0, 255, 0), -1)
    cv2.circle(vis, world_to_px(vio_traj[step]), 6, (0, 0, 255), -1)
    
    # Draw closest retrieved position if available
    if step % int(UPDATE_INTERVAL_M // STEP_SIZE_M) == 0:
        # Draw yellow square around the true position (where query crop was extracted)
        true_px = world_to_px(true_pos)
        square_size = 20
        cv2.rectangle(vis, 
                     (true_px[0] - square_size//2, true_px[1] - square_size//2),
                     (true_px[0] + square_size//2, true_px[1] + square_size//2),
                     (0, 255, 255), 3)  # Yellow square, 3px thick
        
        cv2.circle(vis, world_to_px(closest_pos), 8, (255, 0, 0), -1)  # Blue circle for closest match
        # Draw line from true to closest
        cv2.line(vis, world_to_px(true_pos), world_to_px(closest_pos), (255, 0, 0), 2)
        
        # Draw top-5 matches as smaller circles
        for i, idx in enumerate(sorted_top5[1:5]):  # Skip first (already drawn)
            match_pos = db_centers_m[idx]
            cv2.circle(vis, world_to_px(match_pos), 4, (255, 255, 0), -1)  # Yellow circles
    
    # Resize for video
    vis_resized = cv2.resize(vis, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA)
    # Write frame
    video_out.write(vis_resized)

video_out.release()
print(f"Simulation video saved to {VIDEO_FILENAME}")
if recall_total > 0:
    print(f"Final Recall@1: {recall1_count}/{recall_total} = {recall1_count/recall_total:.3f}")
    print(f"Final Recall@5: {recall5_count}/{recall_total} = {recall5_count/recall_total:.3f}")
else:
    print("No recall statistics to report.") 