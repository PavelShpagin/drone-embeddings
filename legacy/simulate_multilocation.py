import numpy as np
import torch
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import EfficientNetB0+NetVLAD embedder
from train_encoder import SiameseNet, get_eval_transforms

# --- Config ---
M_PER_PIXEL = 4000.0 / 8192.0  # meters per pixel
CROP_SIZE_PX = 224
CROP_STRIDE_PX = CROP_SIZE_PX // 2  # 50% overlap for database
PATCH_SIZE_M = CROP_SIZE_PX * M_PER_PIXEL
VIO_NOISE_STD = 10.0  # meters per step
STEP_SIZE_M = 50.0  # meters per step
UPDATE_INTERVAL_M = 100.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- EfficientNetB0+NetVLAD configuration ---
EFFICIENTNET_WEIGHTS = "training_results/efficientnet_b0/checkpoints/checkpoint_epoch_85.pth"
EFFICIENTNET_VLAD_CENTERS = "efficientnet_vlad_centers.pt"

# --- SuperPoint configuration ---
SUPERPOINT_WEIGHTS = "superpoint_uav_trained/superpoint_uav_epoch_20.pth"
USE_SUPERPOINT = False  # Will be set to True for the second run

# --- Location configuration ---
LOCATIONS = [
    "data/earth_imagery/loc1/45.3395, 29.6287.spring.jpg",
    "data/earth_imagery/loc2/46.6234, 32.7851.spring.jpg", 
    "data/earth_imagery/loc3/48.2650, 24.3913.spring.jpg",
    "data/earth_imagery/loc4/48.5673, 33.4218.spring.jpg",
    "data/earth_imagery/loc5/48.9483, 29.7241.spring.jpg",
    "data/earth_imagery/loc6/49.3721, 31.0945.spring.jpg",
    "data/earth_imagery/loc7/49.8234, 25.3612.spring.jpg",
    "data/earth_imagery/loc8/50.1964, 36.3753.spring.jpg",
    "data/earth_imagery/loc9/50.4162, 30.8906.spring.jpg",
    "data/earth_imagery/loc10/51.0820, 30.6485.spring.jpg"
]

# --- Results storage ---
results = {
    "timestamp": datetime.now().isoformat(),
    "config": {
        "crop_size_px": CROP_SIZE_PX,
        "crop_stride_px": CROP_STRIDE_PX,
        "patch_size_m": PATCH_SIZE_M,
        "spatial_tolerance_m": 1.5 * PATCH_SIZE_M,
        "device": DEVICE,
        "use_superpoint": USE_SUPERPOINT,
        "efficientnet_weights": EFFICIENTNET_WEIGHTS
    },
    "locations": {}
}

# --- EfficientNetB0+NetVLAD Embedder ---
class EfficientNetVLADEmbedder:
    def __init__(self, weights_path, device="cuda", cluster_centers_path=None):
        self.device = device
        self.model = SiameseNet("efficientnet_b0", cluster_centers_path=cluster_centers_path, device=device).to(device)
        self.model.eval()
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.transform = get_eval_transforms()
    def get_embedding(self, pil_img):
        img = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.get_embedding(img)
        return emb.cpu().numpy().flatten()

def match_descriptors(desc1, desc2, threshold=0.8):
    if len(desc1) == 0 or len(desc2) == 0:
        return float('inf')
    dists = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)
    matches = []
    for i in range(len(desc1)):
        sorted_indices = np.argsort(dists[i])
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1] if len(sorted_indices) > 1 else best_idx
        best_dist = dists[i, best_idx]
        second_best_dist = dists[i, second_best_idx]
        if best_dist < threshold * second_best_dist:
            matches.append(best_dist)
    return np.mean(matches) if matches else float('inf')

def run_simulation_for_location(location_path, location_name):
    global USE_SUPERPOINT
    print(f"\n{'='*60}")
    print(f"Testing location: {location_name}")
    print(f"Image: {location_path}")
    print(f"{'='*60}")
    if not os.path.exists(location_path):
        print(f"Warning: {location_path} not found, skipping...")
        return None
    superpoint = None
    if USE_SUPERPOINT:
        from simple_superpoint import SuperPoint
        if os.path.exists(SUPERPOINT_WEIGHTS):
            print(f"Loading SuperPoint from: {SUPERPOINT_WEIGHTS}")
            superpoint = SuperPoint(SUPERPOINT_WEIGHTS, device=DEVICE)
            print("✓ SuperPoint loaded successfully")
        else:
            print(f"Warning: SuperPoint weights not found at {SUPERPOINT_WEIGHTS}")
            print("Continuing without SuperPoint integration...")
            USE_SUPERPOINT = False
    map_image = Image.open(location_path).convert("RGB")
    map_w, map_h = map_image.size
    print(f"Map size: {map_w}x{map_h} pixels")
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
    embedder = EfficientNetVLADEmbedder(EFFICIENTNET_WEIGHTS, device=DEVICE, cluster_centers_path=EFFICIENTNET_VLAD_CENTERS)
    print("Computing embeddings for database crops...")
    db_embeddings = [embedder.get_embedding(img) for img in tqdm(db_images)]
    db_embeddings = np.stack(db_embeddings)
    db_superpoint_descs = []
    if USE_SUPERPOINT and superpoint:
        print("Pre-computing SuperPoint descriptors for database crops...")
        for img in tqdm(db_images, desc="SuperPoint descriptors"):
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            _, _, desc = superpoint.detect(img_cv)
            db_superpoint_descs.append(desc)
    def random_trajectory(start, steps, step_size, noise_std, bounds):
        traj = [np.array(start)]
        for _ in range(steps):
            angle = np.random.uniform(0, 2 * np.pi)
            move = np.array([np.cos(angle), np.sin(angle)]) * step_size
            noisy_move = move + np.random.normal(0, noise_std, 2)
            next_pos = traj[-1] + noisy_move
            next_pos[0] = np.clip(next_pos[0], bounds[0][0], bounds[0][1])
            next_pos[1] = np.clip(next_pos[1], bounds[1][0], bounds[1][1])
            traj.append(next_pos)
        return np.array(traj)
    x_bounds = (-(map_w // 2) * M_PER_PIXEL, (map_w // 2) * M_PER_PIXEL)
    y_bounds = (-(map_h // 2) * M_PER_PIXEL, (map_h // 2) * M_PER_PIXEL)
    num_steps = int((min(map_w, map_h) * M_PER_PIXEL) // STEP_SIZE_M)
    num_steps = min(num_steps * 3, 100)
    true_start = (0.0, 0.0)
    true_traj = random_trajectory(true_start, num_steps, STEP_SIZE_M, 0, (x_bounds, y_bounds))
    vio_traj = random_trajectory(true_start, num_steps, STEP_SIZE_M, VIO_NOISE_STD, (x_bounds, y_bounds))
    recall1_count = 0
    recall5_count = 0
    recall_total = 0
    SPATIAL_TOL_M = 1.5 * PATCH_SIZE_M
    print(f"Spatial tolerance: {SPATIAL_TOL_M:.1f}m ({1.5}x crop size)")
    print(f"Running {num_steps} steps...")
    for step in tqdm(range(1, num_steps+1), desc=f"Simulating {location_name}"):
        if step % int(UPDATE_INTERVAL_M // STEP_SIZE_M) == 0:
            true_pos = true_traj[step]
            px = int(true_pos[0] / M_PER_PIXEL + map_w // 2)
            py = int(true_pos[1] / M_PER_PIXEL + map_h // 2)
            x0 = px - CROP_SIZE_PX // 2
            y0 = py - CROP_SIZE_PX // 2
            x1 = x0 + CROP_SIZE_PX
            y1 = y0 + CROP_SIZE_PX
            query_crop = Image.new('RGB', (CROP_SIZE_PX, CROP_SIZE_PX), color=(0, 0, 0))
            crop_img = map_image.crop((max(0, x0), max(0, y0), min(map_w, x1), min(map_h, y1)))
            paste_x = max(0, -x0)
            paste_y = max(0, -y0)
            query_crop.paste(crop_img, (paste_x, paste_y))
            query_emb = embedder.get_embedding(query_crop)
            l2_dists = np.linalg.norm(db_embeddings - query_emb, axis=1)
            top5_idx = np.argpartition(l2_dists, 5)[:5]
            sorted_top5 = top5_idx[np.argsort(l2_dists[top5_idx])]
            if USE_SUPERPOINT and superpoint:
                query_cv = cv2.cvtColor(np.array(query_crop), cv2.COLOR_RGB2GRAY)
                _, _, query_desc = superpoint.detect(query_cv)
                sp_distances = []
                for idx in sorted_top5:
                    db_desc = db_superpoint_descs[idx]
                    sp_dist = match_descriptors(query_desc, db_desc)
                    sp_distances.append(sp_dist)
                sp_ranking = np.argsort(sp_distances)
                final_top5 = sorted_top5[sp_ranking]
            else:
                final_top5 = sorted_top5
            closest_idx = final_top5[0]
            closest_pos = db_centers_m[closest_idx]
            closest_dist = l2_dists[closest_idx]
            spatial_dists = np.array([np.linalg.norm(np.array(center) - true_pos) for center in db_centers_m])
            recall_total += 1
            top1_spatial_dist = spatial_dists[final_top5[0]]
            top5_spatial_dists = spatial_dists[final_top5]
            if top1_spatial_dist <= SPATIAL_TOL_M:
                recall1_count += 1
            if np.any(top5_spatial_dists <= SPATIAL_TOL_M):
                recall5_count += 1
    recall1_rate = recall1_count / recall_total if recall_total > 0 else 0
    recall5_rate = recall5_count / recall_total if recall_total > 0 else 0
    print(f"\nResults for {location_name}:")
    print(f"  Recall@1: {recall1_count}/{recall_total} = {recall1_rate:.3f}")
    print(f"  Recall@5: {recall5_count}/{recall_total} = {recall5_rate:.3f}")
    if USE_SUPERPOINT:
        print(f"  SuperPoint Integration: ENABLED")
    else:
        print(f"  SuperPoint Integration: DISABLED")
    return {
        "location_name": location_name,
        "image_path": location_path,
        "map_size": f"{map_w}x{map_h}",
        "grid_size": f"{GRID_W}x{GRID_H}",
        "num_crops": len(db_images),
        "num_steps": num_steps,
        "recall_total": recall_total,
        "recall1_count": recall1_count,
        "recall1_rate": recall1_rate,
        "recall5_count": recall5_count,
        "recall5_rate": recall5_rate,
        "spatial_tolerance_m": SPATIAL_TOL_M,
        "superpoint_enabled": USE_SUPERPOINT
    }

print("Multi-Location EfficientNetB0+NetVLAD Simulation")
print("Testing algorithm performance across different geographic locations")
print(f"Device: {DEVICE}")
if USE_SUPERPOINT:
    print(f"SuperPoint Integration: ENABLED (weights: {SUPERPOINT_WEIGHTS})")
else:
    print("SuperPoint Integration: DISABLED")
for i, location_path in enumerate(LOCATIONS):
    location_name = f"loc{i+1}"
    result = run_simulation_for_location(location_path, location_name)
    if result:
        results["locations"][location_name] = result
if results["locations"]:
    recall1_rates = [loc["recall1_rate"] for loc in results["locations"].values()]
    recall5_rates = [loc["recall5_rate"] for loc in results["locations"].values()]
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS ACROSS ALL LOCATIONS")
    print(f"{'='*60}")
    print(f"Average Recall@1: {np.mean(recall1_rates):.3f} ± {np.std(recall1_rates):.3f}")
    print(f"Average Recall@5: {np.mean(recall5_rates):.3f} ± {np.std(recall5_rates):.3f}")
    print(f"Min Recall@1: {np.min(recall1_rates):.3f} (location: {list(results['locations'].keys())[np.argmin(recall1_rates)]})")
    print(f"Max Recall@1: {np.max(recall1_rates):.3f} (location: {list(results['locations'].keys())[np.argmax(recall1_rates)]})")
    results["aggregate"] = {
        "mean_recall1": float(np.mean(recall1_rates)),
        "std_recall1": float(np.std(recall1_rates)),
        "mean_recall5": float(np.mean(recall5_rates)),
        "std_recall5": float(np.std(recall5_rates)),
        "min_recall1": float(np.min(recall1_rates)),
        "max_recall1": float(np.max(recall1_rates)),
        "num_locations_tested": len(results["locations"])
    }
results_file = f"multilocation_efficientnet_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {results_file}")
print("Simulation complete!") 