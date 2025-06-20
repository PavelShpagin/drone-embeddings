import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from third_party.AnyLoc.utilities import DinoV2ExtractFeatures, VLAD
import random

# Settings
IMG_PATH = "data/test/test.jpg"
CROP_SIZE = 200
N_CROPS = 1000
NC = 32  # VLAD clusters
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "dinov2_vits14"
LAYER = 11
FACET = "key"

def meters_to_pixels(meters, img_width, img_height, crop_size):
    # Approximate: assume 1 degree latitude ~ 111km, 1 degree longitude ~ 111km * cos(latitude)
    # For small crops, just use image size and crop size to estimate pixel/meter
    # Here, we assume the image covers a small area, so pixel/meter is roughly constant
    # User should adjust this for real geo-referenced images
    # For now, just set a fixed value (e.g., 2 meters per pixel)
    meters_per_pixel = 2.0  # Example: 1 pixel = 2 meters
    return int(meters / meters_per_pixel)

def generate_crops_with_coords(img_path, crop_size, n_crops):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    crops = []
    coords = []
    n_x = int(np.sqrt(n_crops))
    n_y = n_crops // n_x
    stride_x = (w - crop_size) // (n_x - 1) if n_x > 1 else 0
    stride_y = (h - crop_size) // (n_y - 1) if n_y > 1 else 0
    count = 0
    for iy in range(n_y):
        for ix in range(n_x):
            if count >= n_crops:
                break
            x = ix * stride_x
            y = iy * stride_y
            crop = img.crop((x, y, x + crop_size, y + crop_size))
            crops.append(crop)
            coords.append((x, y))
            count += 1
    return crops, coords

def generate_perturbed_crops(img_path, crop_size, coords, perturb_meters=20):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    perturbed_crops = []
    perturbed_coords = []
    pixel_perturb = meters_to_pixels(perturb_meters, w, h, crop_size)
    for (x, y) in coords:
        dx = random.randint(-pixel_perturb, pixel_perturb)
        dy = random.randint(-pixel_perturb, pixel_perturb)
        x_pert = min(max(x + dx, 0), w - crop_size)
        y_pert = min(max(y + dy, 0), h - crop_size)
        crop = img.crop((x_pert, y_pert, x_pert + crop_size, y_pert + crop_size))
        perturbed_crops.append(crop)
        perturbed_coords.append((x_pert, y_pert))
    return perturbed_crops, perturbed_coords

# 2. Transform for DINOv2
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Resize((224, 224)),
])

def extract_dense_features(imgs, model, device, batch_size=32):
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(imgs), batch_size), desc="Extracting dense features"):
            batch = torch.stack([transform(img) for img in imgs[i:i+batch_size]]).to(device)
            feats = model(batch)  # [B, N_patches, D]
            feats = torch.nn.functional.normalize(feats, dim=2)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)  # [N, N_patches, D]

def main():
    # Generate database crops (grid)
    db_crops, db_coords = generate_crops_with_coords(IMG_PATH, CROP_SIZE, N_CROPS)
    # Generate query crops (perturbed)
    query_crops, query_coords = generate_perturbed_crops(IMG_PATH, CROP_SIZE, db_coords, perturb_meters=20)

    print("Loading DINOv2 model...")
    model = DinoV2ExtractFeatures(MODEL_TYPE, LAYER, FACET, device=DEVICE)

    print("Extracting dense features for database crops...")
    db_feats = extract_dense_features(db_crops, model, DEVICE, BATCH_SIZE)  # [N, N_patches, D]
    print("Extracting dense features for query crops...")
    query_feats = extract_dense_features(query_crops, model, DEVICE, BATCH_SIZE)  # [N, N_patches, D]

    # 3. Build VLAD vocabulary on database crops only
    print(f"Running K-means for VLAD vocabulary (Nc={NC}) on database crops...")
    db_feats_flat = db_feats.reshape(-1, db_feats.shape[-1])
    vlad = VLAD(NC, desc_dim=db_feats_flat.shape[1], device=DEVICE)
    vlad.fit(db_feats_flat)
    print("VLAD vocabulary built.")

    # 4. Compute VLAD descriptors for database and query crops
    print("Aggregating VLAD descriptors for database crops...")
    db_vlad_descs = []
    for i in tqdm(range(0, len(db_feats), BATCH_SIZE), desc="VLAD aggregation (db)"):
        feats_batch = db_feats[i:i+BATCH_SIZE]
        for j in range(feats_batch.shape[0]):
            vlad_desc = vlad.generate(feats_batch[j].to(DEVICE))
            db_vlad_descs.append(vlad_desc.cpu())
    db_vlad_descs = torch.stack(db_vlad_descs)  # [N, Nc*D]

    print("Aggregating VLAD descriptors for query crops...")
    query_vlad_descs = []
    for i in tqdm(range(0, len(query_feats), BATCH_SIZE), desc="VLAD aggregation (query)"):
        feats_batch = query_feats[i:i+BATCH_SIZE]
        for j in range(feats_batch.shape[0]):
            vlad_desc = vlad.generate(feats_batch[j].to(DEVICE))
            query_vlad_descs.append(vlad_desc.cpu())
    query_vlad_descs = torch.stack(query_vlad_descs)  # [N, Nc*D]

    print(f"VLAD descriptors shape: db {db_vlad_descs.shape}, query {query_vlad_descs.shape}")

    # 5. Recall evaluation (query vs. database)
    print("Computing similarity and recall (query vs. database)...")
    sim = query_vlad_descs @ db_vlad_descs.T  # [N_query, N_db]
    recall1 = 0
    recall5 = 0
    n_eval = len(query_crops)
    for i in range(n_eval):
        top5 = torch.topk(sim[i], k=5).indices.cpu().numpy()
        # The correct match is the i-th db crop (since queries are perturbed from db crops)
        if i == top5[0]:
            recall1 += 1
        if i in top5:
            recall5 += 1
    print("\n================ Perturbed Query Recall ================")
    print(f"Recall@1: {recall1/n_eval:.4f} ({recall1}/{n_eval})")
    print(f"Recall@5: {recall5/n_eval:.4f} ({recall5}/{n_eval})")
    print("======================================================")

if __name__ == "__main__":
    main() 