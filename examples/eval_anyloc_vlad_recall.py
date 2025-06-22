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

# --- Sliding window with overlap for DB and Query crops ---
def sliding_window_crops(img_path, crop_size, stride):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    crops = []
    coords = []
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            crop = img.crop((x, y, x + crop_size, y + crop_size))
            crops.append(crop)
            coords.append((x + crop_size // 2, y + crop_size // 2))  # center
    return crops, coords

# --- Aerial Vocabulary: Use random crops from the full image ---
def get_aerial_vocab_crops(img_path, crop_size, n_vocab_crops):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    crops = []
    for _ in range(n_vocab_crops):
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        crop = img.crop((x, y, x + crop_size, y + crop_size))
        crops.append(crop)
    return crops

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

def compute_spatial_distance(c1, c2):
    # c1, c2: (x, y)
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def main():
    # --- Step 1: Build Aerial Vocabulary from random crops ---
    N_VOCAB_CROPS = 500  # Number of crops for aerial vocab
    vocab_crops = get_aerial_vocab_crops(IMG_PATH, CROP_SIZE, N_VOCAB_CROPS)

    # --- Step 2: Build DB and Query crops with overlap ---
    STRIDE = CROP_SIZE // 2  # 50% overlap
    db_crops, db_coords = sliding_window_crops(IMG_PATH, CROP_SIZE, STRIDE)
    query_crops, query_coords = sliding_window_crops(IMG_PATH, CROP_SIZE, STRIDE)
    # Optionally, you could offset the query window for more realism

    print(f"DB/query crops: {len(db_crops)}")

    print("Loading DINOv2 model...")
    model = DinoV2ExtractFeatures(MODEL_TYPE, LAYER, FACET, device=DEVICE)

    print("Extracting dense features for aerial vocabulary...")
    vocab_feats = extract_dense_features(vocab_crops, model, DEVICE, BATCH_SIZE)  # [N, N_patches, D]
    print("Extracting dense features for database crops...")
    db_feats = extract_dense_features(db_crops, model, DEVICE, BATCH_SIZE)  # [N, N_patches, D]
    print("Extracting dense features for query crops...")
    query_feats = extract_dense_features(query_crops, model, DEVICE, BATCH_SIZE)  # [N, N_patches, D]

    # --- Step 3: Build VLAD vocabulary on aerial vocab crops only ---
    print(f"Running K-means for VLAD vocabulary (Nc={NC}) on aerial vocab crops...")
    vocab_feats_flat = vocab_feats.reshape(-1, vocab_feats.shape[-1])
    vlad = VLAD(NC, desc_dim=vocab_feats_flat.shape[1], device=DEVICE)
    vlad.fit(vocab_feats_flat)
    print("VLAD vocabulary built.")

    # --- Step 4: Compute VLAD descriptors for database and query crops ---
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

    # --- Step 5: Recall evaluation with spatial tolerance ---
    print("Computing similarity and recall (query vs. database) with spatial tolerance...")
    sim = query_vlad_descs @ db_vlad_descs.T  # [N_query, N_db]
    recall1 = 0
    recall5 = 0
    n_eval = len(query_crops)
    spatial_tol = 1.5 * CROP_SIZE  # spatial tolerance in pixels
    for i in range(n_eval):
        top5 = torch.topk(sim[i], k=5).indices.cpu().numpy()
        # For each retrieved db index, check if within spatial tolerance
        correct1 = False
        correct5 = False
        for k, db_idx in enumerate(top5):
            dist = compute_spatial_distance(query_coords[i], db_coords[db_idx])
            if dist <= spatial_tol:
                if k == 0:
                    correct1 = True
                correct5 = True
        if correct1:
            recall1 += 1
        if correct5:
            recall5 += 1
    print("\n================ Realistic Overlap+Spatial Tolerance Recall ===============")
    print(f"Recall@1: {recall1/n_eval:.4f} ({recall1}/{n_eval})")
    print(f"Recall@5: {recall5/n_eval:.4f} ({recall5}/{n_eval})")
    print("==========================================================================")

if __name__ == "__main__":
    main() 