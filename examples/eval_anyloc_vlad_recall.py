import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from third_party.AnyLoc.utilities import DinoV2ExtractFeatures, VLAD

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

# 1. Load and crop the image
def generate_crops(img_path, crop_size, n_crops):
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
    print(f"Generated {len(crops)} crops from {img_path}")
    return crops, coords

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
    crops, coords = generate_crops(IMG_PATH, CROP_SIZE, N_CROPS)
    print("Loading DINOv2 model...")
    model = DinoV2ExtractFeatures(MODEL_TYPE, LAYER, FACET, device=DEVICE)
    print("Extracting dense features for all crops...")
    feats = extract_dense_features(crops, model, DEVICE, BATCH_SIZE)  # [N, N_patches, D]
    # 3. Build VLAD vocabulary
    print(f"Running K-means for VLAD vocabulary (Nc={NC}) on all crops...")
    feats_flat = feats.reshape(-1, feats.shape[-1])
    vlad = VLAD(NC, desc_dim=feats_flat.shape[1], device=DEVICE)
    vlad.fit(feats_flat)
    print("VLAD vocabulary built.")
    # 4. Compute VLAD descriptors for all crops
    print("Aggregating VLAD descriptors for all crops...")
    vlad_descs = []
    for i in tqdm(range(0, len(feats), BATCH_SIZE), desc="VLAD aggregation"):
        feats_batch = feats[i:i+BATCH_SIZE]
        for j in range(feats_batch.shape[0]):
            vlad_desc = vlad.generate(feats_batch[j].to(DEVICE))
            vlad_descs.append(vlad_desc.cpu())
    vlad_descs = torch.stack(vlad_descs)  # [N, Nc*D]
    print(f"VLAD descriptors shape: {vlad_descs.shape}")
    # 5. Recall evaluation (self-retrieval)
    print("Computing similarity and recall...")
    sim = vlad_descs @ vlad_descs.T  # [N, N]
    recall1 = 0
    recall5 = 0
    n_eval = len(crops)
    for i in range(n_eval):
        top5 = torch.topk(sim[i], k=5).indices.cpu().numpy()
        if i == top5[0]:
            recall1 += 1
        if i in top5:
            recall5 += 1
    print("\n================ Clean Self-Retrieval Recall ================" )
    print(f"Recall@1: {recall1/n_eval:.4f} ({recall1}/{n_eval})")
    print(f"Recall@5: {recall5/n_eval:.4f} ({recall5}/{n_eval})")
    print("===========================================================")

if __name__ == "__main__":
    main() 