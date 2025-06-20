import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from third_party.AnyLoc.custom_datasets.cropped_image_dataset import CroppedImageDataset
from third_party.AnyLoc.utilities import DinoV2ExtractFeatures

# Settings
TEST_IMG_PATH = "data/test/test.jpg"
TEST_CROP_SIZE = 200  # px
TEST_CROP_STRIDE = 200  # px (non-overlapping)
DB_SIZES = [1000, 5000, 20000, "all"]
DB_ROOT = "data/cropped_loc_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MODEL_TYPE = "dinov2_vits14"
LAYER = 11
FACET = "key"

# Helper: get database dir for size
def get_db_dir(size):
    if size == "all":
        # Find the largest N_crops dir
        all_dirs = [d for d in os.listdir(DB_ROOT) if d.endswith("_crops") and os.path.isdir(os.path.join(DB_ROOT, d))]
        max_dir = max(all_dirs, key=lambda d: int(d.split("_crops")[0]))
        return os.path.join(DB_ROOT, max_dir)
    else:
        return os.path.join(DB_ROOT, f"{size}_crops")

def extract_embeddings(dataset, model, device, batch_size=32):
    all_embeds = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting embeddings", leave=False):
            imgs = imgs.to(device)
            # DINOv2 returns [B, N_patches, D], take mean over patches
            feats = model(imgs)  # [B, N_patches, D]
            feats = feats.mean(dim=1)  # [B, D]
            feats = torch.nn.functional.normalize(feats, dim=1)
            all_embeds.append(feats.cpu())
    return torch.cat(all_embeds, dim=0)

def generate_test_crops(img_path, crop_size=200, stride=200):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    crops = []
    crop_coords = []
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            crop = img.crop((x, y, x + crop_size, y + crop_size))
            crops.append(crop)
            crop_coords.append((x, y))
    print(f"Generated {len(crops)} test crops from {img_path}")
    return crops, crop_coords

def main():
    # Generate test crops on the fly
    test_crops, test_coords = generate_test_crops(TEST_IMG_PATH, TEST_CROP_SIZE, TEST_CROP_STRIDE)
    # Assign synthetic GPS tags (use pixel coords as unique IDs)
    test_gps = test_coords
    # Transform for DINOv2
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((224, 224)),
    ])
    test_imgs = [transform(crop) for crop in test_crops]
    test_imgs = torch.stack(test_imgs)
    # Load DINOv2 model
    print("Loading DINOv2 model...")
    model = DinoV2ExtractFeatures(MODEL_TYPE, LAYER, FACET, device=DEVICE)
    # Extract test embeddings
    print("Extracting test crop embeddings...")
    with torch.no_grad():
        test_embeds = []
        for i in tqdm(range(0, len(test_imgs), BATCH_SIZE), desc="Test batches"):
            batch = test_imgs[i:i+BATCH_SIZE].to(DEVICE)
            feats = model(batch)
            feats = feats.mean(dim=1)
            feats = torch.nn.functional.normalize(feats, dim=1)
            test_embeds.append(feats.cpu())
        test_embeds = torch.cat(test_embeds, dim=0)
    for db_size in DB_SIZES:
        db_dir = get_db_dir(db_size)
        db_metadata = os.path.join(db_dir, "cropped_images_metadata.json")
        print(f"\nEvaluating with database size: {db_size} ({db_dir})")
        db_ds = CroppedImageDataset(db_metadata)
        print(f"Loaded {len(db_ds)} database crops.")
        db_embeds = extract_embeddings(db_ds, model, DEVICE, BATCH_SIZE)
        # For synthetic matching: assign each test crop a unique GPS (its pixel coord), and only count as correct if a db crop has the same GPS (pixel coord)
        db_gps = [(entry["crop_center_lat"], entry["crop_center_lon"]) for entry in db_ds.metadata]
        # Build GPS to index mapping for database
        gps_to_db_idx = {gps: i for i, gps in enumerate(db_gps)}
        # Compute cosine similarity
        print("Computing similarity and recall...")
        sim = test_embeds @ db_embeds.T  # [N_test, N_db]
        recall1 = 0
        recall5 = 0
        n_eval = 0
        for i, gps in enumerate(test_gps):
            if gps not in gps_to_db_idx:
                continue  # No true match in db
            true_idx = gps_to_db_idx[gps]
            top5 = torch.topk(sim[i], k=5).indices.cpu().numpy()
            if true_idx == top5[0]:
                recall1 += 1
            if true_idx in top5:
                recall5 += 1
            n_eval += 1
        print(f"Recall@1: {recall1/n_eval:.4f} ({recall1}/{n_eval})")
        print(f"Recall@5: {recall5/n_eval:.4f} ({recall5}/{n_eval})")

if __name__ == "__main__":
    main() 