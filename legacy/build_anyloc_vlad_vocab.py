import os
import random
from glob import glob
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

# Settings
VPAIR_QUERY_DIR = 'third_party/vpair_sample/queries'
EARTH_IMAGERY_DIR = 'data/earth_imagery'
CROP_SIZE = 224
N_CROPS_PER_IMAGE = 5
N_VOCAB_CROPS = 20000  # Total crops for K-means
N_CLUSTERS = 32
VOCAB_SAVE_PATH = 'c_centers.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = 'dinov2_vits14'
LAYER = 11
FACET = 'key'

# Transform for DINOv2
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Resize((224, 224)),
])

def sample_crops_from_image(img_path, crop_size, n_crops):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    crops = []
    for _ in range(n_crops):
        if w < crop_size or h < crop_size:
            continue
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        crop = img.crop((x, y, x + crop_size, y + crop_size))
        crops.append(crop)
    return crops

def gather_all_crops():
    crops = []
    # VPAIR queries
    vpair_imgs = sorted(glob(os.path.join(VPAIR_QUERY_DIR, '*.png')))
    for img_path in tqdm(vpair_imgs, desc='Sampling VPAIR crops'):
        crops.extend(sample_crops_from_image(img_path, CROP_SIZE, N_CROPS_PER_IMAGE))
    # Earth imagery
    for loc in range(1, 11):
        loc_dir = os.path.join(EARTH_IMAGERY_DIR, f'loc{loc}')
        for img_path in glob(os.path.join(loc_dir, '*.jpg')):
            crops.extend(sample_crops_from_image(img_path, CROP_SIZE, N_CROPS_PER_IMAGE))
    print(f"Total crops gathered: {len(crops)}")
    return crops

def main():
    crops = gather_all_crops()
    # Subsample if too many
    if len(crops) > N_VOCAB_CROPS:
        crops = random.sample(crops, N_VOCAB_CROPS)
    print(f"Using {len(crops)} crops for vocabulary.")
    # Extract dense features
    embedder = AnyLocVLADEmbedder(model_type=MODEL_TYPE, layer=LAYER, facet=FACET, device=DEVICE, n_clusters=N_CLUSTERS)
    print("Extracting dense features for all crops...")
    all_feats = []
    with torch.no_grad():
        for crop in tqdm(crops, desc='Extracting features'):
            feats = embedder.extract_dense_features(crop)  # [N_patches, D]
            all_feats.append(feats)
    all_feats = np.concatenate(all_feats, axis=0)
    print(f"Total features for K-means: {all_feats.shape}")
    # Subsample for K-means if needed
    if all_feats.shape[0] > N_VOCAB_CROPS * 10:
        idx = np.random.choice(all_feats.shape[0], N_VOCAB_CROPS * 10, replace=False)
        feats_for_kmeans = all_feats[idx]
    else:
        feats_for_kmeans = all_feats
    # Run K-means and save cluster centers
    print(f"Running K-means for VLAD vocabulary (clusters={N_CLUSTERS})...")
    embedder.fit_vocabulary_from_features(feats_for_kmeans)
    centers = embedder.vlad.c_centers.cpu()
    torch.save(centers, VOCAB_SAVE_PATH)
    print(f"Saved cluster centers to {VOCAB_SAVE_PATH} (shape: {centers.shape})")

if __name__ == '__main__':
    main() 