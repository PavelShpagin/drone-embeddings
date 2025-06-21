import os
import random
from glob import glob
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from train_encoder import SiameseNet, get_eval_transforms
from sklearn.cluster import KMeans

# Settings
VPAIR_QUERY_DIR = 'third_party/vpair_sample/queries'
CROP_SIZE = 224
N_CROPS_PER_IMAGE = 5
N_VOCAB_CROPS = 20000  # Total crops for K-means
N_CLUSTERS = 32
VOCAB_SAVE_PATH = 'efficientnet_vlad_centers.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transform for EfficientNet
transform = get_eval_transforms()


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
    vpair_imgs = sorted(glob(os.path.join(VPAIR_QUERY_DIR, '*.png')))
    for img_path in tqdm(vpair_imgs, desc='Sampling VPAIR crops'):
        crops.extend(sample_crops_from_image(img_path, CROP_SIZE, N_CROPS_PER_IMAGE))
    print(f"Total crops gathered: {len(crops)}")
    return crops

def main():
    crops = gather_all_crops()
    # Subsample if too many
    if len(crops) > N_VOCAB_CROPS:
        crops = random.sample(crops, N_VOCAB_CROPS)
    print(f"Using {len(crops)} crops for vocabulary.")
    # Extract features
    model = SiameseNet("efficientnet_b0").to(DEVICE)
    model.eval()
    all_feats = []
    with torch.no_grad():
        for crop in tqdm(crops, desc='Extracting features'):
            img = transform(crop).unsqueeze(0).to(DEVICE)
            # Get the projected feature map before NetVLAD
            features = model.feature_extractor(img)[-1]
            projected = model.projection(features)  # [1, C, H, W]
            projected = projected.squeeze(0).permute(1, 2, 0).reshape(-1, projected.shape[1])  # [N_patches, C]
            all_feats.append(projected.cpu().numpy())
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
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(feats_for_kmeans)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    torch.save(centers, VOCAB_SAVE_PATH)
    print(f"Saved cluster centers to {VOCAB_SAVE_PATH} (shape: {centers.shape})")

if __name__ == '__main__':
    main() 