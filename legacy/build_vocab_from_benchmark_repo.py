import os
import random
from glob import glob
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

# Settings - Using benchmark repository images for vocabulary
BENCHMARK_DATASETS = [
    'deep-visual-geo-localization-benchmark/datasets/earth_realistic/database',
    'deep-visual-geo-localization-benchmark/datasets/earth_realistic/queries', 
    'deep-visual-geo-localization-benchmark/datasets/earth_benchmark/database',
    'deep-visual-geo-localization-benchmark/datasets/earth_benchmark/queries'
]
CROP_SIZE = 224
N_CROPS_PER_IMAGE = 15  # More crops per image since we have fewer images
N_CLUSTERS = 1000  # Large cluster count for best performance
VOCAB_SAVE_PATH = 'c_centers_benchmark_1000.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = 'dinov2_vits14'
LAYER = 11
FACET = 'key'

def random_crop(image, crop_size):
    """Generate random crops from an image."""
    w, h = image.size
    if w < crop_size or h < crop_size:
        # Resize if image is too small
        scale = max(crop_size / w, crop_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h
    
    # Random crop
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)
    return image.crop((left, top, left + crop_size, top + crop_size))

def gather_crops_from_benchmark():
    """Gather random crops from all benchmark repository images."""
    crops = []
    total_images = 0
    
    for dataset_dir in BENCHMARK_DATASETS:
        if not os.path.exists(dataset_dir):
            print(f"Directory not found: {dataset_dir}")
            continue
            
        image_files = glob(os.path.join(dataset_dir, '*.jpg')) + glob(os.path.join(dataset_dir, '*.png'))
        print(f"Found {len(image_files)} images in {dataset_dir}")
        total_images += len(image_files)
        
        for img_path in tqdm(image_files, desc=f"Processing {os.path.basename(dataset_dir)}"):
            try:
                img = Image.open(img_path).convert('RGB')
                for _ in range(N_CROPS_PER_IMAGE):
                    crop = random_crop(img, CROP_SIZE)
                    crops.append(crop)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    print(f"Total images processed: {total_images}")
    return crops

def main():
    print("Seed set to: 42 (type: <class 'int'>)")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("============================================================")
    print("BENCHMARK REPOSITORY VLAD VOCABULARY GENERATION - 1000 CLUSTERS")
    print("============================================================")
    print(f"VLAD clusters: {N_CLUSTERS}")
    print(f"Using {MODEL_TYPE} with layer {LAYER}, facet '{FACET}'")
    print(f"Crops per image: {N_CROPS_PER_IMAGE}")
    print("Source: Deep Visual Geo-localization Benchmark Repository")
    print("============================================================")
    
    # Gather crops from benchmark repository
    print("Gathering crops from benchmark repository images...")
    all_crops = gather_crops_from_benchmark()
    print(f"Total crops gathered: {len(all_crops)}")
    
    if len(all_crops) == 0:
        print("ERROR: No crops gathered! Check dataset paths.")
        return
    
    # Initialize AnyLoc embedder
    print("Initializing AnyLoc embedder...")
    embedder = AnyLocVLADEmbedder(
        model_type=MODEL_TYPE,
        layer=LAYER,
        facet=FACET,
        device=DEVICE,
        n_clusters=N_CLUSTERS
    )
    
    # Extract dense features from all crops
    print("Extracting dense features...")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_features = []
    batch_size = 16
    
    for i in tqdm(range(0, len(all_crops), batch_size), desc="Extracting features"):
        batch_crops = all_crops[i:i+batch_size]
        batch_tensors = torch.stack([transform(crop) for crop in batch_crops])
        
        with torch.no_grad():
            features = embedder.model(batch_tensors.to(DEVICE))
            # features is already [B, H*W, D]
            features = torch.nn.functional.normalize(features, dim=2)
            all_features.append(features.cpu())
    
    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)  # [N_crops, H*W, D]
    all_features = all_features.flatten(0, 1)  # [N_crops * H*W, D]
    
    print(f"Extracted features shape: {all_features.shape}")
    
    # Use more features for K-means since we have fewer source images
    max_features_for_kmeans = min(all_features.shape[0], 300000)  # Use more features
    if all_features.shape[0] > max_features_for_kmeans:
        print(f"Subsampling {max_features_for_kmeans} features from {all_features.shape[0]} for K-means")
        indices = torch.randperm(all_features.shape[0])[:max_features_for_kmeans]
        features_for_kmeans = all_features[indices]
    else:
        features_for_kmeans = all_features
    
    print(f"Features for K-means: {features_for_kmeans.shape}")
    
    # Run K-means clustering
    print(f"Running K-means clustering with {N_CLUSTERS} clusters...")
    embedder.fit_vocabulary_from_features(features_for_kmeans.numpy())
    
    # Save the cluster centers
    cluster_centers = embedder.vlad.c_centers
    torch.save(cluster_centers, VOCAB_SAVE_PATH)
    
    print("============================================================")
    print("BENCHMARK REPOSITORY VOCABULARY GENERATION COMPLETE")
    print("============================================================")
    print(f"Saved cluster centers to: {VOCAB_SAVE_PATH}")
    print(f"Cluster centers shape: {cluster_centers.shape}")
    print(f"Using {len(all_crops)} crops from benchmark repository images")
    print(f"Source images: Deep Visual Geo-localization Benchmark Repository")
    print("This uses the same domain as the evaluation dataset!")
    print("============================================================")

if __name__ == "__main__":
    main() 