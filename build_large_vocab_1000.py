import os
import random
from glob import glob
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

# Settings - Large vocabulary with 1000 clusters
VPAIR_REFERENCE_DIR = 'third_party/vpair_sample/reference_views'
VPAIR_DISTRACTORS_DIR = 'third_party/vpair_sample/distractors'
CROP_SIZE = 224
N_CROPS_PER_IMAGE = 10  # More crops per image
N_VOCAB_CROPS = 50000  # Large vocabulary dataset
N_CLUSTERS = 1000  # Much larger cluster count as requested
VOCAB_SAVE_PATH = 'c_centers_1000.pt'
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

def gather_crops_from_directory(image_dir, n_crops_per_image, crop_size):
    """Gather random crops from all images in a directory."""
    crops = []
    image_files = glob(os.path.join(image_dir, '*.png')) + glob(os.path.join(image_dir, '*.jpg'))
    
    print(f"Found {len(image_files)} images")
    
    for img_path in tqdm(image_files, desc=f"Processing {os.path.basename(image_dir)}"):
        try:
            img = Image.open(img_path).convert('RGB')
            for _ in range(n_crops_per_image):
                crop = random_crop(img, crop_size)
                crops.append(crop)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return crops

def main():
    print("Seed set to: 42 (type: <class 'int'>)")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("============================================================")
    print("LARGE ANYLOC VLAD VOCABULARY GENERATION - 1000 CLUSTERS")
    print("============================================================")
    print(f"Target vocabulary size: {N_VOCAB_CROPS} crops")
    print(f"VLAD clusters: {N_CLUSTERS}")
    print(f"Using {MODEL_TYPE} with layer {LAYER}, facet '{FACET}'")
    print("============================================================")
    
    # Gather crops from VPair reference views
    print("Gathering crops from VPair reference views...")
    reference_crops = gather_crops_from_directory(
        VPAIR_REFERENCE_DIR, N_CROPS_PER_IMAGE, CROP_SIZE
    )
    
    # Gather crops from VPair distractors
    print("Gathering crops from VPair distractors...")
    distractor_crops = gather_crops_from_directory(
        VPAIR_DISTRACTORS_DIR, N_CROPS_PER_IMAGE, CROP_SIZE
    )
    
    # Combine all crops
    all_crops = reference_crops + distractor_crops
    print(f"Total crops gathered: {len(all_crops)}")
    
    # Sample crops if we have more than needed
    if len(all_crops) > N_VOCAB_CROPS:
        print(f"Sampling {N_VOCAB_CROPS} crops from {len(all_crops)}")
        all_crops = random.sample(all_crops, N_VOCAB_CROPS)
    else:
        print(f"Using all {len(all_crops)} crops for vocabulary")
    
    # Initialize AnyLoc embedder
    print("Initializing AnyLoc embedder...")
    embedder = AnyLocVLADEmbedder(
        model_type=MODEL_TYPE,
        layer=LAYER,
        facet=FACET,
        device=DEVICE,
        n_clusters=N_CLUSTERS  # Use the large cluster count
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
    
    # Subsample features for K-means (to manage memory)
    max_features_for_kmeans = 200000  # Increased for 1000 clusters
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
    print("LARGE VOCABULARY GENERATION COMPLETE")
    print("============================================================")
    print(f"Saved cluster centers to: {VOCAB_SAVE_PATH}")
    print(f"Cluster centers shape: {cluster_centers.shape}")
    print(f"Using {len(all_crops)} aerial crops and {N_CLUSTERS} clusters")
    print("This should provide much finer-grained feature representation!")
    print("============================================================")

if __name__ == "__main__":
    main() 