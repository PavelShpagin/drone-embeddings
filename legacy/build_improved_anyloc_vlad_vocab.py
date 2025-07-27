import os
import random
from glob import glob
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms as T
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

# Settings - Improved based on FoundLoc paper recommendations
VPAIR_REFERENCE_DIR = 'third_party/vpair_sample/reference_views'
VPAIR_DISTRACTORS_DIR = 'third_party/vpair_sample/distractors'
CROP_SIZE = 224
N_CROPS_PER_IMAGE = 10  # More crops per image
N_VOCAB_CROPS = 50000  # Much larger vocabulary dataset (FoundLoc uses 50k+)
N_CLUSTERS = 64  # More clusters as recommended in FoundLoc
VOCAB_SAVE_PATH = 'c_centers_improved.pt'
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
    """Sample random crops from an image."""
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # Skip images that are too small
        if w < crop_size or h < crop_size:
            return []
            
        crops = []
        for _ in range(n_crops):
            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)
            crop = img.crop((x, y, x + crop_size, y + crop_size))
            crops.append(crop)
        return crops
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return []

def gather_aerial_vocabulary_crops():
    """Gather crops specifically for aerial vocabulary generation using VPair dataset."""
    crops = []
    
    print("Gathering crops from VPair reference views...")
    # VPair reference views (aerial imagery)
    vpair_ref_imgs = sorted(glob(os.path.join(VPAIR_REFERENCE_DIR, '*.png')))
    print(f"Found {len(vpair_ref_imgs)} reference images")
    
    for img_path in tqdm(vpair_ref_imgs, desc='Processing VPair reference views'):
        img_crops = sample_crops_from_image(img_path, CROP_SIZE, N_CROPS_PER_IMAGE)
        crops.extend(img_crops)
    
    print("Gathering crops from VPair distractors...")
    # VPair distractors (more aerial imagery for diversity)
    vpair_distractor_imgs = sorted(glob(os.path.join(VPAIR_DISTRACTORS_DIR, '*.png')))
    print(f"Found {len(vpair_distractor_imgs)} distractor images")
    
    # Use a subset of distractors to avoid overwhelming the vocabulary
    max_distractors = min(1000, len(vpair_distractor_imgs))
    selected_distractors = random.sample(vpair_distractor_imgs, max_distractors)
    
    for img_path in tqdm(selected_distractors, desc='Processing VPair distractors'):
        img_crops = sample_crops_from_image(img_path, CROP_SIZE, N_CROPS_PER_IMAGE)
        crops.extend(img_crops)
    
    print(f"Total crops gathered: {len(crops)}")
    return crops

def extract_features_in_batches(embedder, crops, batch_size=32):
    """Extract features in batches to handle memory efficiently."""
    all_feats = []
    
    for i in tqdm(range(0, len(crops), batch_size), desc='Extracting features'):
        batch_crops = crops[i:i+batch_size]
        batch_feats = []
        
        with torch.no_grad():
            for crop in batch_crops:
                feats = embedder.extract_dense_features(crop)  # [N_patches, D]
                batch_feats.append(feats)
        
        # Concatenate batch features
        if batch_feats:
            batch_feats = np.concatenate(batch_feats, axis=0)
            all_feats.append(batch_feats)
    
    # Concatenate all batches
    if all_feats:
        all_feats = np.concatenate(all_feats, axis=0)
    else:
        all_feats = np.array([])
    
    return all_feats

def main():
    print("="*60)
    print("IMPROVED ANYLOC VLAD VOCABULARY GENERATION")
    print("="*60)
    print(f"Target vocabulary size: {N_VOCAB_CROPS} crops")
    print(f"VLAD clusters: {N_CLUSTERS}")
    print(f"Using {MODEL_TYPE} with layer {LAYER}, facet '{FACET}'")
    print("="*60)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Gather crops from aerial imagery
    crops = gather_aerial_vocabulary_crops()
    
    if len(crops) == 0:
        print("Error: No crops gathered! Check that VPair dataset exists.")
        return
    
    # Subsample if we have too many crops
    if len(crops) > N_VOCAB_CROPS:
        print(f"Subsampling {N_VOCAB_CROPS} crops from {len(crops)} total crops")
        crops = random.sample(crops, N_VOCAB_CROPS)
    else:
        print(f"Using all {len(crops)} crops for vocabulary")
    
    # Initialize embedder
    print(f"Initializing AnyLoc embedder...")
    embedder = AnyLocVLADEmbedder(
        model_type=MODEL_TYPE, 
        layer=LAYER, 
        facet=FACET, 
        device=DEVICE, 
        n_clusters=N_CLUSTERS
    )
    
    # Extract dense features in batches
    print("Extracting dense features...")
    all_feats = extract_features_in_batches(embedder, crops, batch_size=16)
    
    if all_feats.size == 0:
        print("Error: No features extracted!")
        return
    
    print(f"Extracted features shape: {all_feats.shape}")
    
    # Subsample features for K-means if needed (limit to 100k features for memory)
    max_kmeans_feats = 100000
    if all_feats.shape[0] > max_kmeans_feats:
        print(f"Subsampling {max_kmeans_feats} features from {all_feats.shape[0]} for K-means")
        idx = np.random.choice(all_feats.shape[0], max_kmeans_feats, replace=False)
        feats_for_kmeans = all_feats[idx]
    else:
        feats_for_kmeans = all_feats
    
    print(f"Features for K-means: {feats_for_kmeans.shape}")
    
    # Run K-means and save cluster centers
    print(f"Running K-means clustering with {N_CLUSTERS} clusters...")
    embedder.fit_vocabulary_from_features(feats_for_kmeans)
    
    # Save cluster centers
    centers = embedder.vlad.c_centers.cpu()
    torch.save(centers, VOCAB_SAVE_PATH)
    
    print("="*60)
    print("VOCABULARY GENERATION COMPLETE")
    print("="*60)
    print(f"Saved cluster centers to: {VOCAB_SAVE_PATH}")
    print(f"Cluster centers shape: {centers.shape}")
    print(f"Expected improvement: Using {len(crops)} aerial crops and {N_CLUSTERS} clusters")
    print("This should significantly improve recall performance!")
    print("="*60)

if __name__ == '__main__':
    main() 