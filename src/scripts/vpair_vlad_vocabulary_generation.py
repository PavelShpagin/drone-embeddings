#!/usr/bin/env python3
"""
VPAIR VLAD Vocabulary Generation Script

Generates a VLAD vocabulary using DINOv2 features from all VPAIR images.
Uses proper scale (600 images) with 1000 clusters for comprehensive vocabulary.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_dinov2_model():
    """Load DINOv2 model"""
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    return model

def extract_features_from_image(image_path, model, transform, device, patch_size=224, stride=112):
    """Extract DINOv2 features from image patches"""
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return []
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    features = []
    
    # Extract patches with sliding window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Transform and add batch dimension
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                feature = model(patch_tensor)
                features.append(feature.cpu().numpy().flatten())
    
    return features

def main():
    parser = argparse.ArgumentParser(description='Generate VLAD vocabulary for VPAIR dataset')
    parser.add_argument('--vpair_dir', default='third_party/vpair_sample', 
                       help='Path to VPAIR dataset directory')
    parser.add_argument('--output_dir', default='outputs/vpair_vlad', 
                       help='Output directory for vocabulary')
    parser.add_argument('--n_clusters', type=int, default=1000, 
                       help='Number of clusters for VLAD vocabulary')
    parser.add_argument('--max_features', type=int, default=500000, 
                       help='Maximum number of features to use for clustering')
    parser.add_argument('--patches_per_image', type=int, default=15, 
                       help='Maximum patches per image')
    
    args = parser.parse_args()
    
    # Setup paths
    vpair_dir = Path(args.vpair_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_dinov2_model().to(device)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Collect all image paths
    all_images = []
    
    # Add all images from all directories
    for subdir in ['queries', 'reference_views', 'distractors']:
        img_dir = vpair_dir / subdir
        if img_dir.exists():
            images = list(img_dir.glob('*.png'))
            all_images.extend(images)
            print(f"Found {len(images)} images in {subdir}")
    
    print(f"Total images for vocabulary generation: {len(all_images)}")
    
    # Extract features from all images
    print("Extracting features from all images...")
    all_features = []
    
    for img_path in tqdm(all_images, desc="Processing images"):
        features = extract_features_from_image(
            str(img_path), model, transform, device, 
            patch_size=224, stride=150  # Slightly larger stride for efficiency
        )
        
        # Limit patches per image
        if len(features) > args.patches_per_image:
            # Sample evenly distributed patches
            indices = np.linspace(0, len(features)-1, args.patches_per_image, dtype=int)
            features = [features[i] for i in indices]
        
        all_features.extend(features)
    
    print(f"Extracted {len(all_features)} total features")
    
    # Convert to numpy array
    features_array = np.array(all_features)
    print(f"Feature array shape: {features_array.shape}")
    
    # Subsample if too many features
    if len(features_array) > args.max_features:
        print(f"Subsampling {args.max_features} features from {len(features_array)}")
        indices = np.random.choice(len(features_array), args.max_features, replace=False)
        features_array = features_array[indices]
    
    # Perform K-means clustering
    print(f"Performing K-means clustering with {args.n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=args.n_clusters, 
        random_state=42, 
        n_init=10,
        max_iter=300,
        verbose=1
    )
    
    kmeans.fit(features_array)
    
    # Save vocabulary
    vocab_path = output_dir / f'vlad_vocabulary_{args.n_clusters}clusters.pkl'
    
    vocabulary_data = {
        'cluster_centers': kmeans.cluster_centers_,
        'n_clusters': args.n_clusters,
        'feature_dim': features_array.shape[1],
        'total_images_used': len(all_images),
        'total_features_used': len(features_array),
        'model_info': 'DINOv2-ViT-B/14'
    }
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocabulary_data, f)
    
    print(f"Vocabulary saved to: {vocab_path}")
    print(f"Vocabulary statistics:")
    print(f"  - Clusters: {args.n_clusters}")
    print(f"  - Feature dimension: {features_array.shape[1]}")
    print(f"  - Images used: {len(all_images)}")
    print(f"  - Features used: {len(features_array)}")
    
    # Save summary
    summary_path = output_dir / 'vocabulary_generation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"VPAIR VLAD Vocabulary Generation Summary\n")
        f.write(f"=======================================\n\n")
        f.write(f"Dataset: VPAIR (600 images total)\n")
        f.write(f"Model: DINOv2-ViT-B/14\n")
        f.write(f"Clusters: {args.n_clusters}\n")
        f.write(f"Feature dimension: {features_array.shape[1]}\n")
        f.write(f"Images used: {len(all_images)}\n")
        f.write(f"Features extracted: {len(features_array)}\n")
        f.write(f"Patches per image (max): {args.patches_per_image}\n")
        f.write(f"Vocabulary file: {vocab_path.name}\n")
    
    print(f"Summary saved to: {summary_path}")

if __name__ == '__main__':
    main() 