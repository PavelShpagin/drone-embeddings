#!/usr/bin/env python3
"""
DenseUAV VLAD Evaluation Script

Evaluates visual place recognition on DenseUAV dataset using DINOv2+VLAD.
Compares concatenated VLAD vs Chamfer similarity approaches.
Uses test set: drone queries vs satellite gallery images.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_dinov2_model():
    """Load DINOv2 model"""
    print("Loading DINOv2 model...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    return model

def load_vocabulary(vocab_path):
    """Load VLAD vocabulary"""
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    return vocab_data

def extract_features_from_image(image_path, model, transform, device, patch_size=224, stride=112):
    """Extract DINOv2 features from image patches"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return []
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    features = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model(patch_tensor)
                features.append(feature.cpu().numpy().flatten())
    
    return np.array(features)

def compute_vlad_descriptor(features, cluster_centers):
    """Compute VLAD descriptor from features and cluster centers"""
    if len(features) == 0:
        return np.zeros(cluster_centers.shape[0] * cluster_centers.shape[1])
    
    # Assign features to clusters
    distances = np.linalg.norm(features[:, None] - cluster_centers[None, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    
    # Compute VLAD descriptor
    vlad = np.zeros((cluster_centers.shape[0], cluster_centers.shape[1]))
    
    for k in range(cluster_centers.shape[0]):
        # Find features assigned to cluster k
        mask = assignments == k
        if np.sum(mask) > 0:
            # Compute residuals
            residuals = features[mask] - cluster_centers[k]
            vlad[k] = np.sum(residuals, axis=0)
    
    # Flatten and L2 normalize
    vlad_flat = vlad.flatten()
    norm = np.linalg.norm(vlad_flat)
    if norm > 0:
        vlad_flat = vlad_flat / norm
    
    return vlad_flat

def compute_chamfer_similarity(features1, features2):
    """Compute Chamfer similarity between two sets of features"""
    if len(features1) == 0 or len(features2) == 0:
        return 0.0
    
    # Compute pairwise distances
    distances = np.linalg.norm(features1[:, None] - features2[None, :], axis=2)
    
    # Chamfer distance: average of minimum distances in both directions
    d1_to_d2 = np.mean(np.min(distances, axis=1))
    d2_to_d1 = np.mean(np.min(distances, axis=0))
    
    chamfer_distance = (d1_to_d2 + d2_to_d1) / 2
    
    # Convert to similarity (higher is better)
    chamfer_similarity = 1.0 / (1.0 + chamfer_distance)
    
    return chamfer_similarity

def parse_gps_file(gps_file):
    """Parse DenseUAV GPS file to extract coordinates"""
    gps_data = {}
    
    with open(gps_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                filepath = parts[0]
                # Extract coordinates - format: E120.33542972222222 N30.324272222222223
                lon_str = parts[1]  # E120.33542972222222
                lat_str = parts[2]  # N30.324272222222223
                
                # Remove E/N prefix and convert to float
                lon = float(lon_str[1:])
                lat = float(lat_str[1:])
                
                gps_data[filepath] = {'lon': lon, 'lat': lat}
    
    return gps_data

def compute_gps_distance(coord1, coord2):
    """Compute GPS distance in meters using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371000  # Earth radius in meters
    
    lat1, lon1 = radians(coord1['lat']), radians(coord1['lon'])
    lat2, lon2 = radians(coord2['lat']), radians(coord2['lon'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def collect_test_images(denseuav_dir, gps_data):
    """Collect test query and gallery images that have GPS coordinates"""
    denseuav_path = Path(denseuav_dir)
    
    # Since GPS file only has satellite images, we'll use satellite-to-satellite matching
    # Only use images that have GPS coordinates
    satellite_dir = denseuav_path / 'test' / 'gallery_satellite'
    images_with_gps = []
    
    for subdir in satellite_dir.iterdir():
        if subdir.is_dir():
            for img_path in subdir.glob('*.tif'):
                # Check if this image has GPS coordinates
                gps_key = f"test/satellite/{img_path.parent.name}/{img_path.name}"
                if gps_key in gps_data:
                    images_with_gps.append(img_path)
    
    print(f"Found {len(images_with_gps)} images with GPS coordinates")
    
    # Split into query and gallery (use every other directory as query)
    query_images = []
    gallery_images = []
    
    for img_path in images_with_gps:
        dir_num = int(img_path.parent.name)
        if dir_num % 2 == 0:  # Even directories as queries
            query_images.append(img_path)
        else:  # Odd directories as gallery
            gallery_images.append(img_path)
    
    return query_images, gallery_images

def evaluate_recall_at_k(similarities, query_gps, gallery_gps, distance_threshold=25.0):
    """Evaluate recall@K with distance threshold"""
    recalls = {1: [], 5: [], 10: [], 20: []}
    
    for query_path in similarities:
        # Map actual path to GPS file path format
        # GPS file uses test/satellite/XXXXXX/ format, we have test/gallery_satellite/XXXXXX/
        dir_name = query_path.parent.name
        file_name = query_path.name
        
        # GPS file uses test/satellite/ format
        possible_keys = [
            f"test/satellite/{dir_name}/{file_name}",
        ]
        
        query_coord = None
        for key in possible_keys:
            if key in query_gps:
                query_coord = query_gps[key]
                break
        
        if query_coord is None:
            # No GPS data for this query
            for k in recalls:
                recalls[k].append(0.0)
            continue
        
        # Get sorted retrieval results
        retrieval_results = similarities[query_path]
        sorted_results = sorted(retrieval_results.items(), key=lambda x: x[1], reverse=True)
        
        # Find ground truth matches within distance threshold
        ground_truth_matches = []
        for gallery_path in retrieval_results:
            gallery_dir_name = gallery_path.parent.name
            gallery_file_name = gallery_path.name
            
            # GPS file uses test/satellite/ format
            gallery_possible_keys = [
                f"test/satellite/{gallery_dir_name}/{gallery_file_name}",
            ]
            
            gallery_coord = None
            for key in gallery_possible_keys:
                if key in gallery_gps:
                    gallery_coord = gallery_gps[key]
                    break
            
            if gallery_coord is not None:
                distance = compute_gps_distance(query_coord, gallery_coord)
                if distance <= distance_threshold:
                    ground_truth_matches.append(gallery_path)
        
        if len(ground_truth_matches) == 0:
            # No ground truth matches within threshold
            for k in recalls:
                recalls[k].append(0.0)
            continue
        
        # Check recall at different K values
        for k in recalls:
            top_k_results = [result[0] for result in sorted_results[:k]]
            
            # Check if any ground truth match is in top-K
            found_match = any(gt_match in top_k_results for gt_match in ground_truth_matches)
            recalls[k].append(1.0 if found_match else 0.0)
    
    # Compute average recalls
    avg_recalls = {}
    for k in recalls:
        if len(recalls[k]) > 0:
            avg_recalls[k] = np.mean(recalls[k]) * 100  # Convert to percentage
        else:
            avg_recalls[k] = 0.0
    
    return avg_recalls

def main():
    parser = argparse.ArgumentParser(description='Evaluate DenseUAV dataset with VLAD')
    parser.add_argument('--denseuav_dir', default='datasets/DenseUAV', 
                       help='Path to DenseUAV dataset directory')
    parser.add_argument('--vocab_path', required=True,
                       help='Path to VLAD vocabulary file')
    parser.add_argument('--output_dir', default='outputs/denseuav_evaluation', 
                       help='Output directory for results')
    parser.add_argument('--distance_threshold', type=float, default=25.0,
                       help='Distance threshold for positive matches (meters)')
    parser.add_argument('--max_queries', type=int, default=200,
                       help='Maximum number of queries to evaluate')
    parser.add_argument('--max_gallery', type=int, default=500,
                       help='Maximum number of gallery images')
    
    args = parser.parse_args()
    
    # Setup paths
    denseuav_dir = Path(args.denseuav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not denseuav_dir.exists():
        print(f"Error: DenseUAV directory not found at {denseuav_dir}")
        return
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print("Loading VLAD vocabulary...")
    vocab_data = load_vocabulary(args.vocab_path)
    cluster_centers = vocab_data['cluster_centers']
    print(f"Loaded vocabulary with {vocab_data['n_clusters']} clusters")
    
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
    
    # Load GPS data
    print("Loading GPS coordinates...")
    gps_file = denseuav_dir / 'Dense_GPS_test.txt'
    gps_data = parse_gps_file(gps_file)
    print(f"Loaded GPS data for {len(gps_data)} images")
    
    # Collect test images
    print("Collecting test images...")
    query_images, gallery_images = collect_test_images(denseuav_dir, gps_data)
    
    # Subsample if too many
    if len(query_images) > args.max_queries:
        np.random.seed(42)
        indices = np.random.choice(len(query_images), args.max_queries, replace=False)
        query_images = [query_images[i] for i in indices]
    
    if len(gallery_images) > args.max_gallery:
        np.random.seed(42)
        indices = np.random.choice(len(gallery_images), args.max_gallery, replace=False)
        gallery_images = [gallery_images[i] for i in indices]
    
    print(f"Evaluating {len(query_images)} queries against {len(gallery_images)} gallery images")
    
    # Debug: Check if any GPS coordinates match our images
    sample_query = query_images[0] if query_images else None
    sample_gallery = gallery_images[0] if gallery_images else None
    
    if sample_query:
        dir_name = sample_query.parent.name
        file_name = sample_query.name
        test_keys = [f"test/satellite/{dir_name}/{file_name}", f"test/gallery_satellite/{dir_name}/{file_name}"]
        print(f"Sample query: {sample_query}")
        print(f"Testing GPS keys: {test_keys}")
        for key in test_keys:
            if key in gps_data:
                print(f"✓ Found GPS for: {key}")
                break
        else:
            print("✗ No GPS match found for sample query")
            print(f"Available GPS keys (first 5): {list(gps_data.keys())[:5]}")
    
    if sample_gallery:
        dir_name = sample_gallery.parent.name
        file_name = sample_gallery.name
        test_keys = [f"test/satellite/{dir_name}/{file_name}", f"test/gallery_satellite/{dir_name}/{file_name}"]
        print(f"Sample gallery: {sample_gallery}")
        for key in test_keys:
            if key in gps_data:
                print(f"✓ Found GPS for: {key}")
                break
        else:
            print("✗ No GPS match found for sample gallery")
    
    # Extract features for all gallery images
    print("Extracting features from gallery images...")
    gallery_features = {}
    gallery_vlad_descriptors = {}
    
    for gallery_path in tqdm(gallery_images, desc="Processing gallery"):
        features = extract_features_from_image(
            str(gallery_path), model, transform, device
        )
        gallery_features[gallery_path] = features
        
        # Compute VLAD descriptor
        vlad_desc = compute_vlad_descriptor(features, cluster_centers)
        gallery_vlad_descriptors[gallery_path] = vlad_desc
    
    # Evaluate each query
    print("Evaluating queries...")
    vlad_similarities = {}
    chamfer_similarities = {}
    
    for query_path in tqdm(query_images, desc="Processing queries"):
        # Extract query features
        query_features = extract_features_from_image(
            str(query_path), model, transform, device
        )
        
        if len(query_features) == 0:
            continue
        
        # Compute query VLAD descriptor
        query_vlad = compute_vlad_descriptor(query_features, cluster_centers)
        
        # Compare with all gallery images
        vlad_similarities[query_path] = {}
        chamfer_similarities[query_path] = {}
        
        for gallery_path in gallery_images:
            # VLAD similarity (cosine)
            gallery_vlad = gallery_vlad_descriptors[gallery_path]
            vlad_sim = cosine_similarity([query_vlad], [gallery_vlad])[0, 0]
            vlad_similarities[query_path][gallery_path] = vlad_sim
            
            # Chamfer similarity
            gallery_features_array = gallery_features[gallery_path]
            chamfer_sim = compute_chamfer_similarity(query_features, gallery_features_array)
            chamfer_similarities[query_path][gallery_path] = chamfer_sim
    
    # Evaluate recalls
    print("Computing recall metrics...")
    
    vlad_recalls = evaluate_recall_at_k(
        vlad_similarities, gps_data, gps_data, args.distance_threshold
    )
    
    chamfer_recalls = evaluate_recall_at_k(
        chamfer_similarities, gps_data, gps_data, args.distance_threshold
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("DENSEUAV EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\nDataset Statistics:")
    print(f"  - Queries: {len(query_images)}")
    print(f"  - Gallery: {len(gallery_images)}")
    print(f"  - Distance threshold: {args.distance_threshold}m")
    print(f"  - Vocabulary clusters: {vocab_data['n_clusters']}")
    
    print(f"\nConcatenated VLAD Results:")
    for k in [1, 5, 10, 20]:
        if k in vlad_recalls:
            print(f"  Recall@{k:2d}: {vlad_recalls[k]:5.1f}%")
    
    print(f"\nChamfer Similarity Results:")
    for k in [1, 5, 10, 20]:
        if k in chamfer_recalls:
            print(f"  Recall@{k:2d}: {chamfer_recalls[k]:5.1f}%")
    
    # Save results
    results = {
        'concatenated_vlad': vlad_recalls,
        'chamfer_similarity': chamfer_recalls,
        'dataset_info': {
            'total_queries': len(query_images),
            'total_gallery': len(gallery_images),
            'distance_threshold': args.distance_threshold,
            'vocabulary_clusters': vocab_data['n_clusters']
        }
    }
    
    results_file = output_dir / 'denseuav_evaluation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary
    summary_file = output_dir / 'denseuav_evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("DenseUAV VLAD Evaluation Results\n")
        f.write("===============================\n\n")
        f.write(f"Queries: {len(query_images)}\n")
        f.write(f"Gallery: {len(gallery_images)}\n")
        f.write(f"Distance threshold: {args.distance_threshold}m\n")
        f.write(f"Vocabulary clusters: {vocab_data['n_clusters']}\n\n")
        
        f.write("Concatenated VLAD Results:\n")
        for k in [1, 5, 10, 20]:
            if k in vlad_recalls:
                f.write(f"  Recall@{k:2d}: {vlad_recalls[k]:5.1f}%\n")
        
        f.write("\nChamfer Similarity Results:\n")
        for k in [1, 5, 10, 20]:
            if k in chamfer_recalls:
                f.write(f"  Recall@{k:2d}: {chamfer_recalls[k]:5.1f}%\n")
    
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {summary_file}")

if __name__ == '__main__':
    main() 