#!/usr/bin/env python3
"""
VPAIR VLAD Evaluation Script

Evaluates visual place recognition on VPAIR dataset using DINOv2+VLAD.
Compares concatenated VLAD vs Chamfer similarity approaches.
Uses ground truth GPS coordinates for proper distance-based evaluation.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import pickle
import pandas as pd
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

def load_poses(poses_file):
    """Load poses from CSV file"""
    df = pd.read_csv(poses_file)
    poses_dict = {}
    
    for _, row in df.iterrows():
        filename = os.path.basename(row['filepath'])
        poses_dict[filename] = {
            'x': row['x'],
            'y': row['y'],
            'z': row['z']
        }
    
    return poses_dict

def compute_distance(pose1, pose2):
    """Compute Euclidean distance between two poses"""
    dx = pose1['x'] - pose2['x']
    dy = pose1['y'] - pose2['y']
    dz = pose1['z'] - pose2['z']
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def evaluate_recall_at_k(query_poses, reference_poses, similarities, distance_threshold=25.0):
    """Evaluate recall@K with distance threshold"""
    recalls = {1: [], 5: [], 10: [], 20: []}
    
    for i, query_filename in enumerate(similarities.keys()):
        if query_filename not in query_poses:
            continue
            
        query_pose = query_poses[query_filename]
        
        # Get sorted retrieval results
        retrieval_results = similarities[query_filename]
        sorted_results = sorted(retrieval_results.items(), key=lambda x: x[1], reverse=True)
        
        # Find ground truth matches within distance threshold
        ground_truth_matches = []
        for ref_filename in reference_poses:
            if ref_filename in reference_poses:
                ref_pose = reference_poses[ref_filename]
                distance = compute_distance(query_pose, ref_pose)
                if distance <= distance_threshold:
                    ground_truth_matches.append(ref_filename)
        
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
    parser = argparse.ArgumentParser(description='Evaluate VPAIR dataset with VLAD')
    parser.add_argument('--vpair_dir', default='third_party/vpair_sample', 
                       help='Path to VPAIR dataset directory')
    parser.add_argument('--vocab_path', required=True,
                       help='Path to VLAD vocabulary file')
    parser.add_argument('--output_dir', default='outputs/vpair_evaluation', 
                       help='Output directory for results')
    parser.add_argument('--distance_threshold', type=float, default=25.0,
                       help='Distance threshold for positive matches (meters)')
    
    args = parser.parse_args()
    
    # Setup paths
    vpair_dir = Path(args.vpair_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and vocabulary
    model = load_dinov2_model().to(device)
    vocab_data = load_vocabulary(args.vocab_path)
    cluster_centers = vocab_data['cluster_centers']
    
    print(f"Loaded vocabulary with {vocab_data['n_clusters']} clusters")
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load poses
    query_poses = load_poses(vpair_dir / 'poses_query.txt')
    reference_poses = load_poses(vpair_dir / 'poses_reference_view.txt')
    
    print(f"Loaded {len(query_poses)} query poses and {len(reference_poses)} reference poses")
    
    # Process query images
    print("Processing query images...")
    query_features = {}
    query_vlad_descriptors = {}
    
    query_dir = vpair_dir / 'queries'
    for img_path in tqdm(list(query_dir.glob('*.png')), desc="Query images"):
        filename = img_path.name
        
        features = extract_features_from_image(
            str(img_path), model, transform, device, stride=150
        )
        
        if len(features) > 0:
            query_features[filename] = features
            vlad_desc = compute_vlad_descriptor(features, cluster_centers)
            query_vlad_descriptors[filename] = vlad_desc
    
    # Process reference images
    print("Processing reference images...")
    reference_features = {}
    reference_vlad_descriptors = {}
    
    reference_dir = vpair_dir / 'reference_views'
    for img_path in tqdm(list(reference_dir.glob('*.png')), desc="Reference images"):
        filename = img_path.name
        
        features = extract_features_from_image(
            str(img_path), model, transform, device, stride=150
        )
        
        if len(features) > 0:
            reference_features[filename] = features
            vlad_desc = compute_vlad_descriptor(features, cluster_centers)
            reference_vlad_descriptors[filename] = vlad_desc
    
    print(f"Processed {len(query_vlad_descriptors)} queries and {len(reference_vlad_descriptors)} references")
    
    # Evaluate Concatenated VLAD
    print("Evaluating Concatenated VLAD...")
    vlad_similarities = {}
    
    for query_filename, query_vlad in tqdm(query_vlad_descriptors.items(), desc="VLAD similarities"):
        vlad_similarities[query_filename] = {}
        
        for ref_filename, ref_vlad in reference_vlad_descriptors.items():
            # Cosine similarity between VLAD descriptors
            similarity = cosine_similarity([query_vlad], [ref_vlad])[0, 0]
            vlad_similarities[query_filename][ref_filename] = similarity
    
    vlad_recalls = evaluate_recall_at_k(
        query_poses, reference_poses, vlad_similarities, args.distance_threshold
    )
    
    # Evaluate Chamfer Similarity
    print("Evaluating Chamfer Similarity...")
    chamfer_similarities = {}
    
    for query_filename, query_feats in tqdm(query_features.items(), desc="Chamfer similarities"):
        chamfer_similarities[query_filename] = {}
        
        for ref_filename, ref_feats in reference_features.items():
            similarity = compute_chamfer_similarity(query_feats, ref_feats)
            chamfer_similarities[query_filename][ref_filename] = similarity
    
    chamfer_recalls = evaluate_recall_at_k(
        query_poses, reference_poses, chamfer_similarities, args.distance_threshold
    )
    
    # Print results
    print("\n" + "="*60)
    print("VPAIR EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {len(query_vlad_descriptors)} queries, {len(reference_vlad_descriptors)} references")
    print(f"Distance threshold: {args.distance_threshold}m")
    print(f"Vocabulary: {vocab_data['n_clusters']} clusters")
    print()
    
    print("Concatenated VLAD Results:")
    for k in [1, 5, 10, 20]:
        print(f"  Recall@{k:2d}: {vlad_recalls[k]:5.1f}%")
    
    print()
    print("Chamfer Similarity Results:")
    for k in [1, 5, 10, 20]:
        print(f"  Recall@{k:2d}: {chamfer_recalls[k]:5.1f}%")
    
    # Save detailed results
    results = {
        'dataset_info': {
            'total_queries': len(query_vlad_descriptors),
            'total_references': len(reference_vlad_descriptors),
            'distance_threshold': args.distance_threshold,
            'vocabulary_clusters': vocab_data['n_clusters']
        },
        'concatenated_vlad': vlad_recalls,
        'chamfer_similarity': chamfer_recalls
    }
    
    results_file = output_dir / 'vpair_evaluation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save text summary
    summary_file = output_dir / 'vpair_evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("VPAIR Visual Place Recognition Evaluation\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Dataset: {len(query_vlad_descriptors)} queries, {len(reference_vlad_descriptors)} references\n")
        f.write(f"Distance threshold: {args.distance_threshold}m\n")
        f.write(f"Vocabulary: {vocab_data['n_clusters']} clusters\n")
        f.write(f"Model: DINOv2-ViT-B/14\n\n")
        
        f.write("Concatenated VLAD Results:\n")
        for k in [1, 5, 10, 20]:
            f.write(f"  Recall@{k:2d}: {vlad_recalls[k]:5.1f}%\n")
        
        f.write("\nChamfer Similarity Results:\n")
        for k in [1, 5, 10, 20]:
            f.write(f"  Recall@{k:2d}: {chamfer_recalls[k]:5.1f}%\n")
    
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {summary_file}")

if __name__ == '__main__':
    main() 