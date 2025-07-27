#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

def load_images(image_dir):
    """Load all images from a directory."""
    images = []
    paths = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        for img_path in Path(image_dir).glob(ext):
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                paths.append(str(img_path))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, paths

def compute_similarities_concatenated(query_embeddings, db_embeddings):
    """Compute cosine similarities for concatenated VLAD."""
    # Normalize embeddings
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    db_embeddings = torch.nn.functional.normalize(db_embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarities = torch.mm(query_embeddings, db_embeddings.t())
    return similarities

def compute_similarities_chamfer(query_vlad_vectors, db_vlad_vectors):
    """Compute Chamfer similarities."""
    n_queries = len(query_vlad_vectors)
    n_db = len(db_vlad_vectors)
    
    similarities = np.zeros((n_queries, n_db))
    
    for i in tqdm(range(n_queries), desc="Computing Chamfer similarities"):
        for j in range(n_db):
            # Compute Chamfer similarity
            q_vecs = query_vlad_vectors[i]  # [K, D]
            d_vecs = db_vlad_vectors[j]     # [K, D]
            
            # Pairwise inner products: [K, K]
            sim_matrix = torch.mm(q_vecs, d_vecs.t())
            
            # For each query vector, find max similarity with any doc vector
            max_similarities = torch.max(sim_matrix, dim=1)[0]
            
            # Sum over all query vectors
            chamfer_score = torch.sum(max_similarities).item()
            similarities[i, j] = chamfer_score
    
    return similarities

def evaluate_method(embedder, db_images, query_images, method="concatenated"):
    """Evaluate a method (concatenated or chamfer)."""
    print(f"Evaluating {method} method...")
    
    if method == "concatenated":
        # Get concatenated embeddings
        print("Computing database embeddings...")
        db_embeddings = []
        for img in tqdm(db_images, desc="DB embeddings"):
            emb = embedder.get_embedding(img)
            db_embeddings.append(torch.from_numpy(emb))
        db_embeddings = torch.stack(db_embeddings)
        
        print("Computing query embeddings...")
        query_embeddings = []
        for img in tqdm(query_images, desc="Query embeddings"):
            emb = embedder.get_embedding(img)
            query_embeddings.append(torch.from_numpy(emb))
        query_embeddings = torch.stack(query_embeddings)
        
        # Compute similarities
        similarities = compute_similarities_concatenated(query_embeddings, db_embeddings)
        
    elif method == "chamfer":
        # Get VLAD cluster vectors
        print("Computing database VLAD vectors...")
        db_vlad_vectors = []
        for img in tqdm(db_images, desc="DB VLAD vectors"):
            vlad_vecs = embedder.get_vlad_vectors(img)
            db_vlad_vectors.append(torch.from_numpy(vlad_vecs))
        
        print("Computing query VLAD vectors...")
        query_vlad_vectors = []
        for img in tqdm(query_images, desc="Query VLAD vectors"):
            vlad_vecs = embedder.get_vlad_vectors(img)
            query_vlad_vectors.append(torch.from_numpy(vlad_vecs))
        
        # Compute Chamfer similarities
        similarities = compute_similarities_chamfer(query_vlad_vectors, db_vlad_vectors)
        similarities = torch.from_numpy(similarities)
    
    return similarities

def compute_recalls(similarities, ground_truth, recall_values=[1, 5, 10, 20]):
    """Compute recall metrics."""
    n_queries = similarities.shape[0]
    
    # Get top predictions for each query
    _, predictions = torch.topk(similarities, k=max(recall_values), dim=1)
    
    recalls = np.zeros(len(recall_values))
    
    for i in range(n_queries):
        gt_positives = ground_truth[i]
        pred = predictions[i].numpy()
        
        for j, k in enumerate(recall_values):
            if np.any(np.isin(pred[:k], gt_positives)):
                recalls[j:] += 1
                break
    
    recalls = recalls / n_queries * 100
    return recalls

def create_ground_truth(query_paths, db_paths):
    """Create ground truth based on location matching."""
    ground_truth = []
    
    for query_path in query_paths:
        query_name = os.path.basename(query_path)
        query_loc = query_name.split('_')[0]
        
        positives = []
        for i, db_path in enumerate(db_paths):
            db_name = os.path.basename(db_path)
            db_loc = db_name.split('_')[0]
            if db_loc == query_loc:
                positives.append(i)
        
        # If no exact match, use first image as positive
        if not positives:
            positives = [0]
        
        ground_truth.append(positives)
    
    return ground_truth

def main():
    parser = argparse.ArgumentParser(description="Improved DINOv2+VLAD benchmark with better vocabulary")
    parser.add_argument("--dataset_path", type=str, default="datasets/earth_benchmark",
                        help="Path to dataset")
    parser.add_argument("--vocab_path", type=str, default="c_centers_improved.pt",
                        help="Path to improved vocabulary file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--n_clusters", type=int, default=64, help="Number of VLAD clusters")
    args = parser.parse_args()
    
    print("="*80)
    print("IMPROVED DINOv2+VLAD BENCHMARK WITH FOUNDLOC-STYLE VOCABULARY")
    print("="*80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Vocabulary: {args.vocab_path}")
    print(f"Clusters: {args.n_clusters}")
    print("="*80)
    
    # Check if improved vocabulary exists
    if not os.path.exists(args.vocab_path):
        print(f"ERROR: Improved vocabulary not found at {args.vocab_path}")
        print("Please run: python build_improved_anyloc_vlad_vocab.py")
        return
    
    # Load dataset
    db_dir = os.path.join(args.dataset_path, "database")
    query_dir = os.path.join(args.dataset_path, "queries")
    
    if not os.path.exists(db_dir) or not os.path.exists(query_dir):
        print(f"ERROR: Dataset not found at {args.dataset_path}")
        print("Please run: python deep-visual-geo-localization-benchmark/create_earth_dataset.py")
        return
    
    print("Loading database images...")
    db_images, db_paths = load_images(db_dir)
    print(f"Loaded {len(db_images)} database images")
    
    print("Loading query images...")
    query_images, query_paths = load_images(query_dir)
    print(f"Loaded {len(query_images)} query images")
    
    # Create ground truth
    ground_truth = create_ground_truth(query_paths, db_paths)
    
    # Initialize embedder with improved vocabulary
    print("Initializing AnyLoc embedder with improved vocabulary...")
    embedder = AnyLocVLADEmbedder(device=args.device, n_clusters=args.n_clusters)
    
    # Load the improved vocabulary
    print(f"Loading improved vocabulary from {args.vocab_path}...")
    embedder.load_vocabulary(args.vocab_path)
    print("Improved vocabulary loaded successfully!")
    
    # Evaluate both methods
    results = {}
    
    for method in ["concatenated", "chamfer"]:
        print(f"\n{'='*50}")
        print(f"Evaluating {method.upper()} method with improved vocabulary")
        print('='*50)
        
        similarities = evaluate_method(embedder, db_images, query_images, method)
        recalls = compute_recalls(similarities, ground_truth)
        
        results[method] = recalls
        
        print(f"Results: R@1={recalls[0]:.1f}%, R@5={recalls[1]:.1f}%, R@10={recalls[2]:.1f}%, R@20={recalls[3]:.1f}%")
    
    # Print comparison
    print("\n" + "="*80)
    print("IMPROVED RESULTS COMPARISON")
    print("="*80)
    
    concat_recalls = results["concatenated"]
    chamfer_recalls = results["chamfer"]
    
    print(f"{'Method':<25} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'R@20':<8}")
    print("-" * 70)
    print(f"{'Concatenated VLAD':<25} {concat_recalls[0]:<8.1f} {concat_recalls[1]:<8.1f} {concat_recalls[2]:<8.1f} {concat_recalls[3]:<8.1f}")
    print(f"{'Chamfer Similarity':<25} {chamfer_recalls[0]:<8.1f} {chamfer_recalls[1]:<8.1f} {chamfer_recalls[2]:<8.1f} {chamfer_recalls[3]:<8.1f}")
    
    # Calculate improvements
    improvements = [(c - v) / v * 100 if v > 0 else 0 for c, v in zip(chamfer_recalls, concat_recalls)]
    print("-" * 70)
    print(f"{'Improvement (%)':<25} {improvements[0]:<8.1f} {improvements[1]:<8.1f} {improvements[2]:<8.1f} {improvements[3]:<8.1f}")
    
    print("\n" + "="*80)
    print("EXPECTED IMPROVEMENTS WITH FOUNDLOC-STYLE VOCABULARY:")
    print("- Much larger vocabulary dataset (50k+ crops vs 1k)")
    print("- Domain-specific aerial imagery vocabulary")
    print("- More VLAD clusters (64 vs 32)")
    print("- Better feature diversity from VPair dataset")
    print("- Should achieve 70-90%+ recall rates as reported in FoundLoc paper")
    print("="*80)

if __name__ == "__main__":
    main() 