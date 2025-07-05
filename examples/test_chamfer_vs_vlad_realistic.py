import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt
from glob import glob

# Import the updated VLAD embedder
import sys
sys.path.append(str(Path(__file__).parent.parent))
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

# Settings
CROP_SIZE = 224
N_CLUSTERS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "dinov2_vits14"
LAYER = 11
FACET = "key"

# Use multiple Earth imagery locations for more realistic test
DB_LOCATIONS = [
    "data/earth_imagery/loc1",
    "data/earth_imagery/loc2", 
    "data/earth_imagery/loc3",
    "data/earth_imagery/loc4",
    "data/earth_imagery/loc5"
]

QUERY_LOCATIONS = [
    "data/earth_imagery/loc6",
    "data/earth_imagery/loc7",
    "data/earth_imagery/loc8",
    "data/earth_imagery/loc9",
    "data/earth_imagery/loc10"
]

def extract_random_crops(image_paths, n_crops_per_image=20):
    """Extract random crops from multiple images."""
    crops = []
    labels = []  # Track which location each crop comes from
    
    for i, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            continue
            
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        if w < CROP_SIZE or h < CROP_SIZE:
            continue
            
        for _ in range(n_crops_per_image):
            x = random.randint(0, w - CROP_SIZE)
            y = random.randint(0, h - CROP_SIZE)
            crop = img.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))
            crops.append(crop)
            labels.append(i)  # Location index
    
    return crops, labels

def get_image_paths(location_dirs):
    """Get all image paths from location directories."""
    image_paths = []
    for loc_dir in location_dirs:
        if os.path.exists(loc_dir):
            paths = glob(os.path.join(loc_dir, "*.jpg"))
            if paths:
                image_paths.append(paths[0])  # Take first image from each location
    return image_paths

def evaluate_method(embedder, db_crops, db_labels, query_crops, query_labels, method="vlad"):
    """
    Evaluate either VLAD dot product or Chamfer similarity method.
    For this test, we consider a match "correct" if it retrieves from the same location.
    """
    print(f"\n=== Evaluating {method.upper()} method ===")
    
    if method == "vlad":
        # Extract concatenated VLAD embeddings
        print("Extracting VLAD embeddings for database...")
        db_embeddings = [embedder.get_embedding(crop) for crop in tqdm(db_crops)]
        db_embeddings = np.stack(db_embeddings)
        
        print("Extracting VLAD embeddings for queries...")
        query_embeddings = [embedder.get_embedding(crop) for crop in tqdm(query_crops)]
        query_embeddings = np.stack(query_embeddings)
        
        # Compute similarities using dot product
        print("Computing similarities...")
        similarities = np.dot(query_embeddings, db_embeddings.T)  # [n_query, n_db]
        
    elif method == "chamfer":
        # Extract individual VLAD cluster vectors
        print("Extracting VLAD cluster vectors for database...")
        db_vlad_vectors = [embedder.get_vlad_vectors(crop) for crop in tqdm(db_crops)]
        
        print("Extracting VLAD cluster vectors for queries...")
        query_vlad_vectors = [embedder.get_vlad_vectors(crop) for crop in tqdm(query_crops)]
        
        # Compute similarities using Chamfer similarity
        print("Computing Chamfer similarities...")
        n_queries = len(query_vlad_vectors)
        n_db = len(db_vlad_vectors)
        similarities = np.zeros((n_queries, n_db))
        
        for i, query_vecs in enumerate(tqdm(query_vlad_vectors, desc="Query similarities")):
            for j, db_vecs in enumerate(db_vlad_vectors):
                similarities[i, j] = embedder.chamfer_similarity(query_vecs, db_vecs)
    
    # Evaluate "location-based" recall
    # A retrieval is correct if the top-k results include crops from the same location
    recall1_count = 0
    recall5_count = 0
    n_queries = len(query_labels)
    
    print("Computing location-based recall...")
    for i in tqdm(range(n_queries), desc="Recall evaluation"):
        query_location = query_labels[i]
        
        # Get top-5 most similar database crops
        top5_indices = np.argsort(similarities[i])[-5:][::-1]  # Descending order
        
        # Recall@1: Check if top result is from same location
        top1_location = db_labels[top5_indices[0]]
        if top1_location == query_location:
            recall1_count += 1
        
        # Recall@5: Check if any of top-5 results are from same location
        top5_locations = [db_labels[idx] for idx in top5_indices]
        if query_location in top5_locations:
            recall5_count += 1
    
    recall1 = recall1_count / n_queries
    recall5 = recall5_count / n_queries
    
    print(f"{method.upper()} Results:")
    print(f"  Recall@1: {recall1:.4f} ({recall1_count}/{n_queries})")
    print(f"  Recall@5: {recall5:.4f} ({recall5_count}/{n_queries})")
    
    return recall1, recall5, similarities

def main():
    print("Realistic Chamfer Similarity vs VLAD Dot Product Comparison")
    print("Using different locations for database and queries")
    print("=" * 60)
    
    # Get image paths
    db_image_paths = get_image_paths(DB_LOCATIONS)
    query_image_paths = get_image_paths(QUERY_LOCATIONS)
    
    print(f"Database locations: {len(db_image_paths)}")
    print(f"Query locations: {len(query_image_paths)}")
    
    if len(db_image_paths) == 0 or len(query_image_paths) == 0:
        print("Error: Not enough image locations found!")
        print("Make sure data/earth_imagery/loc1-loc10 directories exist with images")
        return
    
    # Extract crops
    print("Extracting database crops...")
    db_crops, db_labels = extract_random_crops(db_image_paths, n_crops_per_image=30)
    
    print("Extracting query crops...")
    query_crops, query_labels = extract_random_crops(query_image_paths, n_crops_per_image=20)
    
    print(f"Database crops: {len(db_crops)} from {len(db_image_paths)} locations")
    print(f"Query crops: {len(query_crops)} from {len(query_image_paths)} locations")
    
    if len(db_crops) == 0 or len(query_crops) == 0:
        print("Error: No crops extracted!")
        return
    
    # Initialize embedder
    print("Initializing AnyLoc VLAD embedder...")
    embedder = AnyLocVLADEmbedder(
        model_type=MODEL_TYPE,
        layer=LAYER,
        facet=FACET,
        device=DEVICE,
        n_clusters=N_CLUSTERS
    )
    
    # Build vocabulary from database crops
    vocab_crops = random.sample(db_crops, min(100, len(db_crops)))
    print("Fitting VLAD vocabulary...")
    embedder.fit_vocabulary(vocab_crops)
    
    # Test both methods
    results = {}
    
    # Method 1: Concatenated VLAD with dot product
    vlad_recall1, vlad_recall5, vlad_similarities = evaluate_method(
        embedder, db_crops, db_labels, query_crops, query_labels, method="vlad"
    )
    results["vlad"] = {"recall1": vlad_recall1, "recall5": vlad_recall5}
    
    # Method 2: Chamfer similarity
    chamfer_recall1, chamfer_recall5, chamfer_similarities = evaluate_method(
        embedder, db_crops, db_labels, query_crops, query_labels, method="chamfer"
    )
    results["chamfer"] = {"recall1": chamfer_recall1, "recall5": chamfer_recall5}
    
    # Print comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON - Location-Based Recall")
    print("=" * 70)
    print(f"{'Method':<15} {'Recall@1':<12} {'Recall@5':<12} {'Improvement'}")
    print("-" * 70)
    print(f"{'VLAD Dot Prod':<15} {vlad_recall1:<12.4f} {vlad_recall5:<12.4f} {'Baseline'}")
    
    r1_improvement = (chamfer_recall1 - vlad_recall1) / vlad_recall1 * 100 if vlad_recall1 > 0 else 0
    r5_improvement = (chamfer_recall5 - vlad_recall5) / vlad_recall5 * 100 if vlad_recall5 > 0 else 0
    
    print(f"{'Chamfer Sim':<15} {chamfer_recall1:<12.4f} {chamfer_recall5:<12.4f} {r1_improvement:+.1f}%/{r5_improvement:+.1f}%")
    
    # Determine winner
    if chamfer_recall1 > vlad_recall1:
        print(f"\n🏆 WINNER: Chamfer Similarity (+{r1_improvement:.1f}% Recall@1)")
    elif vlad_recall1 > chamfer_recall1:
        print(f"\n🏆 WINNER: VLAD Dot Product (+{-r1_improvement:.1f}% Recall@1)")
    else:
        print(f"\n🤝 TIE: Both methods perform equally")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    methods = ['VLAD\nDot Product', 'Chamfer\nSimilarity']
    recall1_scores = [vlad_recall1, chamfer_recall1]
    recall5_scores = [vlad_recall5, chamfer_recall5]
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Colors for better visualization
    colors = ['#1f77b4', '#ff7f0e']
    
    bars1 = ax1.bar(x - width/2, recall1_scores, width, label='Recall@1', alpha=0.8, color=colors[0])
    bars2 = ax1.bar(x + width/2, recall5_scores, width, label='Recall@5', alpha=0.8, color=colors[1])
    
    ax1.set_ylabel('Recall Score')
    ax1.set_title('Location-Based Recall Comparison:\nVLAD vs Chamfer Similarity')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (r1, r5) in enumerate(zip(recall1_scores, recall5_scores)):
        ax1.text(i - width/2, r1 + 0.01, f'{r1:.3f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, r5 + 0.01, f'{r5:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Similarity distribution comparison
    ax2.hist(vlad_similarities.flatten(), bins=50, alpha=0.6, label='VLAD Dot Product', 
             density=True, color=colors[0])
    ax2.hist(chamfer_similarities.flatten(), bins=50, alpha=0.6, label='Chamfer Similarity', 
             density=True, color=colors[1])
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Similarity Score Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_chamfer_vs_vlad_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to realistic_chamfer_vs_vlad_comparison.png")
    
    # Additional analysis
    print(f"\nAdditional Statistics:")
    print(f"VLAD similarities - Mean: {np.mean(vlad_similarities):.4f}, Std: {np.std(vlad_similarities):.4f}")
    print(f"Chamfer similarities - Mean: {np.mean(chamfer_similarities):.4f}, Std: {np.std(chamfer_similarities):.4f}")

if __name__ == "__main__":
    main() 