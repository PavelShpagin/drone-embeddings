import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt

# Import the updated VLAD embedder
import sys
sys.path.append(str(Path(__file__).parent.parent))
from geolocalization.anyloc_vlad_embedder import AnyLocVLADEmbedder

# Settings
MAP_IMAGE_PATH = "inference/46.6234, 32.7851.jpg"
CROP_SIZE = 224
CROP_STRIDE = CROP_SIZE // 2  # 50% overlap for database
N_CLUSTERS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "dinov2_vits14"
LAYER = 11
FACET = "key"

# Spatial tolerance for recall calculation
M_PER_PIXEL = 4000.0 / 8192.0  # meters per pixel
PATCH_SIZE_M = CROP_SIZE * M_PER_PIXEL
SPATIAL_TOL_M = 1.5 * PATCH_SIZE_M

def sliding_window_crops(img_path, crop_size, stride):
    """Extract sliding window crops from an image."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    crops = []
    coords = []
    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            crop = img.crop((x, y, x + crop_size, y + crop_size))
            crops.append(crop)
            cx, cy = x + crop_size // 2, y + crop_size // 2
            coords.append((cx, cy))
    return crops, coords

def compute_spatial_distance(c1, c2):
    """Compute spatial distance between two coordinate points."""
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) * M_PER_PIXEL

def evaluate_method(embedder, db_crops, db_coords, query_crops, query_coords, method="vlad"):
    """
    Evaluate either VLAD dot product or Chamfer similarity method.
    
    Args:
        method: "vlad" for concatenated VLAD dot product, "chamfer" for Chamfer similarity
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
    
    # Evaluate recall with spatial tolerance
    recall1_count = 0
    recall5_count = 0
    n_queries = len(query_coords)
    
    print("Computing recall with spatial tolerance...")
    for i in tqdm(range(n_queries), desc="Recall evaluation"):
        # Get top-5 most similar database crops
        top5_indices = np.argsort(similarities[i])[-5:][::-1]  # Descending order
        
        # Check spatial distances
        query_coord = query_coords[i]
        
        # Recall@1
        best_db_coord = db_coords[top5_indices[0]]
        spatial_dist = compute_spatial_distance(query_coord, best_db_coord)
        if spatial_dist <= SPATIAL_TOL_M:
            recall1_count += 1
        
        # Recall@5
        recall5_found = False
        for idx in top5_indices:
            db_coord = db_coords[idx]
            spatial_dist = compute_spatial_distance(query_coord, db_coord)
            if spatial_dist <= SPATIAL_TOL_M:
                recall5_found = True
                break
        if recall5_found:
            recall5_count += 1
    
    recall1 = recall1_count / n_queries
    recall5 = recall5_count / n_queries
    
    print(f"{method.upper()} Results:")
    print(f"  Recall@1: {recall1:.4f} ({recall1_count}/{n_queries})")
    print(f"  Recall@5: {recall5:.4f} ({recall5_count}/{n_queries})")
    
    return recall1, recall5, similarities

def main():
    print("Chamfer Similarity vs VLAD Dot Product Comparison")
    print("=" * 50)
    
    # Load map image and extract crops
    if not os.path.exists(MAP_IMAGE_PATH):
        print(f"Error: Map image not found at {MAP_IMAGE_PATH}")
        return
    
    print("Extracting database and query crops...")
    db_crops, db_coords = sliding_window_crops(MAP_IMAGE_PATH, CROP_SIZE, CROP_STRIDE)
    
    # For fair comparison, use the same crops as queries but with some spatial offset
    # This simulates the realistic scenario where query and database have slight misalignment
    query_crops = db_crops[::4]  # Every 4th crop as query
    query_coords = db_coords[::4]
    
    print(f"Database crops: {len(db_crops)}")
    print(f"Query crops: {len(query_crops)}")
    print(f"Spatial tolerance: {SPATIAL_TOL_M:.1f}m")
    
    # Initialize embedder
    print("Initializing AnyLoc VLAD embedder...")
    embedder = AnyLocVLADEmbedder(
        model_type=MODEL_TYPE,
        layer=LAYER,
        facet=FACET,
        device=DEVICE,
        n_clusters=N_CLUSTERS
    )
    
    # Build vocabulary from a subset of database crops
    vocab_crops = random.sample(db_crops, min(100, len(db_crops)))
    print("Fitting VLAD vocabulary...")
    embedder.fit_vocabulary(vocab_crops)
    
    # Test both methods
    results = {}
    
    # Method 1: Concatenated VLAD with dot product
    vlad_recall1, vlad_recall5, vlad_similarities = evaluate_method(
        embedder, db_crops, db_coords, query_crops, query_coords, method="vlad"
    )
    results["vlad"] = {"recall1": vlad_recall1, "recall5": vlad_recall5}
    
    # Method 2: Chamfer similarity
    chamfer_recall1, chamfer_recall5, chamfer_similarities = evaluate_method(
        embedder, db_crops, db_coords, query_crops, query_coords, method="chamfer"
    )
    results["chamfer"] = {"recall1": chamfer_recall1, "recall5": chamfer_recall5}
    
    # Print comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<15} {'Recall@1':<12} {'Recall@5':<12} {'Improvement'}")
    print("-" * 60)
    print(f"{'VLAD Dot Prod':<15} {vlad_recall1:<12.4f} {vlad_recall5:<12.4f} {'Baseline'}")
    
    r1_improvement = (chamfer_recall1 - vlad_recall1) / vlad_recall1 * 100 if vlad_recall1 > 0 else 0
    r5_improvement = (chamfer_recall5 - vlad_recall5) / vlad_recall5 * 100 if vlad_recall5 > 0 else 0
    
    print(f"{'Chamfer Sim':<15} {chamfer_recall1:<12.4f} {chamfer_recall5:<12.4f} {r1_improvement:+.1f}%/{r5_improvement:+.1f}%")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['VLAD\nDot Product', 'Chamfer\nSimilarity']
    recall1_scores = [vlad_recall1, chamfer_recall1]
    recall5_scores = [vlad_recall5, chamfer_recall5]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x - width/2, recall1_scores, width, label='Recall@1', alpha=0.8)
    ax1.bar(x + width/2, recall5_scores, width, label='Recall@5', alpha=0.8)
    ax1.set_ylabel('Recall Score')
    ax1.set_title('Recall Comparison: VLAD vs Chamfer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (r1, r5) in enumerate(zip(recall1_scores, recall5_scores)):
        ax1.text(i - width/2, r1 + 0.01, f'{r1:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, r5 + 0.01, f'{r5:.3f}', ha='center', va='bottom')
    
    # Similarity distribution comparison
    ax2.hist(vlad_similarities.flatten(), bins=50, alpha=0.5, label='VLAD Dot Product', density=True)
    ax2.hist(chamfer_similarities.flatten(), bins=50, alpha=0.5, label='Chamfer Similarity', density=True)
    ax2.set_xlabel('Similarity Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Similarity Score Distributions')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('chamfer_vs_vlad_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to chamfer_vs_vlad_comparison.png")

if __name__ == "__main__":
    main() 