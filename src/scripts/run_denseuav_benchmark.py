#!/usr/bin/env python3
"""
DenseUAV Complete Benchmark Runner

Runs the complete DenseUAV visual place recognition benchmark:
1. Generates VLAD vocabulary using DenseUAV images
2. Evaluates both concatenated VLAD and Chamfer similarity
3. Provides comprehensive comparison between the methods

This benchmark tests drone-to-satellite matching using the DenseUAV dataset.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    end_time = time.time()
    
    print(f"\nCompleted in {end_time - start_time:.1f} seconds")
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Run complete DenseUAV benchmark')
    parser.add_argument('--denseuav_dir', default='datasets/DenseUAV', 
                       help='Path to DenseUAV dataset directory')
    parser.add_argument('--output_dir', default='outputs/denseuav_benchmark', 
                       help='Output directory for all results')
    parser.add_argument('--n_clusters', type=int, default=64, 
                       help='Number of clusters for VLAD vocabulary')
    parser.add_argument('--distance_threshold', type=float, default=25.0,
                       help='Distance threshold for positive matches (meters)')
    parser.add_argument('--max_queries', type=int, default=200,
                       help='Maximum number of queries to evaluate')
    parser.add_argument('--max_gallery', type=int, default=500,
                       help='Maximum number of gallery images')
    parser.add_argument('--max_images_vocab', type=int, default=2000,
                       help='Maximum number of images to use for vocabulary generation')
    parser.add_argument('--skip_vocab', action='store_true',
                       help='Skip vocabulary generation if already exists')
    
    args = parser.parse_args()
    
    # Setup paths
    denseuav_dir = Path(args.denseuav_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_dir = output_dir / 'vocabulary'
    eval_dir = output_dir / 'evaluation'
    
    vocab_file = vocab_dir / f'denseuav_vlad_vocabulary_{args.n_clusters}clusters.pkl'
    
    print("DenseUAV Visual Place Recognition Benchmark")
    print("=" * 50)
    print(f"Dataset: {denseuav_dir}")
    print(f"Output: {output_dir}")
    print(f"Vocabulary clusters: {args.n_clusters}")
    print(f"Distance threshold: {args.distance_threshold}m")
    print(f"Max queries: {args.max_queries}")
    print(f"Max gallery: {args.max_gallery}")
    
    # Check if DenseUAV dataset exists
    if not denseuav_dir.exists():
        print(f"\nERROR: DenseUAV dataset not found at {denseuav_dir}")
        print("Please ensure the DenseUAV dataset is available.")
        sys.exit(1)
    
    # Check dataset structure
    required_dirs = ['test/query_drone', 'test/gallery_satellite']
    required_files = ['Dense_GPS_test.txt']
    
    for req_dir in required_dirs:
        if not (denseuav_dir / req_dir).exists():
            print(f"ERROR: Required directory {req_dir} not found in {denseuav_dir}")
            sys.exit(1)
    
    for req_file in required_files:
        if not (denseuav_dir / req_file).exists():
            print(f"ERROR: Required file {req_file} not found in {denseuav_dir}")
            sys.exit(1)
    
    print(f"\n✓ DenseUAV dataset structure verified")
    
    # Step 1: Generate vocabulary (if needed)
    if args.skip_vocab and vocab_file.exists():
        print(f"\n✓ Skipping vocabulary generation (file exists: {vocab_file})")
    else:
        vocab_cmd = [
            'python', 'src/scripts/denseuav_vlad_vocabulary_generation.py',
            '--denseuav_dir', str(denseuav_dir),
            '--output_dir', str(vocab_dir),
            '--n_clusters', str(args.n_clusters),
            '--max_features', '200000',
            '--patches_per_image', '10',
            '--max_images', str(args.max_images_vocab)
        ]
        
        run_command(vocab_cmd, "Vocabulary Generation")
    
    # Verify vocabulary was created
    if not vocab_file.exists():
        print(f"ERROR: Vocabulary file not found: {vocab_file}")
        sys.exit(1)
    
    # Step 2: Run evaluation
    eval_cmd = [
        'python', 'src/scripts/denseuav_vlad_evaluation.py',
        '--denseuav_dir', str(denseuav_dir),
        '--vocab_path', str(vocab_file),
        '--output_dir', str(eval_dir),
        '--distance_threshold', str(args.distance_threshold),
        '--max_queries', str(args.max_queries),
        '--max_gallery', str(args.max_gallery)
    ]
    
    run_command(eval_cmd, "DenseUAV Evaluation")
    
    # Step 3: Print final summary
    print(f"\n{'='*60}")
    print("DENSEUAV BENCHMARK COMPLETED")
    print(f"{'='*60}")
    
    # Try to read and display results
    try:
        import pickle
        results_file = eval_dir / 'denseuav_evaluation_results.pkl'
        
        if results_file.exists():
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            
            print(f"\nDataset Statistics:")
            print(f"  - Queries: {results['dataset_info']['total_queries']}")
            print(f"  - Gallery: {results['dataset_info']['total_gallery']}")
            print(f"  - Distance threshold: {results['dataset_info']['distance_threshold']}m")
            print(f"  - Vocabulary clusters: {results['dataset_info']['vocabulary_clusters']}")
            
            print(f"\n{'='*40}")
            print("CONCATENATED VLAD RESULTS")
            print(f"{'='*40}")
            for k in [1, 5, 10, 20]:
                if k in results['concatenated_vlad']:
                    recall = results['concatenated_vlad'][k]
                    print(f"  Recall@{k:2d}: {recall:5.1f}%")
            
            print(f"\n{'='*40}")
            print("CHAMFER SIMILARITY RESULTS")
            print(f"{'='*40}")
            for k in [1, 5, 10, 20]:
                if k in results['chamfer_similarity']:
                    recall = results['chamfer_similarity'][k]
                    print(f"  Recall@{k:2d}: {recall:5.1f}%")
            
            # Comparison analysis
            vlad_r1 = results['concatenated_vlad'].get(1, 0)
            chamfer_r1 = results['chamfer_similarity'].get(1, 0)
            vlad_r5 = results['concatenated_vlad'].get(5, 0)
            chamfer_r5 = results['chamfer_similarity'].get(5, 0)
            
            print(f"\n{'='*40}")
            print("METHOD COMPARISON")
            print(f"{'='*40}")
            
            # R@1 comparison
            if vlad_r1 > chamfer_r1:
                print(f"  ✓ Concatenated VLAD better at R@1: {vlad_r1:.1f}% vs {chamfer_r1:.1f}%")
            elif chamfer_r1 > vlad_r1:
                print(f"  ✓ Chamfer Similarity better at R@1: {chamfer_r1:.1f}% vs {vlad_r1:.1f}%")
            else:
                print(f"  = Methods tied at R@1: {vlad_r1:.1f}%")
            
            # R@5 comparison
            if vlad_r5 > chamfer_r5:
                print(f"  ✓ Concatenated VLAD better at R@5: {vlad_r5:.1f}% vs {chamfer_r5:.1f}%")
            elif chamfer_r5 > vlad_r5:
                print(f"  ✓ Chamfer Similarity better at R@5: {chamfer_r5:.1f}% vs {vlad_r5:.1f}%")
            else:
                print(f"  = Methods tied at R@5: {vlad_r5:.1f}%")
            
            # Overall winner
            vlad_total = vlad_r1 + vlad_r5
            chamfer_total = chamfer_r1 + chamfer_r5
            
            print(f"\nOverall Performance (R@1 + R@5):")
            print(f"  - Concatenated VLAD: {vlad_total:.1f}%")
            print(f"  - Chamfer Similarity: {chamfer_total:.1f}%")
            
            if vlad_total > chamfer_total:
                print(f"  🏆 Winner: Concatenated VLAD (+{vlad_total - chamfer_total:.1f}%)")
            elif chamfer_total > vlad_total:
                print(f"  🏆 Winner: Chamfer Similarity (+{chamfer_total - vlad_total:.1f}%)")
            else:
                print(f"  🤝 Tie between methods")
        
    except Exception as e:
        print(f"Note: Could not read results file for summary: {e}")
    
    print(f"\nOutput files:")
    print(f"  - Vocabulary: {vocab_file}")
    print(f"  - Results: {eval_dir / 'denseuav_evaluation_results.pkl'}")
    print(f"  - Summary: {eval_dir / 'denseuav_evaluation_summary.txt'}")
    
    print(f"\nBenchmark Context:")
    print(f"  - This benchmark uses the DenseUAV dataset for drone-to-satellite matching")
    print(f"  - Compares concatenated VLAD vs Chamfer similarity for cross-domain retrieval")
    print(f"  - Uses GPS ground truth for evaluation with {args.distance_threshold}m threshold")
    print(f"  - Vocabulary generated from diverse aerial imagery from DenseUAV dataset")
    print(f"  - Results show relative performance of aggregation methods")

if __name__ == '__main__':
    main() 