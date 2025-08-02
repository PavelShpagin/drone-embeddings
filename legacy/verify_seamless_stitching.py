#!/usr/bin/env python3
"""
Verify Seamless Stitching Quality
=================================

Analyzes satellite images to detect visible stitching artifacts and seams.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_stitching_quality(image_path, grid_size=(4, 4)):
    """
    Analyze image for stitching artifacts by checking tile boundaries.
    
    Args:
        image_path: Path to the image file
        grid_size: Expected tile grid size (tiles_x, tiles_y)
    
    Returns:
        dict: Analysis results with stitching quality metrics
    """
    print(f"🔍 Analyzing stitching quality: {image_path.name}")
    
    # Load image
    image = Image.open(image_path)
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    print(f"   📏 Image dimensions: {width}×{height}")
    
    tiles_x, tiles_y = grid_size
    tile_width = width // tiles_x
    tile_height = height // tiles_y
    
    print(f"   🧩 Expected tile size: {tile_width}×{tile_height}")
    
    # Analyze horizontal seams (between vertical tiles)
    horizontal_seam_scores = []
    for row in range(1, tiles_y):
        seam_y = row * tile_height
        if seam_y < height - 1:
            # Extract horizontal line and adjacent lines
            line_above = img_array[seam_y - 1, :, :]
            line_at = img_array[seam_y, :, :]
            line_below = img_array[seam_y + 1, :, :]
            
            # Calculate variance across the seam
            seam_variance = np.var([line_above, line_at, line_below], axis=0)
            avg_seam_variance = np.mean(seam_variance)
            horizontal_seam_scores.append(avg_seam_variance)
            
            print(f"   📐 Horizontal seam {row} (y={seam_y}): variance = {avg_seam_variance:.2f}")
    
    # Analyze vertical seams (between horizontal tiles)
    vertical_seam_scores = []
    for col in range(1, tiles_x):
        seam_x = col * tile_width
        if seam_x < width - 1:
            # Extract vertical line and adjacent lines
            line_left = img_array[:, seam_x - 1, :]
            line_at = img_array[:, seam_x, :]
            line_right = img_array[:, seam_x + 1, :]
            
            # Calculate variance across the seam
            seam_variance = np.var([line_left, line_at, line_right], axis=0)
            avg_seam_variance = np.mean(seam_variance)
            vertical_seam_scores.append(avg_seam_variance)
            
            print(f"   📐 Vertical seam {col} (x={seam_x}): variance = {avg_seam_variance:.2f}")
    
    # Calculate overall quality metrics
    all_seam_scores = horizontal_seam_scores + vertical_seam_scores
    avg_seam_variance = np.mean(all_seam_scores) if all_seam_scores else 0
    max_seam_variance = np.max(all_seam_scores) if all_seam_scores else 0
    
    # Calculate baseline variance for comparison
    # Sample random non-seam areas
    baseline_variances = []
    for _ in range(20):
        rand_y = np.random.randint(10, height - 10)
        rand_x = np.random.randint(10, width - 10)
        # Avoid seam areas
        if not any(abs(rand_x - col * tile_width) < 5 for col in range(1, tiles_x)) and \
           not any(abs(rand_y - row * tile_height) < 5 for row in range(1, tiles_y)):
            patch = img_array[rand_y-1:rand_y+2, rand_x-1:rand_x+2, :]
            baseline_variances.append(np.var(patch))
    
    avg_baseline_variance = np.mean(baseline_variances) if baseline_variances else 0
    
    # Quality assessment
    seam_ratio = avg_seam_variance / avg_baseline_variance if avg_baseline_variance > 0 else float('inf')
    
    print(f"\n📊 Stitching Quality Analysis:")
    print(f"   Average seam variance: {avg_seam_variance:.2f}")
    print(f"   Maximum seam variance: {max_seam_variance:.2f}")
    print(f"   Average baseline variance: {avg_baseline_variance:.2f}")
    print(f"   Seam-to-baseline ratio: {seam_ratio:.2f}")
    
    # Quality interpretation
    if seam_ratio < 1.5:
        quality = "EXCELLENT"
        emoji = "🟢"
        description = "No visible seams detected"
    elif seam_ratio < 2.5:
        quality = "GOOD"
        emoji = "🟡"
        description = "Minor seam artifacts"
    elif seam_ratio < 4.0:
        quality = "FAIR"
        emoji = "🟠"
        description = "Noticeable seam artifacts"
    else:
        quality = "POOR"
        emoji = "🔴"
        description = "Significant stitching lines visible"
    
    print(f"   {emoji} Overall quality: {quality}")
    print(f"   📝 Description: {description}")
    
    return {
        'avg_seam_variance': avg_seam_variance,
        'max_seam_variance': max_seam_variance,
        'avg_baseline_variance': avg_baseline_variance,
        'seam_ratio': seam_ratio,
        'quality': quality,
        'description': description,
        'horizontal_seams': horizontal_seam_scores,
        'vertical_seams': vertical_seam_scores
    }

def compare_versions():
    """Compare stitching quality across different versions."""
    print("🔍 Comparing Stitching Quality Across Versions")
    print("=" * 60)
    
    gee_dir = Path("../data/gee_api")
    
    # Find all versions
    versions = [
        ("Original (Spring v1)", "50.4162, 30.8906.spring.jpg"),
        ("Previous (Summer v1)", "50.4162, 30.8906.summer.jpg"),
        ("Advanced v1 (Autumn)", "50.4162, 30.8906.autumn.jpg")
    ]
    
    # Check if there's a newer spring version (iteration 2)
    spring_v2_path = gee_dir / "50.4162, 30.8906.spring.jpg"
    if spring_v2_path.exists():
        # Get file modification times to determine which is newer
        autumn_v1_path = gee_dir / "50.4162, 30.8906.autumn.jpg"
        if autumn_v1_path.exists():
            spring_time = spring_v2_path.stat().st_mtime
            autumn_time = autumn_v1_path.stat().st_mtime
            
            if spring_time > autumn_time:
                # This is the newer iteration 2 version
                versions = [
                    ("Previous (Summer v1)", "50.4162, 30.8906.summer.jpg"),
                    ("Advanced v1 (Autumn)", "50.4162, 30.8906.autumn.jpg"),
                    ("Perfect v2 (Spring)", "50.4162, 30.8906.spring.jpg")
                ]
    
    results = {}
    
    for version_name, filename in versions:
        image_path = gee_dir / filename
        if image_path.exists():
            print(f"\n🖼️  Analyzing: {version_name}")
            results[version_name] = analyze_stitching_quality(image_path)
        else:
            print(f"\n❌ Missing: {version_name} ({filename})")
    
    # Summary comparison
    print(f"\n🏆 QUALITY COMPARISON SUMMARY")
    print("=" * 40)
    
    for version_name, result in results.items():
        emoji = "🟢" if result['quality'] == "EXCELLENT" else \
                "🟡" if result['quality'] == "GOOD" else \
                "🟠" if result['quality'] == "FAIR" else "🔴"
        print(f"{emoji} {version_name:20} | {result['quality']:10} | Ratio: {result['seam_ratio']:.2f}")
    
    # Determine if iteration needed
    latest_result = list(results.values())[-1] if results else None
    if latest_result:
        needs_iteration = latest_result['seam_ratio'] > 1.5
        print(f"\n{'🔄' if needs_iteration else '✅'} Iteration needed: {needs_iteration}")
        return needs_iteration, latest_result
    
    return True, None

def main():
    """Main verification function."""
    needs_iteration, latest_result = compare_versions()
    
    if not needs_iteration and latest_result:
        print(f"\n🎉 SUCCESS! Seamless stitching achieved!")
        print(f"✅ Quality: {latest_result['quality']}")
        print(f"✅ Description: {latest_result['description']}")
    else:
        print(f"\n🔄 Iteration needed - stitching quality can be improved")
    
    return needs_iteration

if __name__ == "__main__":
    main() 