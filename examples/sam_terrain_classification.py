#!/usr/bin/env python3
"""
SAM Terrain Classification and Binary Separation Script
======================================================
Uses SAM to segment terrain into regions, creates binary separation maps,
and attempts to classify different terrain types.

Features:
- Runs SAM in point and automatic modes
- Creates binary masks with border erosion
- Generates grayscale separation lines
- Attempts terrain classification based on visual features
- Comprehensive visualization of results
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import subprocess
from pathlib import Path
import traceback
from scipy import ndimage
from sklearn.cluster import KMeans
import matplotlib.patches as patches

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def install_dependencies():
    """Install required dependencies."""
    print("🔧 Installing dependencies...")
    
    dependencies = [
        "segment-anything",
        "scikit-learn",
        "scipy"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {dep} - continuing anyway")

def download_sam_model():
    """Download SAM model checkpoint."""
    import urllib.request
    
    sam_checkpoint_path = "sam_vit_h_4b8939.pth"
    
    if os.path.exists(sam_checkpoint_path):
        print(f"✅ SAM checkpoint already exists: {sam_checkpoint_path}")
        return sam_checkpoint_path
    
    try:
        print("📥 Downloading SAM ViT-H checkpoint...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, sam_checkpoint_path)
        print(f"✅ Downloaded SAM checkpoint: {sam_checkpoint_path}")
        return sam_checkpoint_path
    except Exception as e:
        print(f"❌ Failed to download SAM checkpoint: {e}")
        return None

def setup_sam_model():
    """Set up SAM model."""
    try:
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        
        # Download model if needed
        checkpoint_path = download_sam_model()
        if checkpoint_path is None:
            return None, None, None
        
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        
        predictor = SamPredictor(sam)
        
        # Create automatic mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=500,  # Larger minimum area for terrain segments
        )
        
        print(f"✅ SAM model loaded successfully on {device}")
        return sam, predictor, mask_generator
        
    except Exception as e:
        print(f"❌ Failed to setup SAM model: {e}")
        traceback.print_exc()
        return None, None, None

def get_sample_images(data_dir, num_images=20):
    """Get sample images from the unseen crops directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return []
    
    # Get all jpg files
    image_files = list(data_path.glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {data_dir}")
        return []
    
    # Select sample images
    sample_files = image_files[:num_images]
    print(f"📁 Found {len(image_files)} images, using {len(sample_files)} samples")
    
    return sample_files

def create_binary_mask_with_border(mask, border_width=10):
    """Create binary mask with eroded borders."""
    if mask is None:
        return None
    
    # Convert to binary
    binary_mask = mask.astype(np.uint8)
    
    # Create erosion kernel
    kernel = np.ones((border_width, border_width), np.uint8)
    
    # Erode the mask to create border
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    
    return eroded_mask

def extract_region_features(image, mask):
    """Extract basic features from a masked region for classification."""
    if mask is None or np.sum(mask) == 0:
        return {}
    
    # Get pixels in the masked region
    masked_pixels = image[mask > 0]
    
    if len(masked_pixels) == 0:
        return {}
    
    features = {
        'mean_color': np.mean(masked_pixels, axis=0),
        'std_color': np.std(masked_pixels, axis=0),
        'area': np.sum(mask),
        'brightness': np.mean(masked_pixels),
        'color_variance': np.var(masked_pixels, axis=0),
    }
    
    return features

def classify_terrain_region(features):
    """Attempt to classify terrain based on extracted features."""
    if not features:
        return "Unknown"
    
    mean_color = features.get('mean_color', np.array([0, 0, 0]))
    brightness = features.get('brightness', 0)
    area = features.get('area', 0)
    
    # Simple heuristic classification based on color and brightness
    if brightness < 80:
        return "Dark/Shadow"
    elif mean_color[1] > mean_color[0] and mean_color[1] > mean_color[2]:  # Green dominant
        if brightness > 120:
            return "Vegetation/Crops"
        else:
            return "Forest/Dense Vegetation"
    elif mean_color[0] > 150 and mean_color[1] > 150 and mean_color[2] > 150:  # Light colors
        return "Road/Path"
    elif mean_color[2] < 100 and mean_color[1] < 100:  # Low blue and green
        return "Soil/Dirt"
    elif abs(mean_color[0] - mean_color[1]) < 20 and abs(mean_color[1] - mean_color[2]) < 20:
        return "Concrete/Urban"
    else:
        return "Mixed Terrain"

def run_sam_point_mode(predictor, image):
    """Run SAM with point prompting."""
    try:
        predictor.set_image(image)
        
        h, w = image.shape[:2]
        
        # Use multiple strategic points for better terrain segmentation
        points = np.array([
            [w//2, h//2],      # Center
            [w//4, h//4],      # Top-left
            [3*w//4, h//4],    # Top-right
            [w//4, 3*h//4],    # Bottom-left
            [3*w//4, 3*h//4],  # Bottom-right
        ])
        labels = np.array([1, 1, 1, 1, 1])
        
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        # Return all masks for analysis
        return masks, scores
        
    except Exception as e:
        print(f"❌ SAM point mode failed: {e}")
        return None, None

def run_sam_automatic_mode(mask_generator, image):
    """Run SAM with automatic mask generation."""
    try:
        masks_data = mask_generator.generate(image)
        
        if len(masks_data) == 0:
            return None
        
        # Sort by area (largest first)
        masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)
        
        return masks_data
        
    except Exception as e:
        print(f"❌ SAM automatic mode failed: {e}")
        return None

def create_separation_visualization(image, masks_data, mode="automatic"):
    """Create visualization showing original, binary masks, and classified regions."""
    h, w = image.shape[:2]
    
    # Create combined binary mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    region_labels = np.zeros((h, w), dtype=np.int32)
    
    classifications = []
    
    if mode == "automatic" and masks_data:
        # Process automatic masks
        for i, mask_data in enumerate(masks_data[:5]):  # Limit to top 5 masks
            mask = mask_data['segmentation']
            
            # Create binary mask with border
            binary_mask = create_binary_mask_with_border(mask, border_width=10)
            if binary_mask is not None:
                combined_mask = np.maximum(combined_mask, binary_mask)
                region_labels[binary_mask > 0] = i + 1
                
                # Extract features and classify
                features = extract_region_features(image, mask)
                classification = classify_terrain_region(features)
                classifications.append({
                    'region_id': i + 1,
                    'classification': classification,
                    'area': mask_data['area'],
                    'stability_score': mask_data['stability_score']
                })
    
    elif mode == "point" and masks_data:
        # Process point-based masks
        masks, scores = masks_data
        for i, mask in enumerate(masks):
            binary_mask = create_binary_mask_with_border(mask, border_width=10)
            if binary_mask is not None:
                combined_mask = np.maximum(combined_mask, binary_mask)
                region_labels[binary_mask > 0] = i + 1
                
                # Extract features and classify
                features = extract_region_features(image, mask)
                classification = classify_terrain_region(features)
                classifications.append({
                    'region_id': i + 1,
                    'classification': classification,
                    'area': np.sum(mask),
                    'score': scores[i] if i < len(scores) else 0.0
                })
    
    # Create separation lines (edges of masks)
    edges = cv2.Canny(combined_mask, 50, 150)
    
    return combined_mask, region_labels, edges, classifications

def visualize_terrain_analysis(image_path, image, point_results, auto_results, save_dir):
    """Create comprehensive visualization of terrain analysis."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'SAM Terrain Analysis: {image_path.name}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Point mode results
    if point_results:
        point_mask, point_regions, point_edges, point_classes = point_results
        
        # Binary mask
        axes[0, 1].imshow(point_mask, cmap='gray')
        axes[0, 1].set_title('Point Mode: Binary Mask')
        axes[0, 1].axis('off')
        
        # Classified regions overlay
        axes[0, 2].imshow(image)
        colored_regions = np.zeros_like(image)
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        for cls in point_classes:
            region_id = cls['region_id']
            color = colors[region_id % len(colors)][:3]
            mask = point_regions == region_id
            colored_regions[mask] = (np.array(color) * 255).astype(np.uint8)
        
        axes[0, 2].imshow(colored_regions, alpha=0.6)
        axes[0, 2].set_title('Point Mode: Classified Regions')
        axes[0, 2].axis('off')
        
        # Add text annotations for classifications
        y_offset = 0.02
        for cls in point_classes:
            color = colors[cls['region_id'] % len(colors)]
            axes[0, 2].text(0.02, 0.98 - y_offset, 
                          f"R{cls['region_id']}: {cls['classification']}", 
                          transform=axes[0, 2].transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                          fontsize=8)
            y_offset += 0.08
    else:
        axes[0, 1].text(0.5, 0.5, 'Point Mode Failed', ha='center', va='center')
        axes[0, 1].axis('off')
        axes[0, 2].text(0.5, 0.5, 'Point Mode Failed', ha='center', va='center')
        axes[0, 2].axis('off')
    
    # Automatic mode results
    if auto_results:
        auto_mask, auto_regions, auto_edges, auto_classes = auto_results
        
        # Binary mask
        axes[1, 0].imshow(auto_mask, cmap='gray')
        axes[1, 0].set_title('Automatic Mode: Binary Mask')
        axes[1, 0].axis('off')
        
        # Separation lines
        axes[1, 1].imshow(image, alpha=0.7)
        axes[1, 1].imshow(auto_edges, cmap='Reds', alpha=0.8)
        axes[1, 1].set_title('Automatic Mode: Separation Lines')
        axes[1, 1].axis('off')
        
        # Classified regions overlay
        axes[1, 2].imshow(image)
        colored_regions = np.zeros_like(image)
        
        for cls in auto_classes:
            region_id = cls['region_id']
            color = colors[region_id % len(colors)][:3]
            mask = auto_regions == region_id
            colored_regions[mask] = (np.array(color) * 255).astype(np.uint8)
        
        axes[1, 2].imshow(colored_regions, alpha=0.6)
        axes[1, 2].set_title('Automatic Mode: Classified Regions')
        axes[1, 2].axis('off')
        
        # Add text annotations for classifications
        y_offset = 0.02
        for cls in auto_classes:
            color = colors[cls['region_id'] % len(colors)]
            axes[1, 2].text(0.02, 0.98 - y_offset, 
                          f"R{cls['region_id']}: {cls['classification']}", 
                          transform=axes[1, 2].transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                          fontsize=8)
            y_offset += 0.08
    else:
        axes[1, 0].text(0.5, 0.5, 'Automatic Mode Failed', ha='center', va='center')
        axes[1, 0].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Automatic Mode Failed', ha='center', va='center')
        axes[1, 1].axis('off')
        axes[1, 2].text(0.5, 0.5, 'Automatic Mode Failed', ha='center', va='center')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'terrain_analysis_{image_path.stem}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def process_image_terrain_analysis(predictor, mask_generator, image_path, save_dir):
    """Process a single image for terrain analysis."""
    print(f"🔍 Processing: {image_path.name}")
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run SAM in point mode
        print("  📍 Running point mode...")
        point_data = run_sam_point_mode(predictor, image)
        point_results = None
        if point_data[0] is not None:
            point_results = create_separation_visualization(image, point_data, mode="point")
        
        # Run SAM in automatic mode
        print("  🤖 Running automatic mode...")
        auto_data = run_sam_automatic_mode(mask_generator, image)
        auto_results = None
        if auto_data is not None:
            auto_results = create_separation_visualization(image, auto_data, mode="automatic")
        
        # Create visualization
        print("  📊 Creating visualization...")
        save_path = visualize_terrain_analysis(image_path, image, point_results, auto_results, save_dir)
        
        return {
            'image_path': image_path,
            'point_results': point_results,
            'auto_results': auto_results,
            'save_path': save_path,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Failed to process {image_path.name}: {e}")
        traceback.print_exc()
        return {
            'image_path': image_path,
            'success': False,
            'error': str(e)
        }

def main():
    """Main function to run terrain classification analysis."""
    print("🚀 Starting SAM Terrain Classification Analysis")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Setup SAM model
    print("\n🔧 Setting up SAM model...")
    sam_model, sam_predictor, mask_generator = setup_sam_model()
    
    if sam_predictor is None:
        print("❌ Failed to setup SAM model. Exiting.")
        return
    
    # Get sample images
    data_dir = "../data/unseen_crops"
    sample_images = get_sample_images(data_dir, num_images=20)
    
    if len(sample_images) == 0:
        print("❌ No sample images found. Exiting.")
        return
    
    print(f"📷 Processing {len(sample_images)} images for terrain analysis")
    
    # Create save directory
    save_dir = "terrain_analysis_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Process each image
    results = []
    for i, image_path in enumerate(sample_images):
        print(f"\n📊 Processing image {i+1}/{len(sample_images)}")
        result = process_image_terrain_analysis(sam_predictor, mask_generator, image_path, save_dir)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("📋 TERRAIN ANALYSIS SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successfully processed: {len(successful)}/{len(results)} images")
    print(f"❌ Failed: {len(failed)} images")
    
    if successful:
        print(f"📁 Results saved to: {save_dir}/")
        print("\n🎯 Analysis includes:")
        print("  • Original images")
        print("  • Binary masks with 10-pixel border erosion")
        print("  • Separation lines showing terrain boundaries")
        print("  • Automatic terrain classification attempts")
        print("  • Both point and automatic SAM modes tested")
    
    if failed:
        print("\n❌ Failed images:")
        for f in failed:
            print(f"  - {f['image_path'].name}: {f['error']}")
    
    print("\n💡 Notes:")
    print("  • Point mode uses 5 strategic points for segmentation")
    print("  • Automatic mode generates comprehensive masks")
    print("  • Classifications are heuristic-based (color/brightness)")
    print("  • Binary masks have 10-pixel erosion for clear boundaries")
    
    print("\n🎉 Terrain classification analysis completed!")

if __name__ == "__main__":
    main() 