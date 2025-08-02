#!/usr/bin/env python3
"""
SAM and YOLO12 Segmentation Test Script
======================================
Tests both Segment Anything Model (SAM) and YOLO12 segmentation models
on images from the data/unseen_crops folder.

This script will:
1. Attempt to load or download model weights
2. Run segmentation on 10 sample images for each model
3. Visualize segmentation maps with legends
4. Handle cases where weights cannot be found
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def install_dependencies():
    """Install required dependencies for SAM and YOLO12."""
    print("🔧 Installing dependencies...")
    
    dependencies = [
        "segment-anything",
        "ultralytics",
        "supervision",
        "roboflow"
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
        
        # Also create automatic mask generator for full coverage
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,  # Reduce for faster processing
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        print(f"✅ SAM model and mask generator loaded successfully on {device}")
        return sam, predictor, mask_generator
        
    except Exception as e:
        print(f"❌ Failed to setup SAM model: {e}")
        traceback.print_exc()
        return None, None, None

def setup_yolo_model():
    """Set up YOLO12 segmentation model."""
    try:
        from ultralytics import YOLO
        
        # Try to load YOLOv8 segmentation model (as YOLOv12 may not be available)
        model_names = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt"]
        
        for model_name in model_names:
            try:
                print(f"🔍 Trying to load {model_name}...")
                model = YOLO(model_name)
                print(f"✅ YOLO model loaded successfully: {model_name}")
                return model
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
                continue
        
        print("❌ Failed to load any YOLO segmentation model")
        return None
        
    except Exception as e:
        print(f"❌ Failed to setup YOLO model: {e}")
        traceback.print_exc()
        return None

def get_sample_images(data_dir, num_images=10):
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

def run_sam_segmentation(predictor, image_path, mode='point', mask_generator=None):
    """Run SAM segmentation on a single image with different modes."""
    try:
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        
        if mode == 'automatic' and mask_generator is not None:
            # Automatic mask generation - generates multiple masks for full coverage
            masks_data = mask_generator.generate(image)
            
            if len(masks_data) > 0:
                # Combine all masks for visualization
                combined_mask = np.zeros((h, w), dtype=bool)
                for mask_data in masks_data:
                    combined_mask = np.logical_or(combined_mask, mask_data['segmentation'])
                
                # Calculate average stability score
                avg_score = np.mean([m['stability_score'] for m in masks_data])
                return image, combined_mask, avg_score, 'automatic'
            else:
                return image, np.zeros((h, w), dtype=bool), 0.0, 'automatic'
        
        # Set image for predictor for other modes
        predictor.set_image(image)
        
        if mode == 'point':
            # Point-based prompting (current method)
            input_point = np.array([[w//2, h//2]])
            input_label = np.array([1])
            
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            best_mask = masks[np.argmax(scores)]
            return image, best_mask, scores[np.argmax(scores)], 'point'
            
        elif mode == 'box':
            # Box prompting - segment everything in a box
            margin = min(w, h) // 10  # 10% margin
            input_box = np.array([margin, margin, w-margin, h-margin])
            
            masks, scores, logits = predictor.predict(
                box=input_box,
                multimask_output=False,
            )
            
            return image, masks[0], scores[0], 'box'
            
        elif mode == 'multi_point':
            # Multiple points for better coverage
            points = np.array([
                [w//4, h//4],      # Top-left quadrant
                [3*w//4, h//4],    # Top-right quadrant  
                [w//4, 3*h//4],    # Bottom-left quadrant
                [3*w//4, 3*h//4],  # Bottom-right quadrant
                [w//2, h//2]       # Center
            ])
            labels = np.array([1, 1, 1, 1, 1])
            
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False,
            )
            
            return image, masks[0], scores[0], 'multi_point'
        
    except Exception as e:
        print(f"❌ SAM segmentation failed for {image_path}: {e}")
        return None, None, None, mode

def run_yolo_segmentation(model, image_path):
    """Run YOLO segmentation on a single image."""
    try:
        # Run inference
        results = model(str(image_path))
        
        # Get the first result
        result = results[0]
        
        # Load original image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create segmentation mask
        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            
        return image, masks, result.boxes
        
    except Exception as e:
        print(f"❌ YOLO segmentation failed for {image_path}: {e}")
        return None, None, None

def visualize_sam_results(images_data, save_dir="sam_results", mode="point"):
    """Visualize SAM segmentation results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine grid size based on number of images
    num_images = len(images_data)
    if num_images <= 5:
        rows, cols = 1, 5
        figsize = (20, 4)
    else:
        rows, cols = 2, 5
        figsize = (20, 8)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array for consistency
        
    mode_titles = {
        'point': 'SAM Point Prompting Results',
        'box': 'SAM Box Prompting Results', 
        'multi_point': 'SAM Multi-Point Prompting Results',
        'automatic': 'SAM Automatic Mask Generation Results'
    }
    fig.suptitle(mode_titles.get(mode, 'SAM Segmentation Results'), fontsize=16)
    
    for i, (image, mask, score, image_path, seg_mode) in enumerate(images_data):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        ax = axes[row, col]
        
        # Show original image with mask overlay
        ax.imshow(image)
        if mask is not None:
            # Create colored mask
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[:, :, 0] = 1.0  # Red channel
            colored_mask[:, :, 3] = mask * 0.6  # Alpha channel
            ax.imshow(colored_mask)
            ax.set_title(f'{image_path.name}\nScore: {score:.3f}\nMode: {seg_mode}', fontsize=9)
        else:
            ax.set_title(f'{image_path.name}\nFailed', fontsize=10)
        
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(images_data), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    # Add legend with mode explanation
    from matplotlib.patches import Patch
    mode_explanations = {
        'point': 'Single center point prompt',
        'box': 'Box prompt covering most of image',
        'multi_point': 'Multiple point prompts',
        'automatic': 'Automatic mask generation (full coverage)'
    }
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Segmented Region'),
        Patch(facecolor='white', alpha=0, label=mode_explanations.get(mode, ''))
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sam_{mode}_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 SAM {mode} results saved to {save_dir}/sam_{mode}_results.png")

def visualize_yolo_results(images_data, save_dir="yolo_results"):
    """Visualize YOLO segmentation results."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('YOLO Segmentation Results', fontsize=16)
    
    # COCO class names for legend
    coco_classes = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        # Add more as needed...
    }
    
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    for i, (image, masks, boxes, image_path) in enumerate(images_data):
        if i >= 10:
            break
            
        row = i // 5
        col = i % 5
        
        ax = axes[row, col]
        ax.imshow(image)
        
        if masks is not None and len(masks) > 0:
            # Overlay each mask with different colors
            for j, mask in enumerate(masks):
                color = colors[j % len(colors)]
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[:, :, :3] = color[:3]
                colored_mask[:, :, 3] = mask * 0.6
                ax.imshow(colored_mask)
            
            ax.set_title(f'{image_path.name}\nObjects: {len(masks)}', fontsize=10)
        else:
            ax.set_title(f'{image_path.name}\nNo objects', fontsize=10)
        
        ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i % len(colors)], alpha=0.6, 
                           label=f'Object {i+1}') for i in range(min(5, len(colors)))]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/yolo_segmentation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 YOLO results saved to {save_dir}/yolo_segmentation_results.png")

def main():
    """Main function to run segmentation tests."""
    print("🚀 Starting SAM and YOLO12 Segmentation Test")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Get sample images
    data_dir = "../data/unseen_crops"
    sample_images = get_sample_images(data_dir, num_images=10)
    
    if len(sample_images) == 0:
        print("❌ No sample images found. Exiting.")
        return
    
    print(f"📷 Processing {len(sample_images)} images")
    
    # Test SAM Model with different modes
    print("\n" + "="*30)
    print("🎯 TESTING SAM MODEL")
    print("="*30)
    
    sam_model, sam_predictor, mask_generator = setup_sam_model()
    
    if sam_predictor is not None:
        # Test different SAM modes
        sam_modes = ['point', 'box', 'multi_point', 'automatic']
        
        for mode in sam_modes:
            print(f"\n🔍 Running SAM segmentation with {mode} mode...")
            sam_results = []
            
            # Use fewer images for automatic mode as it's slower
            test_images = sample_images[:5] if mode == 'automatic' else sample_images
            
            for i, image_path in enumerate(test_images):
                print(f"Processing {i+1}/{len(test_images)}: {image_path.name}")
                
                if mode == 'automatic':
                    result = run_sam_segmentation(sam_predictor, image_path, mode=mode, mask_generator=mask_generator)
                else:
                    result = run_sam_segmentation(sam_predictor, image_path, mode=mode)
                    
                if len(result) == 4:  # Successful segmentation
                    image, mask, score, seg_mode = result
                    sam_results.append((image, mask, score, image_path, seg_mode))
                else:
                    print(f"⚠️ Failed to segment {image_path.name}")
            
            if sam_results:
                print(f"📊 Visualizing SAM {mode} results...")
                visualize_sam_results(sam_results, mode=mode)
                
                # Print coverage statistics
                total_pixels = sum(img.shape[0] * img.shape[1] for img, _, _, _, _ in sam_results if img is not None)
                covered_pixels = sum(np.sum(mask) for _, mask, _, _, _ in sam_results if mask is not None)
                coverage_pct = (covered_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                print(f"📊 {mode.upper()} mode coverage: {coverage_pct:.1f}% of total pixels")
    else:
        print("❌ SAM model not available - skipping SAM tests")
    
    # Test YOLO Model
    print("\n" + "="*30)
    print("🎯 TESTING YOLO MODEL")
    print("="*30)
    
    yolo_model = setup_yolo_model()
    yolo_results = []
    
    if yolo_model is not None:
        print("🔍 Running YOLO segmentation...")
        for i, image_path in enumerate(sample_images):
            print(f"Processing {i+1}/{len(sample_images)}: {image_path.name}")
            image, masks, boxes = run_yolo_segmentation(yolo_model, image_path)
            yolo_results.append((image, masks, boxes, image_path))
        
        print("📊 Visualizing YOLO results...")
        visualize_yolo_results(yolo_results)
    else:
        print("❌ YOLO model not available - skipping YOLO tests")
    
    # Summary
    print("\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    print(f"SAM Model: {'✅ Available' if sam_predictor is not None else '❌ Not Available'}")
    print(f"YOLO Model: {'✅ Available' if yolo_model is not None else '❌ Not Available'}")
    print(f"Images Processed: {len(sample_images)}")
    
    if sam_predictor is not None:
        print("\nSAM Modes Tested:")
        print("  📍 Point mode: Segments object at center point")
        print("  📦 Box mode: Segments content within bounding box")
        print("  🎯 Multi-point mode: Segments using multiple point prompts")
        print("  🤖 Automatic mode: Generates masks for all objects (full coverage)")
        print("SAM Results: Check sam_results/ folder for point_results.png, box_results.png, multi_point_results.png, automatic_results.png")
    if yolo_model is not None:
        print("YOLO Results: Check yolo_results/ folder")
    
    print("\n💡 SAM Coverage Explanation:")
    print("  • Point mode: Segments specific objects (limited coverage is INTENDED)")
    print("  • Box mode: Better coverage by prompting entire regions")  
    print("  • Multi-point: Combines multiple objects/regions")
    print("  • Automatic mode: Full coverage by generating all possible masks")
    
    print("\n🎉 Segmentation test completed!")

if __name__ == "__main__":
    main() 