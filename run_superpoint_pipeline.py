#!/usr/bin/env python3
"""
Complete SuperPoint UAV pipeline: download weights, generate data, train, visualize.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['torch', 'cv2', 'numpy', 'tqdm', 'requests']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {missing}")
        print("Install with: pip install torch opencv-python numpy tqdm requests")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Complete SuperPoint UAV pipeline")
    parser.add_argument('--earth_imagery_dir', type=str, default='data/earth_imagery',
                       help='Directory containing earth imagery')
    parser.add_argument('--n_crops_per_location', type=int, default=500,
                       help='Number of crops to generate per location')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip downloading pretrained weights')
    parser.add_argument('--skip_data_generation', action='store_true',
                       help='Skip UAV data generation')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training')
    parser.add_argument('--only_visualize', action='store_true',
                       help='Only run visualization')
    
    args = parser.parse_args()
    
    print("🚁 SuperPoint UAV Pipeline")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Define paths
    pretrained_weights_dir = Path("pretrained_weights")
    uav_data_dir = Path("uav_data")
    trained_model_dir = Path("superpoint_uav_trained")
    
    # Step 1: Download pretrained weights
    if not args.skip_download and not args.only_visualize:
        if not run_command("python download_pretrained_superpoint.py", 
                          "Download pretrained SuperPoint weights"):
            print("⚠️ Could not download weights automatically.")
            print("Please manually download SuperPoint weights and place in pretrained_weights/")
            return 1
    
    # Find pretrained weights
    weight_files = list(pretrained_weights_dir.glob("*.pth"))
    if not weight_files:
        print("❌ No pretrained weights found in pretrained_weights/")
        print("Please run download_pretrained_superpoint.py first")
        return 1
    
    pretrained_weights = str(weight_files[0])
    print(f"Using pretrained weights: {pretrained_weights}")
    
    # Step 2: Generate UAV data
    if not args.skip_data_generation and not args.only_visualize:
        if not Path(args.earth_imagery_dir).exists():
            print(f"❌ Earth imagery directory not found: {args.earth_imagery_dir}")
            return 1
        
        cmd = f"python generate_uav_data.py --earth_imagery_dir {args.earth_imagery_dir} --output_dir {uav_data_dir} --n_crops_per_location {args.n_crops_per_location}"
        if not run_command(cmd, "Generate UAV training data"):
            return 1
    
    # Check if UAV data exists
    if not uav_data_dir.exists() or not list(uav_data_dir.glob("*.png")):
        print(f"❌ No UAV data found in {uav_data_dir}")
        if not args.only_visualize:
            print("Please run data generation first")
            return 1
    
    # Step 3: Train SuperPoint on UAV data
    if not args.skip_training and not args.only_visualize:
        cmd = f"python train_superpoint_uav.py --data_dir {uav_data_dir} --pretrained_weights {pretrained_weights} --output_dir {trained_model_dir} --epochs {args.epochs}"
        if not run_command(cmd, "Train SuperPoint on UAV data"):
            return 1
    
    # Find trained model (use pretrained if no trained model exists)
    trained_weights = None
    if trained_model_dir.exists():
        trained_weight_files = list(trained_model_dir.glob("*.pth"))
        if trained_weight_files:
            # Use the final model if it exists, otherwise use the latest checkpoint
            final_model = trained_model_dir / "superpoint_uav_final.pth"
            if final_model.exists():
                trained_weights = str(final_model)
            else:
                trained_weights = str(sorted(trained_weight_files)[-1])
    
    if trained_weights:
        print(f"Using trained weights: {trained_weights}")
        weights_to_use = trained_weights
    else:
        print(f"Using pretrained weights: {pretrained_weights}")
        weights_to_use = pretrained_weights
    
    # Step 4: Visualize results
    print(f"\n{'='*60}")
    print("STEP: Visualize SuperPoint matching")
    print(f"{'='*60}")
    
    if uav_data_dir.exists() and list(uav_data_dir.glob("*.png")):
        cmd = f"python visualize_superpoint_clean.py --weights {weights_to_use} --uav_data {uav_data_dir} --n_tests 3"
        if not run_command(cmd, "Visualize SuperPoint matching"):
            return 1
    else:
        print("⚠️ No UAV data available for visualization")
    
    # Step 5: Test with specific images if available
    earth_imagery_path = Path(args.earth_imagery_dir)
    if earth_imagery_path.exists():
        # Find some test images
        test_images = []
        for loc_dir in earth_imagery_path.glob("loc*"):
            if loc_dir.is_dir():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                    test_images.extend(list(loc_dir.glob(ext))[:2])  # Take first 2 from each location
                if len(test_images) >= 4:
                    break
        
        if len(test_images) >= 2:
            print(f"\nTesting with original earth imagery...")
            img1, img2 = test_images[0], test_images[1]
            cmd = f"python visualize_superpoint_clean.py --weights {weights_to_use} --img1 {img1} --img2 {img2} --output earth_imagery_test.png"
            run_command(cmd, "Test with original earth imagery")
    
    print(f"\n🎉 SuperPoint UAV pipeline completed!")
    print(f"Results:")
    print(f"  - Pretrained weights: {pretrained_weights}")
    if trained_weights:
        print(f"  - Trained weights: {trained_weights}")
    print(f"  - UAV data: {uav_data_dir}")
    print(f"  - Visualizations: superpoint_test_*.png")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 