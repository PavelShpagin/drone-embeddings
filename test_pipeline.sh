#!/bin/bash
set -e  # Exit on error

# Set output directory
OUTPUT_DIR="superpoint_test_run"
mkdir -p "$OUTPUT_DIR"

echo "Testing SuperPoint training pipeline..."

# Stage 1: Generate minimal synthetic data
echo "Stage 1: Generating synthetic data..."
PYTHONPATH=. python -m src.models.superpoint.synthetic_shapes \
    --output_dir "$OUTPUT_DIR/synthetic_data" \
    --n_images 10 \
    --img_size 256

# Stage 2: Train MagicPoint on synthetic data (minimal)
echo "Stage 2: Training MagicPoint..."
PYTHONPATH=. python -m src.models.superpoint.finetune_superpoint \
    --data_dir "$OUTPUT_DIR/synthetic_data" \
    --output_dir "$OUTPUT_DIR/magicpoint" \
    --stage magicpoint \
    --epochs 2 \
    --batch_size 2 \
    --lr 0.001

# Stage 3a: Prepare UAV training data (minimal)
echo "Stage 3a: Preparing UAV training data..."
PYTHONPATH=. python -m src.models.superpoint.prepare_uav_data \
    --earth_imagery_dir "data/earth_imagery" \
    --output_dir "$OUTPUT_DIR/uav_crops" \
    --n_crops_per_location 3 \
    --crop_size 256

# Stage 3b: Homographic adaptation on UAV crops
echo "Stage 3b: Performing homographic adaptation..."
PYTHONPATH=. python -m src.models.superpoint.homographic_adaptation \
    --image_dir "$OUTPUT_DIR/uav_crops" \
    --model_path "$OUTPUT_DIR/magicpoint/final.pth" \
    --output_dir "$OUTPUT_DIR/homographic_data" \
    --n_views 2 \
    --batch_size 2

# Stage 4: Train SuperPoint (minimal)
echo "Stage 4: Training SuperPoint..."
PYTHONPATH=. python -m src.models.superpoint.finetune_superpoint \
    --data_dir "$OUTPUT_DIR/homographic_data" \
    --output_dir "$OUTPUT_DIR/superpoint" \
    --stage superpoint \
    --epochs 2 \
    --batch_size 2 \
    --lr 0.001

echo "Test pipeline complete! Check $OUTPUT_DIR for results." 