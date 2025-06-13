#!/bin/bash

# Set output directory
OUTPUT_DIR="superpoint_training"
mkdir -p "$OUTPUT_DIR"

# Stage 1: Generate synthetic data
echo "Stage 1: Generating synthetic data..."
PYTHONPATH=. python3 -m src.models.superpoint.synthetic_shapes \
    --output_dir "$OUTPUT_DIR/synthetic_data" \
    --n_images 50000 \
    --img_size 256

# Stage 2: Train MagicPoint on synthetic data
echo "Stage 2: Training MagicPoint..."
PYTHONPATH=. python3 -m src.models.superpoint.finetune_superpoint \
    --data_dir "$OUTPUT_DIR/synthetic_data" \
    --output_dir "$OUTPUT_DIR/magicpoint" \
    --stage magicpoint \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001

# Stage 3a: Prepare UAV training data from earth imagery
echo "Stage 3a: Preparing UAV training data from earth imagery..."
PYTHONPATH=. python3 -m src.models.superpoint.prepare_uav_data \
    --earth_imagery_dir "data/earth_imagery" \
    --output_dir "$OUTPUT_DIR/uav_crops" \
    --n_crops_per_location 5000 \
    --crop_size 256

# Stage 3b: Homographic adaptation on UAV crops
echo "Stage 3b: Performing homographic adaptation on UAV crops..."
PYTHONPATH=. python3 -m src.models.superpoint.homographic_adaptation \
    --image_dir "$OUTPUT_DIR/uav_crops" \
    --model_path "$OUTPUT_DIR/magicpoint/final.pth" \
    --output_dir "$OUTPUT_DIR/homographic_data" \
    --n_views 100 \
    --batch_size 8  # Increased batch size for H100

# Stage 4: Train SuperPoint on UAV data
echo "Stage 4: Training SuperPoint..."
PYTHONPATH=. python3 -m src.models.superpoint.finetune_superpoint \
    --data_dir "$OUTPUT_DIR/homographic_data" \
    --output_dir "$OUTPUT_DIR/superpoint" \
    --stage superpoint \
    --epochs 50 \
    --batch_size 32  # Can be increased on H100 if memory allows
    --lr 0.001

echo "Training pipeline complete!" 