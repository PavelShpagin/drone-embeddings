#!/bin/bash
set -e  # Exit on error

OUTPUT_DIR="superpoint_training"

echo "Quick SuperPoint retraining with fixed descriptor loss..."

# Check if we have MagicPoint model
if [ ! -f "$OUTPUT_DIR/magicpoint/final.pth" ]; then
    echo "Error: MagicPoint model not found at $OUTPUT_DIR/magicpoint/final.pth"
    echo "Please run the full pipeline first or check the path."
    exit 1
fi

# Stage 3a: Prepare minimal UAV training data (just 100 crops per location for testing)
echo "Stage 3a: Preparing minimal UAV training data..."
PYTHONPATH=. python3 -m src.models.superpoint.prepare_uav_data \
    --earth_imagery_dir "data/earth_imagery" \
    --output_dir "$OUTPUT_DIR/uav_crops_test" \
    --n_crops_per_location 100 \
    --crop_size 256

# Stage 3b: Homographic adaptation (reduced views for speed)
echo "Stage 3b: Performing homographic adaptation..."
PYTHONPATH=. python3 -m src.models.superpoint.homographic_adaptation \
    --image_dir "$OUTPUT_DIR/uav_crops_test" \
    --model_path "$OUTPUT_DIR/magicpoint/final.pth" \
    --output_dir "$OUTPUT_DIR/homographic_data_test" \
    --n_views 20 \
    --batch_size 8 \
    --num_workers 8

# Stage 4: Train SuperPoint with fixed descriptor loss (short training for testing)
echo "Stage 4: Training SuperPoint with fixes..."
PYTHONPATH=. python3 -m src.models.superpoint.finetune_superpoint \
    --data_dir "$OUTPUT_DIR/homographic_data_test" \
    --output_dir "$OUTPUT_DIR/superpoint_fixed" \
    --stage superpoint \
    --epochs 10 \
    --batch_size 16 \
    --lr 0.001

echo "Quick retraining complete! Testing the fixed model..."

# Test the fixed model
echo "Testing fixed SuperPoint model..."
python examples/visualize_superpoint_matches.py

echo "Done! Check the visualization to see if descriptors are no longer zeros." 