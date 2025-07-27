#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Navigate to the GLVL directory
cd third_party/GLVL

echo "Starting GLVL SuperPoint model training..."
# Run the training script
# We're setting --device to cuda here to explicitly use the GPU, as requested for H100.
# --epochs_num can be adjusted based on desired training length
python3 train.py --device cuda --epochs_num 10

echo "Training complete. Starting evaluation..."
# Run the evaluation script
# --test_dataset_name can be changed to 'RSSDIVCS' or other datasets if needed.
# This will use the best model saved during training.
python3 eval_joint.py --device cuda --test_dataset_name village

echo "GLVL SuperPoint training and evaluation complete." 