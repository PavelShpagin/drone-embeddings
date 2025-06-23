#!/bin/bash
set -e

# Activate conda or python environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

# Set CUDA device (if needed)
export CUDA_VISIBLE_DEVICES=0

# Install dependencies (if not already installed)
pip install -r third_party/pytorch-superpoint/requirements.txt
pip install -r third_party/pytorch-superpoint/requirements_torch.txt
pip install opencv-python-headless tqdm imageio tensorboardX pyyaml

# Move to MagicPoint repo
echo "[INFO] Running MagicPoint keypoint extraction for RSSDIVCS..."
python3 third_party/pytorch-superpoint/export.py export_detector_homoAdapt \
    third_party/pytorch-superpoint/configs/magicpoint_rssdivcs_export.yaml rssdivcs_export --debug

echo "[INFO] Running MagicPoint keypoint extraction for village..."
python3 third_party/pytorch-superpoint/export.py export_detector_homoAdapt \
    third_party/pytorch-superpoint/configs/magicpoint_village_export.yaml village_export --debug

# Placeholder: Run GLVL training and evaluation scripts
# echo "[INFO] Running GLVL training..."
# python3 third_party/GLVL/train.py --config your_config.yaml
# echo "[INFO] Running GLVL evaluation..."
# python3 third_party/GLVL/eval_joint.py --config your_config.yaml

echo "[INFO] Pipeline complete. Keypoints exported. Ready for GLVL training/eval." 