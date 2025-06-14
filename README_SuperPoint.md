# SuperPoint UAV Fine-tuning Pipeline

A clean, simple implementation for fine-tuning SuperPoint on UAV/drone imagery using proven pretrained weights.

## Overview

This pipeline:

1. **Downloads** proven pretrained SuperPoint weights from established repositories
2. **Generates** UAV training data from your earth imagery
3. **Fine-tunes** only the descriptor head on UAV-specific data
4. **Visualizes** matching results

## Quick Start

Run the complete pipeline:

```bash
python run_superpoint_pipeline.py
```

Or run individual steps:

### 1. Download Pretrained Weights

```bash
python download_pretrained_superpoint.py
```

### 2. Generate UAV Data

```bash
python generate_uav_data.py --earth_imagery_dir data/earth_imagery --n_crops_per_location 1000
```

### 3. Train SuperPoint

```bash
python train_superpoint_uav.py --data_dir uav_data --pretrained_weights pretrained_weights/superpoint_v1.pth --epochs 20
```

### 4. Visualize Results

```bash
# Test with UAV data
python visualize_superpoint_clean.py --weights superpoint_uav_trained/superpoint_uav_final.pth --uav_data uav_data

# Test with specific images
python visualize_superpoint_clean.py --weights superpoint_uav_trained/superpoint_uav_final.pth --img1 image1.jpg --img2 image2.jpg
```

## Files

- `simple_superpoint.py` - Clean SuperPoint implementation compatible with proven weights
- `download_pretrained_superpoint.py` - Download pretrained weights
- `generate_uav_data.py` - Generate UAV training data with augmentation
- `train_superpoint_uav.py` - Fine-tune SuperPoint on UAV data
- `visualize_superpoint_clean.py` - Clean visualization of matching results
- `run_superpoint_pipeline.py` - All-in-one pipeline script

## Key Features

- **Proven weights**: Uses established pretrained SuperPoint weights
- **UAV-focused**: Generates realistic UAV crops with augmentation
- **Descriptor-only training**: Only fine-tunes descriptor head, keeps detector frozen
- **Clean visualization**: Shows actual SuperPoint performance without artificial enhancements
- **Homographic adaptation**: Uses proper homographic pairs for training

## Requirements

```bash
pip install torch opencv-python numpy tqdm requests
```

## Training Strategy

- Starts with proven pretrained SuperPoint weights
- Freezes encoder and detector (keypoint detection)
- Only trains descriptor head on UAV-specific data
- Uses homographic adaptation for self-supervised learning
- Fast training (10-20 epochs typically sufficient)

This approach ensures the keypoint detector remains robust while adapting descriptors to UAV imagery characteristics.
