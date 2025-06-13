# SuperPoint Plug-and-Play Module

This module provides a plug-and-play implementation of SuperPoint for keypoint detection and description, using HuggingFace Transformers. It supports:

- Pretrained SuperPoint loading and inference
- Synthetic data generation for self-supervised pretraining
- Large-scale finetuning (e.g., on H100)
- Easy integration with the geolocalization simulation system

## Usage

### 1. Inference

```python
from src.models.superpoint.superpoint_model import SuperPoint
import cv2
sp = SuperPoint(device='cuda')
img = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)
keypoints, scores, descriptors = sp.detect(img)
```

### 2. Generate Synthetic Data

```bash
python -m src.models.superpoint.synthetic_shapes --output_dir synthetic_shapes --n_images 10000 --img_size 128
```

### 3. Finetune SuperPoint

```bash
python -m src.models.superpoint.finetune_superpoint --data_dir synthetic_shapes --output_dir superpoint_ckpts --epochs 50 --batch_size 64 --device cuda
```

## Integration

- Import and use `SuperPoint` in your simulation or retrieval pipeline.
- You can switch between pretrained and finetuned weights by changing the model loading in `superpoint_model.py`.
- To test the effect on retrieval accuracy, use the SuperPoint descriptors for patch matching in your geolocalization system.

## Requirements

- torch
- transformers
- opencv-python
- Pillow
- tqdm

## References

- [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
- [magic-leap-community/superpoint on HuggingFace](https://huggingface.co/magic-leap-community/superpoint)
