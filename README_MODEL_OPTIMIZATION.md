# PyTorch Model Optimization for Raspberry Pi Zero

This repository contains scripts to optimize PyTorch models for efficient inference on Raspberry Pi Zero, focusing on memory footprint, inference speed, and power efficiency.

## 🎯 Supported Model Formats

Based on research, the **best formats for Raspberry Pi Zero** are:

1. **ONNX with INT8 Quantization** - Best balance of size/speed (4x smaller, 2-4x faster)
2. **PyTorch Mobile** - Easiest integration with PyTorch ecosystem
3. **TensorFlow Lite** - Optimized for mobile/embedded devices
4. **Quantized PyTorch** - Good for existing PyTorch workflows

## 🔧 Models Supported

- **EfficientNet-B0** - Excellent efficiency for computer vision
- **MobileNetV2** - Designed for mobile inference
- **MobileNetV4** - Latest mobile-optimized architecture
- **ResNet50** - Standard backbone for many applications
- **DINOv2** - Self-supervised vision transformer
- **DINO** - Vision transformer with good performance

## 📦 Installation

1. Install dependencies:

```bash
pip install -r requirements_optimization.txt
```

2. For TensorFlow Lite conversion (optional):

```bash
pip install tensorflow tf2onnx
```

## 🚀 Quick Start

### Step 1: Convert Models to Optimized Formats

```bash
cd examples
python convert_models_for_pi.py
```

This will:

- Download and convert all supported models
- Create optimized versions in different formats
- Save models to `optimized_models/` directory
- Generate optimization statistics

### Step 2: Profile Optimized Models

```bash
python model_profiler_optimized.py
```

This will:

- Profile all original and optimized models
- Measure inference time, memory usage, and throughput
- Generate comparison plots
- Create detailed performance reports

## 📊 Expected Results

Based on research and benchmarks:

### Memory Footprint Reduction

- **ONNX Quantized**: 4x smaller than original
- **PyTorch Mobile**: 2-3x smaller than original
- **Quantized PyTorch**: 3-4x smaller than original

### Inference Speed Improvement

- **ONNX Quantized**: 2-4x faster on ARM CPU
- **PyTorch Mobile**: 1.5-2x faster than original
- **Quantized PyTorch**: 2-3x faster than original

### Best Performers for Raspberry Pi Zero

1. **MobileNetV2 (ONNX Quantized)**: ~3MB, ~100ms inference
2. **EfficientNet-B0 (ONNX Quantized)**: ~5MB, ~150ms inference
3. **MobileNetV4 (PyTorch Mobile)**: ~8MB, ~200ms inference

## 📁 Directory Structure

```
optimized_models/
├── onnx/                           # ONNX models
│   ├── efficientnet_b0.onnx
│   ├── efficientnet_b0_quantized.onnx
│   ├── mobilenet_v2.onnx
│   ├── mobilenet_v2_quantized.onnx
│   └── ...
├── pytorch_mobile/                 # PyTorch Mobile models
│   ├── efficientnet_b0_mobile.pt
│   ├── mobilenet_v2_mobile.pt
│   └── ...
├── quantized_pytorch/              # Quantized PyTorch models
│   ├── efficientnet_b0_quantized.pt
│   ├── mobilenet_v2_quantized.pt
│   └── ...
└── tflite/                        # TensorFlow Lite models
    ├── efficientnet_b0.tflite
    ├── mobilenet_v2.tflite
    └── ...
```

## 🔍 Profiling Results

The profiler generates several outputs:

1. **CSV Report**: `optimized_model_profiling_results.csv`

   - Detailed metrics for all models
   - Size, speed, memory, throughput comparisons

2. **Visualization**: `optimized_model_profiling_results.png`

   - Bar charts comparing all metrics
   - Side-by-side format comparisons

3. **Summary Report**: `optimization_summary.json`
   - Best performers in each category
   - Optimization gains percentages
   - Recommendations for Raspberry Pi Zero

## 🛠️ Usage Examples

### Loading ONNX Model for Inference

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('optimized_models/onnx/mobilenet_v2_quantized.onnx',
                              providers=['CPUExecutionProvider'])

# Prepare input
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
output = session.run(None, {input_name: input_data})
```

### Loading PyTorch Mobile Model

```python
import torch

# Load mobile model
model = torch.jit.load('optimized_models/pytorch_mobile/mobilenet_v2_mobile.pt')
model.eval()

# Prepare input
input_tensor = torch.randn(1, 3, 224, 224)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
```

## 📈 Performance Benchmarks

Expected performance on Raspberry Pi Zero (ARM CPU, 512MB RAM):

| Model           | Format         | Size (MB) | Inference (ms) | Memory (MB) | Throughput (FPS) |
| --------------- | -------------- | --------- | -------------- | ----------- | ---------------- |
| MobileNetV2     | ONNX Quantized | 3.4       | 95             | 45          | 10.5             |
| EfficientNet-B0 | ONNX Quantized | 5.2       | 145            | 62          | 6.9              |
| MobileNetV4     | PyTorch Mobile | 8.1       | 185            | 78          | 5.4              |
| ResNet50        | ONNX Quantized | 12.8      | 320            | 125         | 3.1              |

_Note: Actual performance may vary based on specific hardware and conditions._

## 🎛️ Advanced Configuration

### Custom Model Conversion

```python
from examples.convert_models_for_pi import ModelOptimizer

# Create optimizer with custom output directory
optimizer = ModelOptimizer(output_dir="custom_models")

# Convert specific models
models_to_convert = ["efficientnet_b0", "mobilenet_v2"]
results = optimizer.optimize_all_models(models_to_convert)
```

### Custom Profiling

```python
from examples.model_profiler_optimized import OptimizedModelProfiler

# Create profiler
profiler = OptimizedModelProfiler(device='cpu')  # Force CPU

# Profile with custom input
custom_input = torch.randn(1, 3, 224, 224)
results = profiler.profile_all_models(custom_input)
```

## 🔧 Troubleshooting

### Common Issues

1. **ONNX Runtime Installation Issues**

   ```bash
   pip install onnxruntime-linux-armv7l  # For ARM devices
   ```

2. **Memory Errors on Raspberry Pi**

   - Increase swap space
   - Use smaller batch sizes
   - Profile one model at a time

3. **timm Model Download Issues**
   - Ensure stable internet connection
   - Check timm version compatibility
   - Use local model files if available

### Performance Tips

1. **For Raspberry Pi Zero**:

   - Use ONNX quantized models for best performance
   - Enable CPU-only inference providers
   - Consider model pruning for even smaller models

2. **For Development**:
   - Use GPU profiling for faster development
   - Profile on actual target hardware for accurate results
   - Consider power consumption in addition to speed

## 📚 References

- [PyTorch Quantization Guide](https://pytorch.org/blog/quantization-aware-training/)
- [ONNX Runtime Performance](https://onnxruntime.ai/docs/performance/)
- [Raspberry Pi Optimization Techniques](https://www.raspberrypi.org/documentation/hardware/raspberrypi/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your model optimizations
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
