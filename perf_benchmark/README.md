# Performance Benchmark for Pi Zero

This directory contains tools for benchmarking DINO models on Raspberry Pi Zero.

## Setup

### 1. Model Creation (Run on development machine)

```bash
# Create quantized models for Pi Zero testing
python create_tflite_models.py
```

This will:

- Create `perf_benchmark/models/` directory
- Generate INT4/INT8 quantized models
- Save model metadata in `model_creation_summary.json`

### 2. Pi Zero Benchmarking

```bash
# On Pi Zero, install minimal dependencies
pip install torch psutil

# Run the benchmark
python perf_benchmark/pi_zero_benchmark.py
```

## Models Tested

- **DINO-S/16** (INT4, INT8) - Small model, input 112x112
- **DINOv2-S/14** (INT4, INT8) - Small DINOv2, input 112x112
- **DINOv2-B/14** (INT4 only) - Base model, larger

## Pi Zero Requirements

- **RAM**: 512MB (requires swap for larger models)
- **Dependencies**: `torch`, `psutil` (minimal PyTorch setup)
- **Storage**: ~50-100MB for quantized models

## Results

Results are saved to `perf_benchmark_results.json` with:

- Model loading time
- Memory usage
- Inference FPS
- Real vs theoretical size comparison

## Tips for Pi Zero

1. **Add swap memory**: `sudo fallocate -l 1G /swapfile`
2. **Use CPU-only PyTorch**: `--index-url https://download.pytorch.org/whl/cpu`
3. **Start with smallest models first**
4. **Monitor memory usage** during benchmarks
