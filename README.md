# DINO Benchmark Suite for Raspberry Pi Zero

A comprehensive benchmarking toolkit for evaluating DINO (Self-Distillation with No Labels) Vision Transformer models under Raspberry Pi Zero constraints (512MB RAM, ARM v6 CPU).

## 🎯 Overview

This benchmark suite simulates the performance of various DINO model configurations on Raspberry Pi Zero hardware, focusing on:

- **Memory Usage**: Monitoring peak and average memory consumption
- **Quantization Effects**: Testing 8-bit, 4-bit, and binary quantization
- **Inference Performance**: Measuring latency and throughput
- **Edge Compatibility**: Determining Pi Zero viability
- **Accuracy Trade-offs**: Evaluating quantization impact on model performance

## 🚀 Quick Start

### Option 1: Simplified Benchmark (No Dependencies)
```bash
python3 dino_benchmark_simple.py
```

### Option 2: Full Benchmark (Requires PyTorch)
```bash
# Setup environment
python3 setup_benchmark.py

# Run benchmark
python3 dino_benchmark.py
```

## 📊 Results Summary

Based on our simulation with Raspberry Pi Zero constraints:

### 🏆 Best Performers

| Metric | Model | Quantization | Value |
|--------|-------|-------------|-------|
| **Best Accuracy** | DINO-Small | none | 79.8% |
| **Best Speed** | DINO-Micro | binary | 22.1 FPS |
| **Smallest Model** | DINO-Micro | binary | 0.1 MB |
| **Best Efficiency** | DINO-Micro | binary | 12.08 acc/sec |

### 📈 Model Compatibility

✅ **All tested configurations are Pi Zero compatible!**

| Model | none | 8bit | 4bit | binary |
|-------|------|------|------|--------|
| DINO-Micro | ✅ 54.8MB | ✅ 30.2MB | ✅ 23.1MB | ✅ 10.3MB |
| DINO-Tiny | ✅ 61.1MB | ✅ 31.7MB | ✅ 25.0MB | ✅ 12.2MB |
| DINO-Small | ✅ 97.7MB | ✅ 57.6MB | ✅ 39.7MB | ✅ 19.0MB |

### 🔧 Quantization Effects

| Quantization | Avg Accuracy | Avg Speed | Avg Size | Size Reduction |
|-------------|-------------|-----------|----------|----------------|
| none | 76.3% | 5.4 FPS | 18.9 MB | 0% |
| 8bit | 74.6% | 6.8 FPS | 4.7 MB | 75% |
| 4bit | 69.9% | 9.2 FPS | 2.4 MB | 87.5% |
| binary | 56.9% | 15.0 FPS | 0.6 MB | 96.9% |

## 🛠️ Technical Details

### Model Architectures

| Model | Dimensions | Depth | Heads | MLP Dim | Parameters |
|-------|------------|-------|-------|---------|------------|
| DINO-Micro | 128 | 3 | 2 | 512 | ~0.8M |
| DINO-Tiny | 192 | 4 | 3 | 768 | ~2.1M |
| DINO-Small | 384 | 6 | 6 | 1536 | ~11.3M |

### Raspberry Pi Zero Specs
- **Memory**: 512MB RAM
- **CPU**: Single-core ARM v6 @ 1GHz
- **GPU**: None
- **Architecture**: ARMv6

### Quantization Methods

1. **8-bit Quantization**: Reduces model size by 75% with minimal accuracy loss
2. **4-bit Quantization**: Aggressive compression (87.5% reduction) with moderate accuracy drop
3. **Binary Quantization**: Extreme compression (96.9% reduction) with significant accuracy impact

## 💡 Recommendations

### For Raspberry Pi Zero Deployment:

1. **Model Selection**:
   - Use **DINO-Micro** or **DINO-Tiny** for optimal compatibility
   - Avoid DINO-Small unless accuracy is critical

2. **Quantization Strategy**:
   - **8-bit**: Best accuracy-size trade-off (recommended)
   - **4-bit**: Good for size-constrained applications
   - **Binary**: Only for extreme size requirements

3. **Optimization Techniques**:
   - Process single images (batch size = 1)
   - Implement model pruning
   - Use mixed-precision quantization
   - Consider knowledge distillation

4. **Memory Management**:
   - Monitor memory usage continuously
   - Implement dynamic model loading
   - Use memory-mapped model storage

## 📁 File Structure

```
dino-benchmark/
├── dino_benchmark.py              # Full benchmark with PyTorch
├── dino_benchmark_simple.py       # Dependency-free simulation
├── setup_benchmark.py             # Environment setup
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── dino_benchmark_results.json    # Generated results
└── benchmark_config.json          # Configuration file
```

## 🔧 Setup Instructions

### System Requirements
- Python 3.7+
- 512MB+ RAM (for testing)
- ARM or x86/x64 architecture

### Installation

1. **Clone/Download** the benchmark files
2. **Run setup** (optional, for full PyTorch benchmark):
   ```bash
   python3 setup_benchmark.py
   ```
3. **Execute benchmark**:
   ```bash
   # Simple version (no dependencies)
   python3 dino_benchmark_simple.py
   
   # Full version (requires PyTorch)
   python3 dino_benchmark.py
   ```

## 📊 Output Files

### `dino_benchmark_results.json`
Detailed benchmark results including:
- Model configurations
- Performance metrics
- Memory usage statistics
- Compatibility flags

### Console Output
Real-time progress with:
- Memory usage monitoring
- Performance visualizations
- Compatibility analysis
- Optimization recommendations

## 🎨 Visualizations

The benchmark generates ASCII-based visualizations:

```
📊 Memory Usage by Model (MB):
  DINO-Micro: █████                    54.8MB ✅
   DINO-Tiny: ██████                   61.1MB ✅
  DINO-Small: █████████                97.7MB ✅

🔧 Quantization Effects (Size Reduction):
8bit Quantization:
    DINO-Micro: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     75.0% reduction
```

## 🚨 Important Notes

### Simulation Accuracy
- Results are based on research data and mathematical models
- Actual hardware performance may vary
- Memory estimations include model weights and activations
- Timing simulations account for ARM v6 performance characteristics

### Limitations
- Network I/O not simulated
- Thermal throttling not considered
- SD card access latency not included
- Real-world optimizations may improve performance

## 🔬 Research Background

### DINO (Self-Distillation with No Labels)
DINO is a self-supervised learning method for Vision Transformers that:
- Uses teacher-student knowledge distillation
- Requires no labeled data for training
- Produces excellent semantic segmentation features
- Achieves strong k-NN classification performance

### Key Publications
- "Emerging Properties in Self-Supervised Vision Transformers" (Caron et al., 2021)
- Various quantization papers for Vision Transformers
- Edge deployment optimization research

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Real hardware validation
- Additional model architectures
- Advanced quantization techniques
- Optimization strategies
- Mobile deployment examples

## 📜 License

This benchmark suite is provided as-is for research and educational purposes.

## 🔗 References

1. [DINO Paper](https://arxiv.org/abs/2104.14294)
2. [Vision Transformer Quantization Survey](https://arxiv.org/abs/2106.14156)
3. [Binary Neural Networks for Edge Computing](https://arxiv.org/abs/1602.02830)
4. [Raspberry Pi Zero Documentation](https://www.raspberrypi.org/products/raspberry-pi-zero/)

---

**Last Updated**: July 2025  
**Status**: Active Development  
**Compatibility**: Raspberry Pi Zero, Pi Zero W, Pi Zero 2 W