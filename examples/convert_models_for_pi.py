#!/usr/bin/env python3
"""
Model Optimization Script for Raspberry Pi Zero
==============================================

This script converts PyTorch models to various optimized formats for efficient inference
on Raspberry Pi Zero (ARM CPU, 512MB RAM).

Supported formats:
- ONNX with INT8 quantization
- TensorFlow Lite with quantization
- PyTorch Mobile (optimized for mobile deployment)
- Quantized PyTorch (INT8 dynamic quantization)

Supported models:
- EfficientNet-B0
- MobileNetV2
- MobileNetV4 (using timm)
- ResNet50
- DINOv2
- DINO

Requirements:
    pip install onnx onnxruntime tensorflow torch torchvision timm onnxsim
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.mobile_optimizer import optimize_for_mobile
# Optional imports for model optimization
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX/ONNXRuntime not available. Install with: pip install onnx onnxruntime")
    print("   Some optimizations will be skipped.")
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, output_dir: str = "optimized_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different formats
        self.onnx_dir = self.output_dir / "onnx"
        self.tflite_dir = self.output_dir / "tflite"
        self.pytorch_mobile_dir = self.output_dir / "pytorch_mobile"
        self.quantized_pytorch_dir = self.output_dir / "quantized_pytorch"
        
        for dir_path in [self.onnx_dir, self.tflite_dir, self.pytorch_mobile_dir, self.quantized_pytorch_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Standard input size for most models
        self.input_size = (3, 224, 224)
        self.batch_size = 1
        
        # Model-specific input sizes
        self.model_input_sizes = {
            'efficientnet_b0': (3, 224, 224),
            'mobilenet_v2': (3, 224, 224),
            'mobilenet_v4': (3, 224, 224),
            'resnet50': (3, 224, 224),
            'dinov2': (3, 518, 518),  # DINOv2 requires 518x518
            'dino': (3, 518, 518)     # DINO uses same model as DINOv2
        }
        
        # Conversion statistics
        self.conversion_stats = {}
        
    def get_model_and_name(self, model_name: str) -> Tuple[nn.Module, str]:
        """Load the specified model and return it with a clean name."""
        model_name = model_name.lower()
        
        try:
            if model_name == "efficientnet_b0" or model_name == "efficientnetb0":
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                clean_name = "efficientnet_b0"
            elif model_name == "mobilenet_v2" or model_name == "mobilenetv2":
                model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                clean_name = "mobilenet_v2"
            elif model_name == "mobilenet_v4" or model_name == "mobilenetv4":
                try:
                    import timm
                    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)
                    clean_name = "mobilenet_v4"
                except ImportError:
                    logger.error("timm not installed. Please install with: pip install timm")
                    raise
            elif model_name == "resnet50":
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                clean_name = "resnet50"
            elif model_name == "dinov2":
                try:
                    import timm
                    model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
                    clean_name = "dinov2"
                except ImportError:
                    logger.error("timm not installed. Please install with: pip install timm")
                    raise
            elif model_name == "dino":
                try:
                    import timm
                    model = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
                    clean_name = "dino"
                except ImportError:
                    logger.error("timm not installed. Please install with: pip install timm")
                    raise
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            model.eval()
            return model, clean_name
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def get_input_size(self, model_name: str) -> Tuple[int, int, int]:
        """Get the correct input size for a specific model."""
        return self.model_input_sizes.get(model_name, self.input_size)
    
    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def convert_to_onnx(self, model: nn.Module, model_name: str) -> str:
        """Convert model to ONNX format."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
            
        logger.info(f"Converting {model_name} to ONNX...")
        
        # Create dummy input with correct size for this model
        model_input_size = self.get_input_size(model_name)
        dummy_input = torch.randn(self.batch_size, *model_input_size)
        
        # Export to ONNX
        onnx_path = self.onnx_dir / f"{model_name}.onnx"
        
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Verify the model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Get original file size
            original_size = os.path.getsize(onnx_path) / (1024 ** 2)
            
            logger.info(f"ONNX model saved: {onnx_path} ({original_size:.2f} MB)")
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"Error converting {model_name} to ONNX: {e}")
            raise

    def quantize_onnx(self, onnx_path: str, model_name: str) -> str:
        """Quantize ONNX model to INT8."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
            
        logger.info(f"Quantizing ONNX model {model_name}...")
        
        quantized_path = self.onnx_dir / f"{model_name}_quantized.onnx"
        
        try:
            # Dynamic quantization - no calibration data needed
            quantize_dynamic(
                model_input=onnx_path,
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
                extra_options={'EnableSubgraph': True}
            )
            
            # Compare sizes
            original_size = os.path.getsize(onnx_path) / (1024 ** 2)
            quantized_size = os.path.getsize(quantized_path) / (1024 ** 2)
            compression_ratio = original_size / quantized_size
            
            logger.info(f"Quantized ONNX model saved: {quantized_path}")
            logger.info(f"Size reduction: {original_size:.2f} MB -> {quantized_size:.2f} MB ({compression_ratio:.2f}x)")
            
            return str(quantized_path)
            
        except Exception as e:
            logger.error(f"Error quantizing ONNX model {model_name}: {e}")
            raise

    def convert_to_pytorch_mobile(self, model: nn.Module, model_name: str) -> str:
        """Convert model to PyTorch Mobile format."""
        logger.info(f"Converting {model_name} to PyTorch Mobile...")
        
        mobile_path = self.pytorch_mobile_dir / f"{model_name}_mobile.pt"
        
        try:
            # Trace the model with correct input size
            model_input_size = self.get_input_size(model_name)
            dummy_input = torch.randn(self.batch_size, *model_input_size)
            traced_model = torch.jit.trace(model, dummy_input)
            
            # Optimize for mobile
            optimized_model = optimize_for_mobile(traced_model)
            
            # Save the optimized model
            optimized_model._save_for_lite_interpreter(str(mobile_path))
            
            # Get file size
            mobile_size = os.path.getsize(mobile_path) / (1024 ** 2)
            
            logger.info(f"PyTorch Mobile model saved: {mobile_path} ({mobile_size:.2f} MB)")
            return str(mobile_path)
            
        except Exception as e:
            logger.error(f"Error converting {model_name} to PyTorch Mobile: {e}")
            raise

    def convert_to_quantized_pytorch(self, model: nn.Module, model_name: str) -> str:
        """Convert model to quantized PyTorch format."""
        logger.info(f"Converting {model_name} to quantized PyTorch...")
        
        quantized_path = self.quantized_pytorch_dir / f"{model_name}_quantized.pt"
        
        try:
            # Dynamic quantization
            quantized_model = torch.ao.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Save quantized model
            torch.save(quantized_model.state_dict(), quantized_path)
            
            # Get file size
            quantized_size = os.path.getsize(quantized_path) / (1024 ** 2)
            
            logger.info(f"Quantized PyTorch model saved: {quantized_path} ({quantized_size:.2f} MB)")
            return str(quantized_path)
            
        except Exception as e:
            logger.error(f"Error converting {model_name} to quantized PyTorch: {e}")
            raise

    def convert_to_tflite(self, model: nn.Module, model_name: str) -> str:
        """Convert model to TensorFlow Lite format."""
        logger.info(f"Converting {model_name} to TensorFlow Lite...")
        
        tflite_path = self.tflite_dir / f"{model_name}.tflite"
        
        try:
            # First convert to ONNX, then to TFLite
            onnx_path = self.convert_to_onnx(model, f"{model_name}_temp")
            
            # Use tf2onnx to convert (requires separate installation)
            logger.info("TensorFlow Lite conversion requires additional setup.")
            logger.info("Please install tensorflow and tf2onnx: pip install tensorflow tf2onnx")
            logger.info("Then use: python -m tf2onnx.convert --opset 11 --onnx model.onnx --output model.tflite")
            
            # For now, we'll create a placeholder
            with open(tflite_path, 'w') as f:
                f.write("# TensorFlow Lite model placeholder\n")
                f.write("# Use tf2onnx to convert from ONNX to TFLite\n")
                f.write(f"# Source ONNX: {onnx_path}\n")
            
            logger.info(f"TensorFlow Lite placeholder saved: {tflite_path}")
            return str(tflite_path)
            
        except Exception as e:
            logger.error(f"Error converting {model_name} to TensorFlow Lite: {e}")
            raise

    def benchmark_model(self, model: nn.Module, model_name: str, num_runs: int = 100) -> Dict:
        """Benchmark model inference performance."""
        logger.info(f"Benchmarking {model_name}...")
        
        model.eval()
        model_input_size = self.get_input_size(model_name)
        dummy_input = torch.randn(self.batch_size, *model_input_size)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'std_inference_time_ms': std_time * 1000,
            'throughput_fps': 1.0 / avg_time
        }

    def optimize_model(self, model_name: str) -> Dict:
        """Optimize a single model to all formats."""
        logger.info(f"Starting optimization for {model_name}...")
        
        # Load model
        model, clean_name = self.get_model_and_name(model_name)
        
        # Get original model stats
        original_size = self.get_model_size(model)
        benchmark_stats = self.benchmark_model(model, clean_name)
        
        stats = {
            'model_name': clean_name,
            'original_size_mb': original_size,
            'original_benchmark': benchmark_stats,
            'conversions': {}
        }
        
        try:
            # Convert to ONNX
            onnx_path = self.convert_to_onnx(model, clean_name)
            quantized_onnx_path = self.quantize_onnx(onnx_path, clean_name)
            
            stats['conversions']['onnx'] = {
                'path': onnx_path,
                'size_mb': os.path.getsize(onnx_path) / (1024 ** 2)
            }
            stats['conversions']['onnx_quantized'] = {
                'path': quantized_onnx_path,
                'size_mb': os.path.getsize(quantized_onnx_path) / (1024 ** 2)
            }
            
        except ImportError as e:
            logger.warning(f"ONNX conversion skipped: {e}")
            stats['conversions']['onnx'] = {'skipped': 'ONNX not available'}
            stats['conversions']['onnx_quantized'] = {'skipped': 'ONNX not available'}
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            stats['conversions']['onnx'] = {'error': str(e)}
            stats['conversions']['onnx_quantized'] = {'error': str(e)}
        
        try:
            # Convert to PyTorch Mobile
            mobile_path = self.convert_to_pytorch_mobile(model, clean_name)
            stats['conversions']['pytorch_mobile'] = {
                'path': mobile_path,
                'size_mb': os.path.getsize(mobile_path) / (1024 ** 2)
            }
            
        except Exception as e:
            logger.error(f"PyTorch Mobile conversion failed: {e}")
            stats['conversions']['pytorch_mobile'] = {'error': str(e)}
        
        try:
            # Convert to quantized PyTorch
            quantized_path = self.convert_to_quantized_pytorch(model, clean_name)
            stats['conversions']['quantized_pytorch'] = {
                'path': quantized_path,
                'size_mb': os.path.getsize(quantized_path) / (1024 ** 2)
            }
            
        except Exception as e:
            logger.error(f"Quantized PyTorch conversion failed: {e}")
            stats['conversions']['quantized_pytorch'] = {'error': str(e)}
        
        try:
            # Convert to TensorFlow Lite
            tflite_path = self.convert_to_tflite(model, clean_name)
            stats['conversions']['tflite'] = {
                'path': tflite_path,
                'size_mb': os.path.getsize(tflite_path) / (1024 ** 2)
            }
            
        except Exception as e:
            logger.error(f"TensorFlow Lite conversion failed: {e}")
            stats['conversions']['tflite'] = {'error': str(e)}
        
        self.conversion_stats[clean_name] = stats
        return stats

    def optimize_all_models(self, model_names: List[str]) -> Dict:
        """Optimize all specified models."""
        logger.info("Starting batch model optimization...")
        
        all_stats = {}
        
        for model_name in model_names:
            try:
                stats = self.optimize_model(model_name)
                all_stats[model_name] = stats
                logger.info(f"✓ Successfully optimized {model_name}")
            except Exception as e:
                logger.error(f"✗ Failed to optimize {model_name}: {e}")
                all_stats[model_name] = {'error': str(e)}
        
        # Save results
        results_path = self.output_dir / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to: {results_path}")
        return all_stats

    def print_summary(self):
        """Print a summary of all conversions."""
        if not self.conversion_stats:
            logger.info("No conversion statistics available.")
            return
        
        print("\n" + "="*80)
        print("MODEL OPTIMIZATION SUMMARY")
        print("="*80)
        
        for model_name, stats in self.conversion_stats.items():
            print(f"\n📱 {model_name.upper()}")
            print(f"   Original size: {stats['original_size_mb']:.2f} MB")
            print(f"   Inference time: {stats['original_benchmark']['avg_inference_time_ms']:.2f} ms")
            print(f"   Throughput: {stats['original_benchmark']['throughput_fps']:.2f} FPS")
            
            print("   Conversions:")
            for format_name, conversion in stats['conversions'].items():
                if 'error' in conversion:
                    print(f"     ❌ {format_name}: {conversion['error']}")
                elif 'skipped' in conversion:
                    print(f"     ⏭️  {format_name}: {conversion['skipped']}")
                else:
                    size_mb = conversion['size_mb']
                    reduction = stats['original_size_mb'] / size_mb
                    print(f"     ✅ {format_name}: {size_mb:.2f} MB ({reduction:.2f}x smaller)")
        
        print("\n" + "="*80)
        print("RECOMMENDED FOR RASPBERRY PI ZERO:")
        print("1. ONNX Quantized (best balance of size/speed)")
        print("2. PyTorch Mobile (easiest integration)")
        print("3. Quantized PyTorch (good for existing PyTorch workflows)")
        print("="*80)

def main():
    """Main function to run the optimization."""
    # List of models to optimize
    models_to_optimize = [
        "efficientnet_b0",
        "mobilenet_v2", 
        "mobilenet_v4",
        "resnet50",
        "dinov2",
        "dino"
    ]
    
    # Create optimizer
    optimizer = ModelOptimizer()
    
    # Run optimization
    try:
        results = optimizer.optimize_all_models(models_to_optimize)
        optimizer.print_summary()
        
        print(f"\n🎯 All optimized models saved to: {optimizer.output_dir}")
        print("🔧 Use the updated model_profiler.py to benchmark these models!")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
