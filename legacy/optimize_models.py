#!/usr/bin/env python3
"""
Model Optimization Script for Raspberry Pi Zero Deployment
Supports:
- PyTorch Mobile conversion
- INT8/INT4 quantization
- Model size reduction
- Performance benchmarking
- Different input sizes for different models
"""

import os
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.mobile_optimizer import optimize_for_mobile
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import models
try:
    from simple_superpoint import SuperPoint, SuperPointNet
    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False
    print("SuperPoint not available - skipping")

import torchvision.models as models
import timm

class ModelOptimizer:
    def __init__(self, device='cpu'):  # Use CPU by default for Pi Zero compatibility
        self.device = device
        self.models = {}
        self.model_configs = {}
        self.results = {}
        
    def get_model_config(self, model_name):
        """Get configuration for each model type."""
        configs = {
            'superpoint': {
                'input_size': (1, 224, 224),
                'batch_size': 1,
                'description': 'SuperPoint keypoint detector'
            },
            'mobilenetv2': {
                'input_size': (3, 224, 224),
                'batch_size': 1,
                'description': 'MobileNetV2 classification'
            },
            'mobilenetv3': {
                'input_size': (3, 224, 224),
                'batch_size': 1,
                'description': 'MobileNetV3 classification'
            },
            'efficientnet': {
                'input_size': (3, 224, 224),
                'batch_size': 1,
                'description': 'EfficientNet-B0 classification'
            },
            'resnet50': {
                'input_size': (3, 224, 224),
                'batch_size': 1,
                'description': 'ResNet-50 classification'
            },
            'dino': {
                'input_size': (3, 224, 224),
                'batch_size': 1,
                'description': 'DINO vision transformer'
            },
            'dinov2': {
                'input_size': (3, 518, 518),
                'batch_size': 1,
                'description': 'DINOv2 vision transformer'
            }
        }
        return configs.get(model_name, configs['mobilenetv2'])
        
    def load_models(self):
        """Load all models for optimization."""
        print("Loading models...")
        
        # Load SuperPoint
        if SUPERPOINT_AVAILABLE:
            try:
                superpoint = SuperPointNet().to(self.device)
                superpoint.eval()
                self.models['superpoint'] = superpoint
                print("✓ SuperPoint loaded")
            except Exception as e:
                print(f"✗ Error loading SuperPoint: {e}")
        
        # MobileNetV2
        try:
            mobilenetv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            mobilenetv2.eval()
            self.models['mobilenetv2'] = mobilenetv2
            print("✓ MobileNetV2 loaded")
        except Exception as e:
            print(f"✗ Error loading MobileNetV2: {e}")
            
        # MobileNetV3 (as MobileNetV4 placeholder)
        try:
            mobilenetv3 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            mobilenetv3.eval()
            self.models['mobilenetv3'] = mobilenetv3
            print("✓ MobileNetV3 loaded")
        except Exception as e:
            print(f"✗ Error loading MobileNetV3: {e}")
            
        # EfficientNet-B0
        try:
            efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
            efficientnet.eval()
            self.models['efficientnet'] = efficientnet
            print("✓ EfficientNet-B0 loaded")
        except Exception as e:
            print(f"✗ Error loading EfficientNet-B0: {e}")
            
        # ResNet50
        try:
            resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            resnet50.eval()
            self.models['resnet50'] = resnet50
            print("✓ ResNet50 loaded")
        except Exception as e:
            print(f"✗ Error loading ResNet50: {e}")
        
        # DINO (original)
        try:
            dino = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
            dino.eval()
            self.models['dino'] = dino
            print("✓ DINO loaded")
        except Exception as e:
            print(f"✗ Error loading DINO: {e}")
            
        # DINOv2
        try:
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            dinov2.eval()
            self.models['dinov2'] = dinov2
            print("✓ DINOv2 loaded")
        except Exception as e:
            print(f"✗ Error loading DINOv2: {e}")
            
        print(f"Successfully loaded {len(self.models)} models")
        
    def prepare_for_quantization(self, model, model_name):
        """Prepare model for quantization by adding necessary observers."""
        try:
            # Clone the model to avoid modifying original
            model_copy = type(model)()
            model_copy.load_state_dict(model.state_dict())
            model_copy.eval()
            
            # Set quantization config
            model_copy.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare model
            model_prepared = torch.quantization.prepare(model_copy)
            return model_prepared
        except Exception as e:
            print(f"Error preparing {model_name} for quantization: {e}")
            return None
            
    def quantize_model(self, model, model_name, quantization_type='int8'):
        """Quantize model to INT8 or INT4."""
        print(f"  Quantizing to {quantization_type.upper()}...")
        
        try:
            # Get model config
            config = self.get_model_config(model_name)
            input_size = config['input_size']
            
            # Prepare model for quantization
            model_prepared = self.prepare_for_quantization(model, model_name)
            if model_prepared is None:
                return None
                
            # Generate calibration data
            calibration_data = torch.randn(5, *input_size)
            
            # Run calibration
            with torch.no_grad():
                for i in range(len(calibration_data)):
                    try:
                        _ = model_prepared(calibration_data[i:i+1])
                    except Exception as e:
                        print(f"Calibration error for {model_name}: {e}")
                        return None
                        
            # Quantize
            if quantization_type == 'int8':
                model_quantized = torch.quantization.convert(model_prepared)
            else:
                # Try dynamic quantization for INT4
                model_quantized = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8  # Use INT8 as INT4 is very limited
                )
                
            return model_quantized
        except Exception as e:
            print(f"    Error quantizing {model_name}: {e}")
            return None
            
    def optimize_for_mobile(self, model, model_name):
        """Optimize model for mobile deployment."""
        print(f"  Optimizing for mobile...")
        
        try:
            # Get model config
            config = self.get_model_config(model_name)
            input_size = config['input_size']
            
            # Create example input
            example_input = torch.randn(1, *input_size)
            
            # Trace the model
            model.eval()
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
            
            # Optimize for mobile
            optimized_model = optimize_for_mobile(traced_model)
            
            return optimized_model
        except Exception as e:
            print(f"    Error optimizing {model_name} for mobile: {e}")
            return None
            
    def benchmark_model(self, model, model_name, variant_name, num_runs=50):
        """Benchmark model performance."""
        print(f"    Benchmarking {variant_name}...")
        
        try:
            # Get model config
            base_model_name = model_name.split('_')[0]
            config = self.get_model_config(base_model_name)
            input_size = config['input_size']
            
            results = {
                'model_name': variant_name,
                'base_model': base_model_name,
                'variant': variant_name.split('_')[-1],
                'description': config['description'],
                'input_size': input_size,
                'size_mb': self.get_model_size(model),
                'inference_times_ms': [],
                'memory_usage_mb': []
            }
            
            # Generate sample input
            input_tensor = torch.randn(1, *input_size)
                
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    try:
                        _ = model(input_tensor)
                    except Exception as e:
                        print(f"    Warmup failed for {variant_name}: {e}")
                        return None
                    
            # Benchmark
            with torch.no_grad():
                for _ in range(num_runs):
                    try:
                        start_time = time.time()
                        _ = model(input_tensor)
                        inference_time = (time.time() - start_time) * 1000  # Convert to ms
                        results['inference_times_ms'].append(inference_time)
                        
                        # Memory usage (basic approximation)
                        memory_usage = results['size_mb']  # Approximate
                        results['memory_usage_mb'].append(memory_usage)
                    except Exception as e:
                        print(f"    Benchmark iteration failed for {variant_name}: {e}")
                        continue
                        
            if not results['inference_times_ms']:
                print(f"    No successful benchmark runs for {variant_name}")
                return None
                
            # Calculate statistics
            results['avg_inference_time_ms'] = np.mean(results['inference_times_ms'])
            results['std_inference_time_ms'] = np.std(results['inference_times_ms'])
            results['min_inference_time_ms'] = np.min(results['inference_times_ms'])
            results['max_inference_time_ms'] = np.max(results['inference_times_ms'])
            results['avg_memory_usage_mb'] = np.mean(results['memory_usage_mb'])
            results['throughput_fps'] = 1000 / results['avg_inference_time_ms']
            
            # Remove raw data to save space
            del results['inference_times_ms']
            del results['memory_usage_mb']
            
            return results
        except Exception as e:
            print(f"    Error benchmarking {variant_name}: {e}")
            return None
        
    def get_model_size(self, model):
        """Get model size in MB."""
        try:
            if hasattr(model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            else:
                # For traced models, estimate size
                param_size = sum(p.numel() * 4 for p in model.parameters())  # Assume float32
                buffer_size = 0
                
            size_mb = (param_size + buffer_size) / (1024 * 1024)
            return size_mb
        except Exception as e:
            print(f"    Error calculating model size: {e}")
            return 0.0
        
    def optimize_all_models(self):
        """Optimize all loaded models."""
        print("\nStarting model optimization...")
        
        total_models = len(self.models)
        processed = 0
        
        for model_name, model in self.models.items():
            processed += 1
            print(f"\n[{processed}/{total_models}] Processing {model_name}...")
            
            # Original model benchmarks
            print(f"  Original model:")
            original_results = self.benchmark_model(model, model_name, f"{model_name}_original")
            if original_results:
                self.results[f"{model_name}_original"] = original_results
            
            # INT8 Quantization
            print(f"  INT8 quantization:")
            model_int8 = self.quantize_model(model, model_name, 'int8')
            if model_int8 is not None:
                int8_results = self.benchmark_model(model_int8, model_name, f"{model_name}_int8")
                if int8_results:
                    self.results[f"{model_name}_int8"] = int8_results
                    
            # Mobile Optimization
            print(f"  Mobile optimization:")
            model_mobile = self.optimize_for_mobile(model, model_name)
            if model_mobile is not None:
                mobile_results = self.benchmark_model(model_mobile, model_name, f"{model_name}_mobile")
                if mobile_results:
                    self.results[f"{model_name}_mobile"] = mobile_results
                    
        print(f"\nOptimization complete! Processed {len(self.results)} model variants.")
                
    def save_results(self, output_dir='optimization_results'):
        """Save optimization results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save results as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f"optimization_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nResults saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        
        if self.results:
            for model_name, results in self.results.items():
                print(f"\n{model_name}:")
                print(f"  Size: {results['size_mb']:.2f} MB")
                print(f"  Avg Inference: {results['avg_inference_time_ms']:.2f} ms")
                print(f"  Throughput: {results['throughput_fps']:.2f} FPS")
                print(f"  Input Size: {results['input_size']}")
        else:
            print("No results to display.")
            
def main():
    print("PyTorch Model Optimizer for Raspberry Pi Zero")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Load models
    optimizer.load_models()
    
    if not optimizer.models:
        print("No models loaded successfully. Exiting.")
        return
    
    # Optimize and benchmark
    optimizer.optimize_all_models()
    
    # Save results
    optimizer.save_results()
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\nCheck the optimization_results/ directory for detailed results.")

if __name__ == "__main__":
    main() 