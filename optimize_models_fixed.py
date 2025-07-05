#!/usr/bin/env python3
"""
Fixed Model Optimization Script for Raspberry Pi Zero Deployment
Properly implements PyTorch Mobile optimizations with verification
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
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.results = {}
        self.saved_models_dir = Path('optimization_results/saved_models')
        self.saved_models_dir.mkdir(parents=True, exist_ok=True)
        
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
            
        # MobileNetV3
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
        
    def get_model_size_mb(self, model):
        """Get model size in MB with proper handling for different model types."""
        try:
            if hasattr(model, '_save_to_state_dict'):
                # For traced/scripted models, save to file and check size
                temp_path = '/tmp/temp_model.pt'
                model.save(temp_path)
                size_bytes = os.path.getsize(temp_path)
                os.remove(temp_path)
                return size_bytes / (1024 * 1024)
            elif hasattr(model, 'parameters'):
                # For regular PyTorch models
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / (1024 * 1024)
            else:
                return 0.0
        except Exception as e:
            print(f"    Warning: Could not calculate model size: {e}")
            return 0.0
    
    def optimize_for_mobile(self, model, model_name):
        """Optimize model for mobile deployment with verification."""
        print(f"  Optimizing for PyTorch Mobile...")
        
        try:
            # Get model config
            config = self.get_model_config(model_name)
            input_size = config['input_size']
            
            # Create example input
            example_input = torch.randn(1, *input_size)
            
            # Step 1: Trace the model
            print(f"    Tracing model...")
            model.eval()
            with torch.no_grad():
                # Verify model works with example input first
                original_output = model(example_input)
                traced_model = torch.jit.trace(model, example_input)
                traced_output = traced_model(example_input)
                
                # Verify traced model produces same output
                if isinstance(original_output, tuple):
                    output_diff = torch.max(torch.abs(original_output[0] - traced_output[0])).item()
                else:
                    output_diff = torch.max(torch.abs(original_output - traced_output)).item()
                
                if output_diff > 1e-5:
                    print(f"    Warning: Traced model output differs by {output_diff}")
                else:
                    print(f"    ✓ Tracing successful, output difference: {output_diff:.2e}")
            
            # Step 2: Apply mobile optimizations
            print(f"    Applying mobile optimizations...")
            optimized_model = optimize_for_mobile(
                traced_model,
                optimization_blocklist={
                    # Exclude problematic ops if needed
                },
                preserved_methods=[
                    # Preserve methods if needed
                ]
            )
            
            # Step 3: Verify optimized model
            print(f"    Verifying optimized model...")
            with torch.no_grad():
                optimized_output = optimized_model(example_input)
                
                if isinstance(original_output, tuple):
                    optimized_diff = torch.max(torch.abs(original_output[0] - optimized_output[0])).item()
                else:
                    optimized_diff = torch.max(torch.abs(original_output - optimized_output)).item()
                
                if optimized_diff > 1e-4:
                    print(f"    Warning: Optimized model output differs by {optimized_diff}")
                else:
                    print(f"    ✓ Mobile optimization successful, output difference: {optimized_diff:.2e}")
            
            # Step 4: Save the optimized model
            mobile_model_path = self.saved_models_dir / f"{model_name}_mobile.pt"
            optimized_model.save(str(mobile_model_path))
            print(f"    ✓ Mobile model saved to: {mobile_model_path}")
            
            # Step 5: Calculate and compare sizes
            original_size = self.get_model_size_mb(model)
            optimized_size = self.get_model_size_mb(optimized_model)
            
            print(f"    Original size: {original_size:.2f} MB")
            print(f"    Optimized size: {optimized_size:.2f} MB")
            if original_size > 0:
                compression_ratio = original_size / optimized_size if optimized_size > 0 else float('inf')
                print(f"    Compression ratio: {compression_ratio:.2f}x")
            
            return optimized_model
            
        except Exception as e:
            print(f"    Error optimizing {model_name} for mobile: {e}")
            return None
    
    def quantize_model(self, model, model_name, quantization_type='int8'):
        """Quantize model to INT8."""
        print(f"  Attempting {quantization_type.upper()} quantization...")
        
        try:
            # Get model config
            config = self.get_model_config(model_name)
            input_size = config['input_size']
            
            # Try dynamic quantization (simpler and more reliable)
            print(f"    Using dynamic quantization...")
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            
            # Test the quantized model
            example_input = torch.randn(1, *input_size)
            with torch.no_grad():
                original_output = model(example_input)
                quantized_output = quantized_model(example_input)
                
                if isinstance(original_output, tuple):
                    output_diff = torch.max(torch.abs(original_output[0] - quantized_output[0])).item()
                else:
                    output_diff = torch.max(torch.abs(original_output - quantized_output)).item()
                
                print(f"    ✓ Quantization successful, output difference: {output_diff:.2e}")
            
            # Save quantized model
            quantized_model_path = self.saved_models_dir / f"{model_name}_quantized.pt"
            torch.save(quantized_model.state_dict(), quantized_model_path)
            print(f"    ✓ Quantized model saved to: {quantized_model_path}")
            
            return quantized_model
            
        except Exception as e:
            print(f"    Error quantizing {model_name}: {e}")
            return None
            
    def benchmark_model(self, model, model_name, variant_name, num_runs=50):
        """Benchmark model performance with improved timing."""
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
                'size_mb': self.get_model_size_mb(model),
                'inference_times_ms': [],
                'memory_usage_mb': []
            }
            
            # Generate sample input
            input_tensor = torch.randn(1, *input_size)
            
            # Extended warmup to ensure stable timing
            print(f"      Warming up...")
            with torch.no_grad():
                for _ in range(10):
                    try:
                        _ = model(input_tensor)
                    except Exception as e:
                        print(f"    Warmup failed for {variant_name}: {e}")
                        return None
            
            # Benchmark with more precise timing
            print(f"      Running {num_runs} benchmark iterations...")
            successful_runs = 0
            with torch.no_grad():
                for i in range(num_runs):
                    try:
                        # Use more precise timing
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        start_time = time.perf_counter()
                        
                        _ = model(input_tensor)
                        
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        end_time = time.perf_counter()
                        
                        inference_time = (end_time - start_time) * 1000  # Convert to ms
                        
                        # Only record reasonable timing values
                        if inference_time > 0 and inference_time < 10000:  # Between 0 and 10 seconds
                            results['inference_times_ms'].append(inference_time)
                            successful_runs += 1
                        
                    except Exception as e:
                        print(f"    Benchmark iteration {i} failed for {variant_name}: {e}")
                        continue
                        
            if successful_runs < 5:
                print(f"    Insufficient successful benchmark runs for {variant_name} ({successful_runs}/{num_runs})")
                return None
                
            # Calculate statistics
            results['successful_runs'] = successful_runs
            results['avg_inference_time_ms'] = np.mean(results['inference_times_ms'])
            results['std_inference_time_ms'] = np.std(results['inference_times_ms'])
            results['min_inference_time_ms'] = np.min(results['inference_times_ms'])
            results['max_inference_time_ms'] = np.max(results['inference_times_ms'])
            results['median_inference_time_ms'] = np.median(results['inference_times_ms'])
            results['throughput_fps'] = 1000 / results['avg_inference_time_ms']
            
            print(f"      ✓ {successful_runs}/{num_runs} successful runs")
            print(f"      ✓ Avg inference: {results['avg_inference_time_ms']:.2f} ms")
            print(f"      ✓ Throughput: {results['throughput_fps']:.2f} FPS")
            print(f"      ✓ Model size: {results['size_mb']:.2f} MB")
            
            # Remove raw data to save space
            del results['inference_times_ms']
            
            return results
        except Exception as e:
            print(f"    Error benchmarking {variant_name}: {e}")
            return None
        
    def optimize_all_models(self):
        """Optimize all loaded models."""
        print("\nStarting comprehensive model optimization...")
        
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
            
            # Mobile Optimization (prioritize this as it's most important for Pi Zero)
            print(f"  PyTorch Mobile optimization:")
            model_mobile = self.optimize_for_mobile(model, model_name)
            if model_mobile is not None:
                mobile_results = self.benchmark_model(model_mobile, model_name, f"{model_name}_mobile")
                if mobile_results:
                    self.results[f"{model_name}_mobile"] = mobile_results
            
            # INT8 Quantization
            print(f"  INT8 quantization:")
            model_int8 = self.quantize_model(model, model_name, 'int8')
            if model_int8 is not None:
                int8_results = self.benchmark_model(model_int8, model_name, f"{model_name}_int8")
                if int8_results:
                    self.results[f"{model_name}_int8"] = int8_results
                    
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
                print(f"  Successful runs: {results.get('successful_runs', 'N/A')}")
        else:
            print("No results to display.")
            
def main():
    print("Fixed PyTorch Model Optimizer for Raspberry Pi Zero")
    print("=" * 60)
    print("Includes proper mobile optimization verification")
    
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
    print(f"\nOptimized models saved to: {optimizer.saved_models_dir}")
    print("Check the optimization_results/ directory for detailed results.")

if __name__ == "__main__":
    main() 