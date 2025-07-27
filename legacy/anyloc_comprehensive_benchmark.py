#!/usr/bin/env python3
"""
AnyLoc-Inspired Comprehensive Model Benchmark
Includes DINOv2 ViT-S/14, ViT-B/14, ViT-L/14 models with detailed optimization logging
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import SuperPoint
try:
    from simple_superpoint import SuperPointNet
    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False

class AnyLocBenchmark:
    def __init__(self):
        self.device = 'cpu'
        self.results = {}
        self.optimization_summary = {}
        
    def print_header(self, title):
        """Print formatted header"""
        print(f"\n{'='*80}")
        print(f"🎯 {title}")
        print(f"{'='*80}")
        
    def print_model_header(self, model_name):
        """Print model-specific header"""
        print(f"\n{'─'*60}")
        print(f"🔬 BENCHMARKING: {model_name}")
        print(f"{'─'*60}")
        
    def print_optimization_benefits(self, model_name, original_metrics, optimized_metrics, optimization_type):
        """Print detailed optimization benefits"""
        print(f"\n🚀 {optimization_type.upper()} OPTIMIZATION BENEFITS:")
        print(f"   Model: {model_name}")
        
        # Calculate improvements
        fps_improvement = (optimized_metrics['fps'] - original_metrics['fps']) / original_metrics['fps'] * 100
        time_improvement = (original_metrics['inference_time'] - optimized_metrics['inference_time']) / original_metrics['inference_time'] * 100
        
        print(f"   📊 Performance Gains:")
        print(f"      • FPS: {original_metrics['fps']:.1f} → {optimized_metrics['fps']:.1f} ({fps_improvement:+.1f}%)")
        print(f"      • Inference: {original_metrics['inference_time']:.1f}ms → {optimized_metrics['inference_time']:.1f}ms ({time_improvement:+.1f}%)")
        
        # Memory comparison
        if 'memory_mb' in original_metrics and 'memory_mb' in optimized_metrics:
            if original_metrics['memory_mb'] > 0 and optimized_metrics['memory_mb'] > 0:
                memory_reduction = (original_metrics['memory_mb'] - optimized_metrics['memory_mb']) / original_metrics['memory_mb'] * 100
                print(f"      • Memory: {original_metrics['memory_mb']:.1f}MB → {optimized_metrics['memory_mb']:.1f}MB ({memory_reduction:+.1f}%)")
        
        # Pi Zero suitability
        pi_zero_suitable = optimized_metrics['memory_mb'] < 100 and optimized_metrics['fps'] > 10
        print(f"   🥧 Pi Zero Suitability: {'✅ EXCELLENT' if pi_zero_suitable else '⚠️ MARGINAL' if optimized_metrics['memory_mb'] < 200 else '❌ TOO LARGE'}")
        
        if optimization_type == "mobile":
            print(f"   ⚡ x86 Performance: Slower (expected), ARM Performance: Faster (target)")
        elif optimization_type == "quantized":
            print(f"   ⚡ Quantization: {'✅ SUCCESS' if fps_improvement > 0 else '❌ FAILED'}")
    
    def get_model_size_mb(self, model):
        """Calculate model size in MB"""
        if hasattr(model, 'save'):
            # For traced models, we need to save and measure
            temp_path = "temp_model.pt"
            try:
                torch.jit.save(model, temp_path)
                size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                os.remove(temp_path)
                return size_mb
            except:
                pass
        
        # Fallback: calculate from parameters
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size / (1024 * 1024)
    
    def benchmark_model(self, model, model_name, input_tensor, num_runs=100):
        """Benchmark a single model"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        memory_mb = self.get_model_size_mb(model)
        
        return {
            'inference_time': avg_time * 1000,  # ms
            'fps': fps,
            'memory_mb': memory_mb,
            'std_time': std_time * 1000
        }
    
    def create_dinov2_model(self, variant):
        """Create DINOv2 model"""
        try:
            model = torch.hub.load('facebookresearch/dinov2', variant)
            return model
        except Exception as e:
            print(f"❌ Failed to load {variant}: {e}")
            return None
    
    def create_superpoint_model(self):
        """Create SuperPoint model"""
        if not SUPERPOINT_AVAILABLE:
            return None
        try:
            model = SuperPointNet()
            return model
        except Exception as e:
            print(f"❌ Failed to load SuperPoint: {e}")
            return None
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark on all models"""
        self.print_header("AnyLoc-Inspired Comprehensive Model Benchmark")
        print("🔍 Testing DINOv2 models from AnyLoc paper + optimization techniques")
        
        # Model configurations
        models_config = [
            # SuperPoint (keypoint detection)
            {
                'name': 'SuperPoint',
                'create_fn': self.create_superpoint_model,
                'input_size': (1, 224, 224),
                'description': 'Keypoint detection model'
            },
            # Standard CNN models
            {
                'name': 'MobileNetV2',
                'create_fn': lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
                'input_size': (3, 224, 224),
                'description': 'Lightweight CNN'
            },
            {
                'name': 'EfficientNet-B0',
                'create_fn': lambda: torch.hub.load('pytorch/vision:v0.10.0', 'efficientnet_b0', pretrained=True),
                'input_size': (3, 224, 224),
                'description': 'Efficient CNN'
            },
            # AnyLoc DINOv2 models (ViT-S/14: 21M params ≃ 84 MB)
            {
                'name': 'DINOv2-ViT-S/14',
                'create_fn': lambda: self.create_dinov2_model('dinov2_vits14'),
                'input_size': (3, 224, 224),
                'description': 'DINOv2 Small (21M params, ~84MB)'
            },
            {
                'name': 'DINOv2-ViT-B/14',
                'create_fn': lambda: self.create_dinov2_model('dinov2_vitb14'),
                'input_size': (3, 224, 224),
                'description': 'DINOv2 Base (86M params, ~344MB)'
            },
            {
                'name': 'DINOv2-ViT-L/14',
                'create_fn': lambda: self.create_dinov2_model('dinov2_vitl14'),
                'input_size': (3, 224, 224),
                'description': 'DINOv2 Large (300M params, ~1.2GB)'
            },
        ]
        
        # Results storage
        all_results = {}
        
        for model_config in models_config:
            model_name = model_config['name']
            self.print_model_header(f"{model_name} - {model_config['description']}")
            
            # Create model
            model = model_config['create_fn']()
            if model is None:
                print(f"❌ Skipping {model_name} - model creation failed")
                continue
                
            model = model.to(self.device).eval()
            
            # Create input tensor
            input_tensor = torch.randn(1, *model_config['input_size']).to(self.device)
            
            # Benchmark original model
            print("📊 Benchmarking ORIGINAL model...")
            original_metrics = self.benchmark_model(model, model_name, input_tensor)
            
            print(f"   ✅ Original Performance:")
            print(f"      • FPS: {original_metrics['fps']:.1f}")
            print(f"      • Inference: {original_metrics['inference_time']:.1f}ms")
            print(f"      • Memory: {original_metrics['memory_mb']:.1f}MB")
            
            # Store results
            all_results[model_name] = {
                'original': original_metrics,
                'description': model_config['description']
            }
            
            # Try quantization
            print("\n🔧 Attempting INT8 Quantization...")
            try:
                # Prepare for quantization
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                model_prepared = torch.quantization.prepare(model, inplace=False)
                
                # Calibrate (simple forward pass)
                with torch.no_grad():
                    for _ in range(10):
                        _ = model_prepared(input_tensor)
                
                # Convert to quantized model
                model_quantized = torch.quantization.convert(model_prepared, inplace=False)
                
                # Benchmark quantized model
                quantized_metrics = self.benchmark_model(model_quantized, f"{model_name}_quantized", input_tensor)
                all_results[model_name]['quantized'] = quantized_metrics
                
                self.print_optimization_benefits(model_name, original_metrics, quantized_metrics, "quantized")
                
            except Exception as e:
                print(f"   ❌ Quantization failed: {e}")
                all_results[model_name]['quantized'] = None
            
            # Try mobile optimization
            print("\n🔧 Attempting Mobile Optimization...")
            try:
                # Trace the model
                model_traced = torch.jit.trace(model, input_tensor)
                
                # Optimize for mobile
                model_mobile = optimize_for_mobile(model_traced)
                
                # Benchmark mobile model
                mobile_metrics = self.benchmark_model(model_mobile, f"{model_name}_mobile", input_tensor)
                all_results[model_name]['mobile'] = mobile_metrics
                
                self.print_optimization_benefits(model_name, original_metrics, mobile_metrics, "mobile")
                
            except Exception as e:
                print(f"   ❌ Mobile optimization failed: {e}")
                all_results[model_name]['mobile'] = None
        
        # Print comprehensive summary
        self.print_summary(all_results)
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def print_summary(self, results):
        """Print comprehensive summary"""
        self.print_header("📋 COMPREHENSIVE BENCHMARK SUMMARY")
        
        print("🏆 TOP PERFORMERS BY CATEGORY:")
        
        # Best FPS
        best_fps = max(results.items(), key=lambda x: x[1]['original']['fps'])
        print(f"   🚀 Fastest FPS: {best_fps[0]} ({best_fps[1]['original']['fps']:.1f} FPS)")
        
        # Smallest memory
        smallest_memory = min(results.items(), key=lambda x: x[1]['original']['memory_mb'])
        print(f"   🗜️  Smallest Memory: {smallest_memory[0]} ({smallest_memory[1]['original']['memory_mb']:.1f}MB)")
        
        # Best Pi Zero candidates
        print(f"\n🥧 RASPBERRY PI ZERO RECOMMENDATIONS:")
        pi_candidates = []
        for name, data in results.items():
            metrics = data['original']
            if metrics['memory_mb'] < 100 and metrics['fps'] > 10:
                pi_candidates.append((name, metrics))
        
        if pi_candidates:
            pi_candidates.sort(key=lambda x: x[1]['fps'], reverse=True)
            for i, (name, metrics) in enumerate(pi_candidates[:3]):
                print(f"   {i+1}. {name}: {metrics['fps']:.1f} FPS, {metrics['memory_mb']:.1f}MB")
        else:
            print("   ⚠️  No models meet Pi Zero criteria (>10 FPS, <100MB)")
        
        # DINOv2 specific analysis
        print(f"\n🔬 DINOV2 ANYLOC MODELS ANALYSIS:")
        dinov2_models = {k: v for k, v in results.items() if 'DINOv2' in k}
        
        if dinov2_models:
            print("   Model Size vs Performance Trade-off:")
            for name, data in dinov2_models.items():
                metrics = data['original']
                efficiency = metrics['fps'] / metrics['memory_mb'] * 1000  # FPS per MB * 1000
                print(f"   • {name}: {metrics['fps']:.1f} FPS, {metrics['memory_mb']:.1f}MB (Efficiency: {efficiency:.1f})")
        
        # Optimization effectiveness
        print(f"\n⚡ OPTIMIZATION EFFECTIVENESS:")
        for name, data in results.items():
            if data.get('quantized') and data.get('mobile'):
                orig_fps = data['original']['fps']
                quant_fps = data['quantized']['fps']
                mobile_fps = data['mobile']['fps']
                
                quant_improvement = (quant_fps - orig_fps) / orig_fps * 100
                mobile_change = (mobile_fps - orig_fps) / orig_fps * 100
                
                print(f"   • {name}:")
                print(f"     - Quantization: {quant_improvement:+.1f}% FPS")
                print(f"     - Mobile: {mobile_change:+.1f}% FPS (ARM will be better)")
    
    def save_results(self, results):
        """Save results to files"""
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON
        json_path = output_dir / "anyloc_benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        csv_path = output_dir / "anyloc_benchmark_summary.csv"
        with open(csv_path, 'w') as f:
            f.write("Model,FPS,Inference_Time_ms,Memory_MB,Quantized_FPS,Mobile_FPS,Description\n")
            for name, data in results.items():
                orig = data['original']
                quant_fps = data.get('quantized', {}).get('fps', 'N/A')
                mobile_fps = data.get('mobile', {}).get('fps', 'N/A')
                f.write(f"{name},{orig['fps']:.1f},{orig['inference_time']:.1f},{orig['memory_mb']:.1f},{quant_fps},{mobile_fps},{data['description']}\n")
        
        print(f"\n💾 Results saved to:")
        print(f"   • {json_path}")
        print(f"   • {csv_path}")

def main():
    """Main execution function"""
    print("🚀 Starting AnyLoc-Inspired Comprehensive Benchmark")
    print("📝 This benchmark tests DINOv2 models from the AnyLoc paper")
    print("🔧 Including optimization techniques: Quantization & Mobile optimization")
    
    benchmark = AnyLocBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n🎉 Benchmark Complete!")
    print("📊 Check optimization_results/ directory for detailed results")
    
    return results

if __name__ == "__main__":
    main() 