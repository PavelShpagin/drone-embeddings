#!/usr/bin/env python3
"""
Fixed AnyLoc DINOv2 Benchmark - Focus on DINOv2 Models
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
import warnings
warnings.filterwarnings('ignore')

# Try to import SuperPoint
try:
    from simple_superpoint import SuperPointNet
    SUPERPOINT_AVAILABLE = True
except ImportError:
    SUPERPOINT_AVAILABLE = False

class AnyLocDINOv2Benchmark:
    def __init__(self):
        self.device = 'cpu'
        
    def print_header(self, title):
        print(f"\n{'='*80}")
        print(f"🎯 {title}")
        print(f"{'='*80}")
        
    def print_model_header(self, model_name, description):
        print(f"\n{'─'*60}")
        print(f"🔬 BENCHMARKING: {model_name}")
        print(f"📋 Description: {description}")
        print(f"{'─'*60}")
        
    def print_optimization_benefits(self, model_name, original_metrics, optimized_metrics, optimization_type):
        print(f"\n🚀 {optimization_type.upper()} OPTIMIZATION BENEFITS:")
        print(f"   Model: {model_name}")
        
        fps_improvement = (optimized_metrics['fps'] - original_metrics['fps']) / original_metrics['fps'] * 100
        time_improvement = (original_metrics['inference_time'] - optimized_metrics['inference_time']) / original_metrics['inference_time'] * 100
        memory_reduction = (original_metrics['memory_mb'] - optimized_metrics['memory_mb']) / original_metrics['memory_mb'] * 100
        
        print(f"   📊 Performance Gains:")
        print(f"      • FPS: {original_metrics['fps']:.1f} → {optimized_metrics['fps']:.1f} ({fps_improvement:+.1f}%)")
        print(f"      • Inference: {original_metrics['inference_time']:.1f}ms → {optimized_metrics['inference_time']:.1f}ms ({time_improvement:+.1f}%)")
        print(f"      • Memory: {original_metrics['memory_mb']:.1f}MB → {optimized_metrics['memory_mb']:.1f}MB ({memory_reduction:+.1f}%)")
        
        # Pi Zero suitability
        pi_zero_excellent = optimized_metrics['memory_mb'] < 100 and optimized_metrics['fps'] > 10
        pi_zero_marginal = optimized_metrics['memory_mb'] < 200 and optimized_metrics['fps'] > 5
        
        if pi_zero_excellent:
            suitability = "✅ EXCELLENT"
        elif pi_zero_marginal:
            suitability = "⚠️ MARGINAL"
        else:
            suitability = "❌ TOO LARGE"
            
        print(f"   🥧 Pi Zero Suitability: {suitability}")
        
        if optimization_type == "mobile":
            print(f"   ⚡ x86 Performance: Slower (expected), ARM Performance: Faster (target)")
        
    def get_model_size_mb(self, model):
        """Calculate model size in MB"""
        if hasattr(model, 'save'):
            temp_path = "temp_model.pt"
            try:
                torch.jit.save(model, temp_path)
                size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                os.remove(temp_path)
                return size_mb
            except:
                pass
        
        # Fallback
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size / (1024 * 1024)
    
    def benchmark_model(self, model, model_name, input_tensor, num_runs=50):
        """Benchmark a single model"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
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
        fps = 1.0 / avg_time
        memory_mb = self.get_model_size_mb(model)
        
        return {
            'inference_time': avg_time * 1000,  # ms
            'fps': fps,
            'memory_mb': memory_mb,
        }
    
    def create_dinov2_model(self, variant):
        """Create DINOv2 model"""
        try:
            print(f"   🔄 Loading {variant} from torch hub...")
            model = torch.hub.load('facebookresearch/dinov2', variant, pretrained=True)
            print(f"   ✅ Successfully loaded {variant}")
            return model
        except Exception as e:
            print(f"   ❌ Failed to load {variant}: {e}")
            return None
    
    def run_dinov2_benchmark(self):
        """Run benchmark focused on DINOv2 models"""
        self.print_header("AnyLoc DINOv2 Model Benchmark")
        print("🔍 Testing DINOv2 models from AnyLoc paper with optimization benefits logging")
        print("📝 Focus: ViT-S/14 (21M params ≃ 84MB), ViT-B/14, ViT-L/14")
        
        # DINOv2 Model configurations
        models_config = [
            {
                'name': 'DINOv2-ViT-S/14',
                'variant': 'dinov2_vits14',
                'description': 'DINOv2 Small (21M params, ~84MB) - AnyLoc paper'
            },
            {
                'name': 'DINOv2-ViT-B/14',
                'variant': 'dinov2_vitb14',
                'description': 'DINOv2 Base (86M params, ~344MB) - AnyLoc paper'
            },
            {
                'name': 'DINOv2-ViT-L/14',
                'variant': 'dinov2_vitl14',
                'description': 'DINOv2 Large (300M params, ~1.2GB) - AnyLoc paper'
            },
            # Add some comparison models
            {
                'name': 'SuperPoint',
                'variant': 'superpoint',
                'description': 'Keypoint detection model (comparison)'
            },
            {
                'name': 'MobileNetV2',
                'variant': 'mobilenetv2',
                'description': 'Lightweight CNN (comparison)'
            }
        ]
        
        all_results = {}
        
        for model_config in models_config:
            model_name = model_config['name']
            variant = model_config['variant']
            description = model_config['description']
            
            self.print_model_header(model_name, description)
            
            # Create model
            if variant == 'superpoint':
                if SUPERPOINT_AVAILABLE:
                    model = SuperPointNet()
                    input_tensor = torch.randn(1, 1, 224, 224).to(self.device)
                else:
                    print("   ❌ SuperPoint not available - skipping")
                    continue
            elif variant == 'mobilenetv2':
                model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                input_tensor = torch.randn(1, 3, 224, 224).to(self.device)
            else:
                model = self.create_dinov2_model(variant)
                input_tensor = torch.randn(1, 3, 224, 224).to(self.device)
            
            if model is None:
                print(f"   ❌ Skipping {model_name} - model creation failed")
                continue
                
            model = model.to(self.device).eval()
            
            # Benchmark original model
            print("   📊 Benchmarking ORIGINAL model...")
            original_metrics = self.benchmark_model(model, model_name, input_tensor)
            
            print(f"   ✅ Original Performance:")
            print(f"      • FPS: {original_metrics['fps']:.1f}")
            print(f"      • Inference: {original_metrics['inference_time']:.1f}ms")
            print(f"      • Memory: {original_metrics['memory_mb']:.1f}MB")
            
            # Store results
            all_results[model_name] = {
                'original': original_metrics,
                'description': description,
                'variant': variant
            }
            
            # Try mobile optimization
            print("\n   🔧 Attempting Mobile Optimization...")
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
        self.print_header("📋 ANYLOC DINOV2 BENCHMARK SUMMARY")
        
        print("🏆 ANYLOC DINOV2 MODELS PERFORMANCE:")
        
        # DINOv2 specific analysis
        dinov2_models = {k: v for k, v in results.items() if 'DINOv2' in k}
        
        if dinov2_models:
            print("\n🔬 DINOv2 Models Analysis (from AnyLoc paper):")
            for name, data in dinov2_models.items():
                metrics = data['original']
                mobile_metrics = data.get('mobile', {})
                
                print(f"\n   • {name}:")
                print(f"     - Original: {metrics['fps']:.1f} FPS, {metrics['memory_mb']:.1f}MB")
                if mobile_metrics:
                    print(f"     - Mobile: {mobile_metrics['fps']:.1f} FPS, {mobile_metrics['memory_mb']:.1f}MB")
                    
                # Calculate efficiency
                efficiency = metrics['fps'] / metrics['memory_mb'] * 100
                print(f"     - Efficiency: {efficiency:.2f} FPS/MB")
                
                # Pi Zero assessment
                pi_zero_ok = metrics['memory_mb'] < 500  # More lenient for DINOv2
                print(f"     - Pi Zero: {'✅ Possible' if pi_zero_ok else '❌ Too large'}")
        
        # Best performers
        print(f"\n🥇 TOP PERFORMERS:")
        if results:
            best_fps = max(results.items(), key=lambda x: x[1]['original']['fps'])
            smallest = min(results.items(), key=lambda x: x[1]['original']['memory_mb'])
            
            print(f"   🚀 Fastest: {best_fps[0]} ({best_fps[1]['original']['fps']:.1f} FPS)")
            print(f"   🗜️  Smallest: {smallest[0]} ({smallest[1]['original']['memory_mb']:.1f}MB)")
        
        # Pi Zero recommendations
        print(f"\n🥧 RASPBERRY PI ZERO RECOMMENDATIONS:")
        pi_candidates = []
        for name, data in results.items():
            metrics = data['original']
            if metrics['memory_mb'] < 100:
                pi_candidates.append((name, metrics))
        
        if pi_candidates:
            pi_candidates.sort(key=lambda x: x[1]['fps'], reverse=True)
            for i, (name, metrics) in enumerate(pi_candidates):
                print(f"   {i+1}. {name}: {metrics['fps']:.1f} FPS, {metrics['memory_mb']:.1f}MB")
        else:
            print("   ⚠️  DINOv2 models are large for Pi Zero (>100MB)")
            print("   💡 Consider using smaller models or quantization")
    
    def save_results(self, results):
        """Save results to files"""
        output_dir = Path("optimization_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON
        json_path = output_dir / "dinov2_benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        csv_path = output_dir / "dinov2_benchmark_summary.csv"
        with open(csv_path, 'w') as f:
            f.write("Model,FPS,Inference_Time_ms,Memory_MB,Mobile_FPS,Mobile_Memory_MB,Description\n")
            for name, data in results.items():
                orig = data['original']
                mobile = data.get('mobile', {})
                mobile_fps = mobile.get('fps', 'N/A')
                mobile_mb = mobile.get('memory_mb', 'N/A')
                f.write(f"{name},{orig['fps']:.1f},{orig['inference_time']:.1f},{orig['memory_mb']:.1f},{mobile_fps},{mobile_mb},{data['description']}\n")
        
        print(f"\n💾 Results saved to:")
        print(f"   • {json_path}")
        print(f"   • {csv_path}")

def main():
    print("🚀 Starting AnyLoc DINOv2 Benchmark")
    print("📝 Testing DINOv2 ViT-S/14, ViT-B/14, ViT-L/14 from AnyLoc paper")
    print("🔧 With detailed optimization benefits logging")
    
    benchmark = AnyLocDINOv2Benchmark()
    results = benchmark.run_dinov2_benchmark()
    
    print("\n🎉 DINOv2 Benchmark Complete!")
    print("📊 Check optimization_results/ directory for detailed results")
    
    return results

if __name__ == "__main__":
    main() 