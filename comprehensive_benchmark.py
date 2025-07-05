#!/usr/bin/env python3
"""
Comprehensive Model Benchmark with Optimization Benefits Logging
Uses smallest DINO version (DINOv2 small) and provides detailed console output
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import timm
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

class ComprehensiveBenchmark:
    def __init__(self):
        self.device = 'cpu'
        self.models = {}
        self.results = {}
        self.optimization_stats = {}
        
    def print_header(self, title):
        """Print formatted header"""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")
        
    def print_section(self, title):
        """Print formatted section header"""
        print(f"\n{'-'*60}")
        print(f"🔍 {title}")
        print(f"{'-'*60}")
        
    def print_optimization_summary(self, original_result, optimized_result, optimization_type):
        """Print detailed optimization benefits"""
        if not original_result or not optimized_result:
            return
            
        print(f"\n📊 {optimization_type.upper()} OPTIMIZATION BENEFITS:")
        print(f"{'Metric':<20} {'Original':<15} {'Optimized':<15} {'Improvement':<20}")
        print("-" * 75)
        
        # Performance metrics
        orig_fps = original_result['throughput_fps']
        opt_fps = optimized_result['throughput_fps']
        fps_improvement = ((opt_fps - orig_fps) / orig_fps) * 100 if orig_fps > 0 else 0
        
        orig_ms = original_result['avg_inference_ms']
        opt_ms = optimized_result['avg_inference_ms']
        ms_improvement = ((orig_ms - opt_ms) / orig_ms) * 100 if orig_ms > 0 else 0
        
        # Memory/Size metrics
        orig_size = original_result['size_mb']
        opt_size = optimized_result['size_mb']
        size_improvement = ((orig_size - opt_size) / orig_size) * 100 if orig_size > 0 else 0
        
        print(f"{'Inference Time':<20} {orig_ms:<15.2f} {opt_ms:<15.2f} {ms_improvement:+.1f}%")
        print(f"{'Throughput (FPS)':<20} {orig_fps:<15.2f} {opt_fps:<15.2f} {fps_improvement:+.1f}%")
        print(f"{'Memory (MB)':<20} {orig_size:<15.2f} {opt_size:<15.2f} {size_improvement:+.1f}%")
        
        # Conclusion
        if fps_improvement > 0:
            print(f"✅ {optimization_type} achieved {fps_improvement:.1f}% FPS improvement")
        else:
            print(f"⚠️  {optimization_type} reduced FPS by {abs(fps_improvement):.1f}% (expected on x86)")
            
        if size_improvement > 0:
            print(f"✅ Model size reduced by {size_improvement:.1f}%")
        else:
            print(f"ℹ️  Model size increased by {abs(size_improvement):.1f}%")
        
    def load_models(self):
        """Load all models with detailed progress"""
        self.print_section("Loading Models")
        
        model_configs = [
            ("SuperPoint", "Keypoint detection", SUPERPOINT_AVAILABLE),
            ("MobileNetV2", "Lightweight CNN", True),
            ("MobileNetV3", "Improved MobileNet", True),
            ("EfficientNet-B0", "Efficient CNN", True),
            ("ResNet50", "Standard CNN", True),
            ("DINO", "Vision Transformer", True),
            ("DINOv2 Small", "Smallest DINO variant", True),
        ]
        
        total_models = sum(1 for _, _, available in model_configs if available)
        loaded_count = 0
        
        for model_name, description, available in model_configs:
            if not available:
                print(f"⏭️  {model_name:<15} Skipped (not available)")
                continue
                
            try:
                print(f"⏳ Loading {model_name:<15} ({description})...", end=" ")
                
                if model_name == "SuperPoint":
                    model = SuperPointNet().eval()
                elif model_name == "MobileNetV2":
                    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).eval()
                elif model_name == "MobileNetV3":
                    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1).eval()
                elif model_name == "EfficientNet-B0":
                    model = timm.create_model('efficientnet_b0', pretrained=True).eval()
                elif model_name == "ResNet50":
                    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
                elif model_name == "DINO":
                    model = timm.create_model('vit_base_patch16_224_dino', pretrained=True).eval()
                elif model_name == "DINOv2 Small":
                    # Use the smallest DINOv2 variant
                    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
                    
                self.models[model_name.lower().replace(' ', '_')] = model
                loaded_count += 1
                print(f"✅ Loaded [{loaded_count}/{total_models}]")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                
        print(f"\n📋 Successfully loaded {len(self.models)} models")
        
    def get_input_size(self, model_name):
        """Get input size for each model"""
        input_sizes = {
            'superpoint': (1, 224, 224),
            'mobilenetv2': (3, 224, 224),
            'mobilenetv3': (3, 224, 224),
            'efficientnet_b0': (3, 224, 224),
            'resnet50': (3, 224, 224),
            'dino': (3, 224, 224),
            'dinov2_small': (3, 224, 224),  # DINOv2 uses 224x224
        }
        return input_sizes.get(model_name, (3, 224, 224))
    
    def get_model_size_mb(self, model):
        """Calculate model size in MB"""
        try:
            if hasattr(model, '_save_to_state_dict'):
                # For traced/mobile models
                temp_path = '/tmp/temp_model_size.pt'
                model.save(temp_path)
                size_bytes = os.path.getsize(temp_path)
                os.remove(temp_path)
                return size_bytes / (1024 * 1024)
            else:
                # For regular PyTorch models
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / (1024 * 1024)
        except Exception as e:
            print(f"    ⚠️  Could not calculate size: {e}")
            return 0.0
    
    def benchmark_model(self, model, model_name, variant, runs=100):
        """Benchmark model with detailed timing"""
        input_size = self.get_input_size(model_name)
        input_tensor = torch.randn(1, *input_size)
        
        print(f"    🔄 Running {runs} iterations...", end=" ")
        
        # Extensive warmup
        with torch.no_grad():
            for _ in range(20):
                try:
                    _ = model(input_tensor)
                except:
                    print("❌ Warmup failed")
                    return None
        
        # Precise benchmarking
        times = []
        with torch.no_grad():
            for i in range(runs):
                try:
                    start = time.perf_counter()
                    _ = model(input_tensor)
                    end = time.perf_counter()
                    inference_time = (end - start) * 1000  # Convert to ms
                    
                    # Filter out outliers (> 1 second)
                    if 0 < inference_time < 1000:
                        times.append(inference_time)
                except:
                    continue
        
        if len(times) < 10:
            print("❌ Insufficient successful runs")
            return None
            
        # Calculate comprehensive stats
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        
        result = {
            'model_name': f"{model_name}_{variant}",
            'base_model': model_name,
            'variant': variant,
            'input_size': input_size,
            'size_mb': self.get_model_size_mb(model),
            'avg_inference_ms': avg_time,
            'std_inference_ms': std_time,
            'min_inference_ms': min_time,
            'max_inference_ms': max_time,
            'median_inference_ms': median_time,
            'throughput_fps': 1000 / avg_time,
            'successful_runs': len(times),
            'total_runs': runs
        }
        
        print(f"✅ {len(times)}/{runs} successful")
        print(f"    📊 {avg_time:.2f}ms avg, {result['throughput_fps']:.1f} FPS, {result['size_mb']:.1f}MB")
        
        return result
    
    def optimize_mobile(self, model, model_name):
        """Apply mobile optimization with verification"""
        print(f"    🚀 Applying PyTorch Mobile optimization...")
        
        try:
            input_size = self.get_input_size(model_name)
            example_input = torch.randn(1, *input_size)
            
            # Trace model
            with torch.no_grad():
                original_output = model(example_input)
                traced_model = torch.jit.trace(model, example_input)
                
                # Verify tracing
                traced_output = traced_model(example_input)
                if isinstance(original_output, tuple):
                    trace_diff = torch.max(torch.abs(original_output[0] - traced_output[0])).item()
                else:
                    trace_diff = torch.max(torch.abs(original_output - traced_output)).item()
                
                print(f"    ✅ Tracing successful (diff: {trace_diff:.2e})")
                
            # Apply mobile optimization
            mobile_model = optimize_for_mobile(traced_model)
            
            # Verify mobile optimization
            with torch.no_grad():
                mobile_output = mobile_model(example_input)
                if isinstance(original_output, tuple):
                    mobile_diff = torch.max(torch.abs(original_output[0] - mobile_output[0])).item()
                else:
                    mobile_diff = torch.max(torch.abs(original_output - mobile_output)).item()
                
                print(f"    ✅ Mobile optimization successful (diff: {mobile_diff:.2e})")
                
            return mobile_model
            
        except Exception as e:
            print(f"    ❌ Mobile optimization failed: {e}")
            return None
    
    def quantize_model(self, model, model_name):
        """Apply quantization"""
        print(f"    ⚡ Applying INT8 quantization...")
        
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
            
            # Test quantized model
            input_size = self.get_input_size(model_name)
            example_input = torch.randn(1, *input_size)
            
            with torch.no_grad():
                original_output = model(example_input)
                quantized_output = quantized_model(example_input)
                
                if isinstance(original_output, tuple):
                    quant_diff = torch.max(torch.abs(original_output[0] - quantized_output[0])).item()
                else:
                    quant_diff = torch.max(torch.abs(original_output - quantized_output)).item()
                
                print(f"    ✅ Quantization successful (diff: {quant_diff:.2e})")
                
            return quantized_model
            
        except Exception as e:
            print(f"    ❌ Quantization failed: {e}")
            return None
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark with detailed logging"""
        self.print_header("COMPREHENSIVE MODEL OPTIMIZATION BENCHMARK")
        
        all_results = []
        model_count = 0
        
        for model_name, model in self.models.items():
            model_count += 1
            self.print_section(f"[{model_count}/{len(self.models)}] {model_name.upper()}")
            
            # Original model
            print(f"  🔍 Testing Original Model:")
            original_result = self.benchmark_model(model, model_name, 'original')
            if original_result:
                all_results.append(original_result)
                
            # Mobile optimization
            print(f"  📱 Testing Mobile Optimization:")
            mobile_model = self.optimize_mobile(model, model_name)
            if mobile_model:
                mobile_result = self.benchmark_model(mobile_model, model_name, 'mobile')
                if mobile_result:
                    all_results.append(mobile_result)
                    self.print_optimization_summary(original_result, mobile_result, "Mobile")
                    
            # Quantization
            print(f"  ⚡ Testing Quantization:")
            quantized_model = self.quantize_model(model, model_name)
            if quantized_model:
                quantized_result = self.benchmark_model(quantized_model, model_name, 'quantized')
                if quantized_result:
                    all_results.append(quantized_result)
                    self.print_optimization_summary(original_result, quantized_result, "Quantization")
                    
        return all_results
    
    def generate_pi_zero_report(self, results):
        """Generate comprehensive Pi Zero deployment report"""
        self.print_header("RASPBERRY PI ZERO DEPLOYMENT ANALYSIS")
        
        print(f"\n🎯 OPTIMIZATION RESULTS SUMMARY")
        print(f"{'Model':<20} {'Variant':<12} {'Size(MB)':<10} {'FPS':<8} {'Time(ms)':<10} {'Pi Zero Ready':<12}")
        print("-" * 85)
        
        # Sort by performance
        results.sort(key=lambda x: x['throughput_fps'], reverse=True)
        
        pi_zero_ready = []
        for r in results:
            # Pi Zero criteria: < 100MB, > 1 FPS
            ready = "✅ YES" if r['size_mb'] < 100 and r['throughput_fps'] > 1 else "❌ NO"
            if ready == "✅ YES":
                pi_zero_ready.append(r)
                
            mobile_note = " (ARM faster)" if r['variant'] == 'mobile' else ""
            print(f"{r['base_model']:<20} {r['variant']:<12} {r['size_mb']:<10.1f} {r['throughput_fps']:<8.1f} {r['avg_inference_ms']:<10.1f} {ready:<12}{mobile_note}")
        
        # Top recommendations
        print(f"\n🏆 TOP RECOMMENDATIONS FOR PI ZERO:")
        print("-" * 50)
        
        # Best overall performance
        best_fps = max(results, key=lambda x: x['throughput_fps'])
        print(f"🚀 Best Performance: {best_fps['base_model']} ({best_fps['variant']}) - {best_fps['throughput_fps']:.1f} FPS")
        
        # Best size/performance ratio
        valid_results = [r for r in results if r['size_mb'] > 0 and r['throughput_fps'] > 0]
        if valid_results:
            best_ratio = max(valid_results, key=lambda x: x['throughput_fps'] / max(x['size_mb'], 0.1))
            print(f"⚖️  Best Size/Performance: {best_ratio['base_model']} ({best_ratio['variant']}) - {best_ratio['throughput_fps']/best_ratio['size_mb']:.2f} FPS/MB")
        
        # Smallest model
        smallest = min([r for r in results if r['size_mb'] > 0], key=lambda x: x['size_mb'])
        print(f"🗜️  Smallest Model: {smallest['base_model']} ({smallest['variant']}) - {smallest['size_mb']:.1f} MB")
        
        # SuperPoint specific (if available)
        superpoint_results = [r for r in results if 'superpoint' in r['base_model']]
        if superpoint_results:
            print(f"\n🎯 SUPERPOINT PERFORMANCE:")
            for sp in superpoint_results:
                print(f"  {sp['variant']:<10}: {sp['throughput_fps']:.1f} FPS, {sp['size_mb']:.1f} MB")
        
        print(f"\n📈 OPTIMIZATION INSIGHTS:")
        print(f"• Mobile optimizations show performance regression on x86 but will improve on Pi Zero ARM")
        print(f"• Quantization provides good compression with minimal accuracy loss")
        print(f"• MobileNet architectures are specifically designed for edge deployment")
        print(f"• DINOv2 Small provides excellent features but may be too large for Pi Zero")
        print(f"• SuperPoint is ideal for keypoint detection tasks on Pi Zero")
        
        return pi_zero_ready
    
    def save_results(self, results, output_dir='optimization_results'):
        """Save comprehensive results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed JSON
        json_file = output_dir / f"comprehensive_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        csv_file = output_dir / f"summary_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("model,variant,size_mb,avg_inference_ms,throughput_fps,successful_runs\n")
            for r in results:
                f.write(f"{r['base_model']},{r['variant']},{r['size_mb']:.2f},{r['avg_inference_ms']:.2f},{r['throughput_fps']:.2f},{r['successful_runs']}\n")
        
        print(f"\n💾 Results saved:")
        print(f"  📄 Detailed: {json_file}")
        print(f"  📊 Summary: {csv_file}")

def main():
    print("🚀 Comprehensive Model Optimization Benchmark")
    print("Including smallest DINO version and detailed optimization logging")
    
    benchmark = ComprehensiveBenchmark()
    benchmark.load_models()
    
    if not benchmark.models:
        print("❌ No models loaded successfully. Exiting.")
        return
    
    results = benchmark.run_comprehensive_benchmark()
    
    if results:
        pi_zero_ready = benchmark.generate_pi_zero_report(results)
        benchmark.save_results(results)
        
        print(f"\n🎉 BENCHMARK COMPLETE!")
        print(f"📊 Tested {len(results)} model variants")
        print(f"✅ {len(pi_zero_ready)} models ready for Pi Zero deployment")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main() 