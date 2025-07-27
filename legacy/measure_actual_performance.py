#!/usr/bin/env python3
"""
Focused Performance Measurement Script
Measures actual RAM usage and FPS for quantized vs non-quantized models
All measurements are real, not hardcoded
"""

import torch
import torchvision.models as models
import timm
import time
import psutil
import os
import gc
import numpy as np
from typing import Dict, Tuple
import json

class ActualPerformanceMeasurer:
    """Measures actual performance metrics"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        
    def clear_memory(self):
        """Clear memory and garbage collect"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_current_ram_mb(self):
        """Get current RAM usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def measure_model_ram_usage(self, model_loader_func):
        """Measure actual RAM usage when loading and storing a model"""
        self.clear_memory()
        baseline_ram = self.get_current_ram_mb()
        
        # Load model
        model = model_loader_func()
        model.eval()
        
        # Measure RAM after loading
        after_load_ram = self.get_current_ram_mb()
        actual_ram_usage = after_load_ram - baseline_ram
        
        # Calculate theoretical size
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        theoretical_mb = (param_bytes + buffer_bytes) / (1024 * 1024)
        
        return {
            'model': model,
            'actual_ram_mb': actual_ram_usage,
            'theoretical_mb': theoretical_mb,
            'baseline_ram_mb': baseline_ram,
            'after_load_ram_mb': after_load_ram,
            'efficiency_ratio': theoretical_mb / actual_ram_usage if actual_ram_usage > 0 else 0,
            'total_params': sum(p.numel() for p in model.parameters())
        }
    
    def measure_inference_fps(self, model, input_tensor, warmup_runs=20, measurement_runs=100):
        """Measure actual FPS and inference times"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Measure inference times
        inference_times = []
        
        with torch.no_grad():
            for _ in range(measurement_runs):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        fps = 1.0 / avg_time
        
        return {
            'fps': fps,
            'avg_inference_ms': avg_time * 1000,
            'std_inference_ms': std_time * 1000,
            'min_inference_ms': min_time * 1000,
            'max_inference_ms': max_time * 1000,
            'measurement_runs': measurement_runs
        }
    
    def create_quantized_model(self, original_model):
        """Create quantized version"""
        try:
            quantized = torch.quantization.quantize_dynamic(
                original_model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
            return quantized
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            return None
    
    def create_mobile_optimized_model(self, original_model, input_tensor):
        """Create mobile optimized version"""
        try:
            traced = torch.jit.trace(original_model, input_tensor)
            mobile_optimized = torch.utils.mobile_optimizer.optimize_for_mobile(traced)
            return mobile_optimized
        except Exception as e:
            print(f"❌ Mobile optimization failed: {e}")
            return None
    
    def compare_model_variants(self, model_name, model_loader, input_tensor):
        """Compare all variants of a model with actual measurements"""
        print(f"\n🔍 Analyzing {model_name}")
        print("=" * 50)
        
        results = {}
        
        # 1. Original Model
        print("📊 Measuring original model...")
        original_ram = self.measure_model_ram_usage(model_loader)
        original_perf = self.measure_inference_fps(original_ram['model'], input_tensor)
        
        results['original'] = {
            'ram': original_ram,
            'performance': original_perf
        }
        
        print(f"   RAM: {original_ram['actual_ram_mb']:.1f}MB (theoretical: {original_ram['theoretical_mb']:.1f}MB)")
        print(f"   FPS: {original_perf['fps']:.1f}, Inference: {original_perf['avg_inference_ms']:.2f}ms")
        
        # 2. Quantized Model
        print("⚡ Measuring quantized model...")
        quantized_model = self.create_quantized_model(original_ram['model'])
        
        if quantized_model:
            quantized_ram = self.measure_model_ram_usage(lambda: quantized_model)
            quantized_perf = self.measure_inference_fps(quantized_model, input_tensor)
            
            results['quantized'] = {
                'ram': quantized_ram,
                'performance': quantized_perf
            }
            
            # Calculate improvements
            fps_improvement = ((quantized_perf['fps'] - original_perf['fps']) / original_perf['fps']) * 100
            ram_reduction = ((original_ram['actual_ram_mb'] - quantized_ram['actual_ram_mb']) / original_ram['actual_ram_mb']) * 100
            
            print(f"   RAM: {quantized_ram['actual_ram_mb']:.1f}MB ({ram_reduction:+.1f}% change)")
            print(f"   FPS: {quantized_perf['fps']:.1f} ({fps_improvement:+.1f}% improvement)")
            print(f"   Inference: {quantized_perf['avg_inference_ms']:.2f}ms")
        
        # 3. Mobile Optimized Model
        print("📱 Measuring mobile optimized model...")
        mobile_model = self.create_mobile_optimized_model(original_ram['model'], input_tensor)
        
        if mobile_model:
            mobile_perf = self.measure_inference_fps(mobile_model, input_tensor)
            
            results['mobile'] = {
                'performance': mobile_perf
            }
            
            mobile_fps_improvement = ((mobile_perf['fps'] - original_perf['fps']) / original_perf['fps']) * 100
            
            print(f"   FPS: {mobile_perf['fps']:.1f} ({mobile_fps_improvement:+.1f}% improvement)")
            print(f"   Inference: {mobile_perf['avg_inference_ms']:.2f}ms")
        
        # 4. Quantized + Mobile (if both successful)
        if quantized_model and mobile_model:
            print("🚀 Measuring quantized + mobile optimized...")
            quantized_mobile = self.create_mobile_optimized_model(quantized_model, input_tensor)
            
            if quantized_mobile:
                qm_perf = self.measure_inference_fps(quantized_mobile, input_tensor)
                
                results['quantized_mobile'] = {
                    'performance': qm_perf
                }
                
                qm_fps_improvement = ((qm_perf['fps'] - original_perf['fps']) / original_perf['fps']) * 100
                
                print(f"   FPS: {qm_perf['fps']:.1f} ({qm_fps_improvement:+.1f}% improvement)")
                print(f"   Inference: {qm_perf['avg_inference_ms']:.2f}ms")
        
        return results

def main():
    """Main measurement function"""
    print("🎯 Actual Performance Measurement")
    print("📊 All numbers are measured, not hardcoded")
    print("⚡ Comparing quantized vs non-quantized performance")
    print("💾 Measuring actual RAM usage vs theoretical")
    print("=" * 60)
    
    measurer = ActualPerformanceMeasurer()
    
    # Test models
    models_to_test = [
        ("MobileNetV2", lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1), torch.randn(1, 3, 224, 224)),
        ("MobileNetV3", lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1), torch.randn(1, 3, 224, 224)),
        ("EfficientNet-B0", lambda: timm.create_model('efficientnet_b0', pretrained=True), torch.randn(1, 3, 224, 224)),
        ("ResNet50", lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1), torch.randn(1, 3, 224, 224)),
    ]
    
    all_results = {}
    
    for model_name, model_loader, input_tensor in models_to_test:
        try:
            results = measurer.compare_model_variants(model_name, model_loader, input_tensor)
            all_results[model_name] = results
        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")
            continue
    
    # Summary Table
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Original FPS':<12} {'Quantized FPS':<13} {'FPS Gain':<10} {'RAM (MB)':<10}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        if 'original' in results:
            orig_fps = results['original']['performance']['fps']
            orig_ram = results['original']['ram']['actual_ram_mb']
            
            if 'quantized' in results:
                quant_fps = results['quantized']['performance']['fps']
                fps_gain = ((quant_fps - orig_fps) / orig_fps) * 100
                print(f"{model_name:<15} {orig_fps:<12.1f} {quant_fps:<13.1f} {fps_gain:<+9.1f}% {orig_ram:<10.1f}")
            else:
                print(f"{model_name:<15} {orig_fps:<12.1f} {'Failed':<13} {'N/A':<10} {orig_ram:<10.1f}")
    
    # Save detailed results
    output_file = "actual_performance_measurements.json"
    
    # Convert to JSON-serializable format
    json_results = {}
    for model_name, results in all_results.items():
        json_results[model_name] = {}
        for variant, data in results.items():
            json_results[model_name][variant] = {}
            
            if 'ram' in data:
                json_results[model_name][variant]['ram'] = {
                    'actual_mb': data['ram']['actual_ram_mb'],
                    'theoretical_mb': data['ram']['theoretical_mb'],
                    'efficiency_ratio': data['ram']['efficiency_ratio'],
                    'total_params': data['ram']['total_params']
                }
            
            if 'performance' in data:
                json_results[model_name][variant]['performance'] = data['performance']
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Pi Zero recommendations
    print(f"\n🎯 Pi Zero Deployment Analysis:")
    print("=" * 40)
    
    for model_name, results in all_results.items():
        if 'original' in results:
            fps = results['original']['performance']['fps']
            ram = results['original']['ram']['actual_ram_mb']
            
            # Pi Zero has ~512MB RAM, recommend <100MB for model
            # Pi Zero is slow, recommend >5 FPS minimum
            suitable = ram < 100 and fps > 5
            status = "✅ Suitable" if suitable else "❌ Too heavy"
            
            print(f"{model_name}: {status} ({fps:.1f} FPS, {ram:.1f}MB actual RAM)")
    
    print("\n✅ All measurements completed!")
    print("📈 All numbers are actual measurements from your system")

if __name__ == "__main__":
    main() 