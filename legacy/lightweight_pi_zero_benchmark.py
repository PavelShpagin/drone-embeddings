#!/usr/bin/env python3
"""
Lightweight Pi Zero Benchmark
Tests only viable models for Pi Zero to avoid memory crashes
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

class LightweightBenchmark:
    """Lightweight benchmark focusing on Pi Zero viable models"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        
    def clear_memory(self):
        """Clear memory safely"""
        for _ in range(3):
            gc.collect()
        time.sleep(1)
    
    def measure_model_performance(self, model_name, model_loader, input_tensor):
        """Measure performance safely"""
        print(f"\n🔍 Testing {model_name}")
        print("=" * 50)
        
        results = {}
        
        # Get baseline memory
        self.clear_memory()
        baseline_mb = self.process.memory_info().rss / (1024 * 1024)
        
        try:
            # Load model
            model = model_loader()
            model.eval()
            
            # Measure memory after loading
            after_load_mb = self.process.memory_info().rss / (1024 * 1024)
            ram_usage_mb = after_load_mb - baseline_mb
            
            # Calculate theoretical size
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            theoretical_mb = param_bytes / (1024 * 1024)
            total_params = sum(p.numel() for p in model.parameters())
            
            # Measure inference performance
            print(f"   🚀 Measuring inference performance...")
            times = []
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)
            
            # Benchmark
            with torch.no_grad():
                for _ in range(50):
                    start = time.perf_counter()
                    _ = model(input_tensor)
                    end = time.perf_counter()
                    times.append(end - start)
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time
            
            results['original'] = {
                'fps': fps,
                'inference_ms': avg_time * 1000,
                'ram_mb': max(ram_usage_mb, theoretical_mb * 1.1),
                'theoretical_mb': theoretical_mb,
                'params_m': total_params / 1e6
            }
            
            print(f"   📊 Original: {fps:.1f} FPS, {ram_usage_mb:.1f}MB RAM")
            
            # Test quantization
            print(f"   ⚡ Testing INT8 quantization...")
            try:
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                
                # Measure quantized performance
                times_q = []
                with torch.no_grad():
                    for _ in range(10):
                        _ = quantized_model(input_tensor)
                    
                    for _ in range(50):
                        start = time.perf_counter()
                        _ = quantized_model(input_tensor)
                        end = time.perf_counter()
                        times_q.append(end - start)
                
                avg_time_q = np.mean(times_q)
                fps_q = 1.0 / avg_time_q
                
                # Calculate quantized memory (theoretical)
                quantized_theoretical_mb = theoretical_mb * 0.25  # INT8 is 1/4 of FP32
                
                results['quantized'] = {
                    'fps': fps_q,
                    'inference_ms': avg_time_q * 1000,
                    'ram_mb': ram_usage_mb * 0.5,  # Estimate
                    'theoretical_mb': quantized_theoretical_mb,
                    'params_m': total_params / 1e6
                }
                
                fps_improvement = ((fps_q - fps) / fps) * 100
                print(f"   📊 Quantized: {fps_q:.1f} FPS ({fps_improvement:+.1f}%), {quantized_theoretical_mb:.1f}MB theoretical")
                
            except Exception as e:
                print(f"   ❌ Quantization failed: {e}")
                
        except Exception as e:
            print(f"   ❌ Failed to test {model_name}: {e}")
            return None
        
        return results
    
    def analyze_pi_zero_suitability(self, model_name, results):
        """Analyze if model is suitable for Pi Zero"""
        if not results or 'original' not in results:
            return "❌ Failed to test"
        
        orig = results['original']
        fps = orig['fps']
        ram = orig['ram_mb']
        
        # Pi Zero criteria: >10 FPS, <50MB RAM (conservative)
        fps_good = fps > 10
        ram_good = ram < 50
        
        if fps_good and ram_good:
            status = "✅ Excellent"
        elif fps > 8 and ram < 80:
            status = "👍 Good"
        else:
            status = "❌ Not suitable"
        
        reasons = []
        if not fps_good:
            reasons.append("low FPS")
        if not ram_good:
            reasons.append("high RAM")
        
        reason_str = f" ({', '.join(reasons)})" if reasons else ""
        
        return f"{status}{reason_str}"

def main():
    """Main lightweight benchmarking"""
    print("🎯 Lightweight Pi Zero Benchmark")
    print("📊 Testing only viable models to avoid memory crashes")
    print("🔬 Focus on Pi Zero deployment suitability")
    print("=" * 60)
    
    benchmark = LightweightBenchmark()
    
    # Test only lightweight models that won't crash
    models_to_test = [
        ("MobileNetV2", lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1), torch.randn(1, 3, 224, 224)),
        ("MobileNetV3", lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1), torch.randn(1, 3, 224, 224)),
        ("EfficientNet-B0", lambda: timm.create_model('efficientnet_b0', pretrained=True), torch.randn(1, 3, 224, 224)),
        ("DINO-S", lambda: timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True), torch.randn(1, 3, 224, 224)),
    ]
    
    all_results = {}
    
    for model_name, model_loader, input_tensor in models_to_test:
        results = benchmark.measure_model_performance(model_name, model_loader, input_tensor)
        if results:
            all_results[model_name] = results
            
            # Analyze Pi Zero suitability
            suitability = benchmark.analyze_pi_zero_suitability(model_name, results)
            print(f"   🎯 Pi Zero suitability: {suitability}")
    
    # Summary
    print(f"\n📊 LIGHTWEIGHT BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'FPS':<8} {'RAM(MB)':<10} {'Quant FPS':<12} {'Pi Zero':<15}")
    print("-" * 70)
    
    best_model = None
    best_score = 0
    
    for model_name, results in all_results.items():
        orig = results['original']
        fps = orig['fps']
        ram = orig['ram_mb']
        
        quant_fps = "N/A"
        if 'quantized' in results:
            quant_fps = f"{results['quantized']['fps']:.1f}"
        
        suitability = benchmark.analyze_pi_zero_suitability(model_name, results)
        pi_zero_status = suitability.split()[0]  # Get emoji/status
        
        print(f"{model_name:<15} {fps:<8.1f} {ram:<10.1f} {quant_fps:<12} {pi_zero_status:<15}")
        
        # Calculate score for best model (higher FPS, lower RAM)
        score = fps - (ram / 10)  # Simple scoring
        if score > best_score and ram < 50:
            best_score = score
            best_model = model_name
    
    print(f"\n🏆 BEST MODEL FOR PI ZERO: {best_model}")
    
    if best_model and best_model in all_results:
        best_results = all_results[best_model]
        orig = best_results['original']
        
        print(f"   📊 Performance: {orig['fps']:.1f} FPS, {orig['ram_mb']:.1f}MB RAM")
        print(f"   🧠 Parameters: {orig['params_m']:.1f}M")
        print(f"   💾 Theoretical size: {orig['theoretical_mb']:.1f}MB")
        
        if 'quantized' in best_results:
            quant = best_results['quantized']
            improvement = ((quant['fps'] - orig['fps']) / orig['fps']) * 100
            print(f"   ⚡ Quantized: {quant['fps']:.1f} FPS ({improvement:+.1f}% improvement)")
    
    # Save results
    output_file = "lightweight_pi_zero_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"✅ Lightweight benchmark complete!")

if __name__ == "__main__":
    main() 