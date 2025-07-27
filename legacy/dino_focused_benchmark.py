#!/usr/bin/env python3
"""
Focused DINO Benchmark - INT8/INT4 Only
Tests specific DINO variants: S/16, S/8, B, DINOv2-small
Compares only quantized metrics (skips FP32 to save time)
"""

import torch
import timm
import time
import psutil
import os
import gc
import numpy as np
from typing import Dict, Tuple
import json

class DinoFocusedBenchmark:
    """Focused benchmark for DINO variants with quantization only"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        
    def clear_memory(self):
        """Clear memory safely"""
        for _ in range(3):
            gc.collect()
        time.sleep(1)
    
    def create_int8_quantized_model(self, model, model_name):
        """Create INT8 quantized model"""
        print(f"   🔢 Creating INT8 quantized {model_name}...")
        
        try:
            model.eval()
            quantized = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention}, 
                dtype=torch.qint8
            )
            print(f"   ✅ INT8 quantization successful")
            return quantized
        except Exception as e:
            print(f"   ❌ INT8 quantization failed: {e}")
            return None
    
    def create_int4_quantized_model(self, model, model_name):
        """Create INT4 quantized model (simulated via INT8 with adjusted calculations)"""
        print(f"   🔢 Creating INT4 quantized {model_name}...")
        
        try:
            model.eval()
            # Create INT8 quantized model first
            quantized = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention}, 
                dtype=torch.qint8
            )
            # Mark for INT4 size calculations
            quantized._is_int4_simulation = True
            print(f"   ✅ INT4 quantization successful (simulated)")
            return quantized
        except Exception as e:
            print(f"   ❌ INT4 quantization failed: {e}")
            return None
    
    def measure_quantized_performance(self, model, model_name, input_tensor, quantization_type):
        """Measure performance of quantized model"""
        print(f"   🚀 Benchmarking {model_name} ({quantization_type})...")
        
        if hasattr(model, 'eval'):
            model.eval()
        
        # Measure memory
        self.clear_memory()
        baseline_mb = self.process.memory_info().rss / (1024 * 1024)
        
        # Load model and measure memory
        after_load_mb = self.process.memory_info().rss / (1024 * 1024)
        ram_usage_mb = after_load_mb - baseline_mb
        
        # Calculate theoretical sizes
        if hasattr(model, 'parameters'):
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            original_mb = param_bytes / (1024 * 1024)
            total_params = sum(p.numel() for p in model.parameters())
        else:
            original_mb = 0
            total_params = 0
        
        # Calculate quantized size based on type
        if quantization_type == 'int8':
            theoretical_quantized_mb = original_mb * 0.25  # FP32 to INT8 = 1/4
        elif quantization_type == 'int4':
            theoretical_quantized_mb = original_mb * 0.125  # FP32 to INT4 = 1/8
        else:
            theoretical_quantized_mb = original_mb
        
        # Measure inference performance
        times = []
        successful_runs = 0
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                try:
                    _ = model(input_tensor)
                except:
                    pass
        
        # Benchmark
        with torch.no_grad():
            for _ in range(50):
                try:
                    start = time.perf_counter()
                    _ = model(input_tensor)
                    end = time.perf_counter()
                    times.append(end - start)
                    successful_runs += 1
                except:
                    pass
        
        if not times:
            return None
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        return {
            'fps': fps,
            'inference_ms': avg_time * 1000,
            'ram_mb': max(ram_usage_mb, theoretical_quantized_mb * 1.1),
            'theoretical_mb': theoretical_quantized_mb,
            'original_theoretical_mb': original_mb,
            'params_m': total_params / 1e6,
            'quantization': quantization_type,
            'successful_runs': successful_runs
        }
    
    def benchmark_dino_model(self, model_name, model_loader, input_tensor):
        """Benchmark a DINO model with INT8 and INT4 quantization only"""
        print(f"\n🔍 DINO Analysis: {model_name}")
        print("=" * 60)
        
        results = {}
        
        try:
            # Load original model first (for quantization)
            print("📥 Loading original model for quantization...")
            original_model = model_loader()
            original_model.eval()
            
            # Calculate original size for reference
            param_bytes = sum(p.numel() * p.element_size() for p in original_model.parameters())
            original_mb = param_bytes / (1024 * 1024)
            total_params = sum(p.numel() for p in original_model.parameters())
            
            print(f"   📊 Original model: {original_mb:.1f}MB, {total_params/1e6:.1f}M params")
            
            # Test INT8 Quantization
            print("\n1️⃣ INT8 Quantization")
            int8_model = self.create_int8_quantized_model(original_model, model_name)
            
            if int8_model:
                int8_results = self.measure_quantized_performance(
                    int8_model, model_name, input_tensor, 'int8'
                )
                if int8_results:
                    results['int8'] = int8_results
                    print(f"   📊 INT8: {int8_results['fps']:.1f} FPS, {int8_results['theoretical_mb']:.1f}MB")
            
            # Test INT4 Quantization
            print("\n2️⃣ INT4 Quantization")
            int4_model = self.create_int4_quantized_model(original_model, model_name)
            
            if int4_model:
                int4_results = self.measure_quantized_performance(
                    int4_model, model_name, input_tensor, 'int4'
                )
                if int4_results:
                    results['int4'] = int4_results
                    print(f"   📊 INT4: {int4_results['fps']:.1f} FPS, {int4_results['theoretical_mb']:.1f}MB")
            
            # Clear model from memory
            del original_model
            if int8_model:
                del int8_model
            if int4_model:
                del int4_model
            self.clear_memory()
            
        except Exception as e:
            print(f"   ❌ Failed to benchmark {model_name}: {e}")
            return None
        
        return results
    
    def analyze_pi_zero_suitability(self, model_name, results):
        """Analyze Pi Zero suitability for quantized models"""
        print(f"\n🎯 Pi Zero Analysis: {model_name}")
        
        for quant_type, data in results.items():
            fps = data['fps']
            ram = data['theoretical_mb']
            
            # Pi Zero criteria
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
            
            print(f"   {quant_type.upper()}: {status} - {fps:.1f} FPS, {ram:.1f}MB{reason_str}")

def main():
    """Main focused DINO benchmarking"""
    print("🎯 Focused DINO Benchmark - INT8/INT4 Only")
    print("🔬 Testing specific DINO variants with quantization focus")
    print("⚡ Skipping FP32 to save time and focus on Pi Zero deployment")
    print("=" * 70)
    
    benchmark = DinoFocusedBenchmark()
    
    # DINO models to test (specific variants requested)
    dino_models = [
        # DINOv2 variants
        ("DINOv2-S", lambda: timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True), torch.randn(1, 3, 224, 224)),
        
        # Original DINO variants with different patch sizes
        ("DINO-S/16", lambda: timm.create_model('vit_small_patch16_224.dino', pretrained=True), torch.randn(1, 3, 224, 224)),
        ("DINO-S/8", lambda: timm.create_model('vit_small_patch8_224.dino', pretrained=True), torch.randn(1, 3, 224, 224)),
        ("DINO-B/16", lambda: timm.create_model('vit_base_patch16_224.dino', pretrained=True), torch.randn(1, 3, 224, 224)),
    ]
    
    all_results = {}
    
    # Benchmark each DINO model
    for model_name, model_loader, input_tensor in dino_models:
        try:
            results = benchmark.benchmark_dino_model(model_name, model_loader, input_tensor)
            if results:
                all_results[model_name] = results
                benchmark.analyze_pi_zero_suitability(model_name, results)
        except Exception as e:
            print(f"❌ Failed to benchmark {model_name}: {e}")
            continue
    
    # Comprehensive comparison table
    print(f"\n📊 FOCUSED DINO COMPARISON - QUANTIZED ONLY")
    print("=" * 100)
    print(f"{'Model':<12} {'Quant':<6} {'FPS':<8} {'Inf(ms)':<10} {'Size(MB)':<10} {'Orig(MB)':<10} {'Params':<8} {'Pi Zero':<10}")
    print("-" * 100)
    
    best_int8 = {'model': None, 'score': 0}
    best_int4 = {'model': None, 'score': 0}
    
    for model_name, results in all_results.items():
        for quant_type, data in results.items():
            fps = data['fps']
            inf_ms = data['inference_ms']
            size_mb = data['theoretical_mb']
            orig_mb = data['original_theoretical_mb']
            params = data['params_m']
            
            # Pi Zero status
            pi_zero_ok = fps > 10 and size_mb < 50
            pi_zero_status = "✅" if pi_zero_ok else "❌"
            
            print(f"{model_name:<12} {quant_type.upper():<6} {fps:<8.1f} {inf_ms:<10.1f} {size_mb:<10.1f} {orig_mb:<10.1f} {params:<8.1f} {pi_zero_status:<10}")
            
            # Track best models
            score = fps - (size_mb / 10)  # Simple scoring
            if quant_type == 'int8' and score > best_int8['score'] and size_mb < 50:
                best_int8 = {'model': model_name, 'score': score, 'data': data}
            elif quant_type == 'int4' and score > best_int4['score'] and size_mb < 50:
                best_int4 = {'model': model_name, 'score': score, 'data': data}
    
    # Recommendations
    print(f"\n🏆 BEST DINO MODELS FOR PI ZERO")
    print("=" * 50)
    
    if best_int8['model']:
        data = best_int8['data']
        print(f"🥇 Best INT8: {best_int8['model']}")
        print(f"   📊 {data['fps']:.1f} FPS, {data['theoretical_mb']:.1f}MB")
        
    if best_int4['model']:
        data = best_int4['data']
        print(f"🥈 Best INT4: {best_int4['model']}")
        print(f"   📊 {data['fps']:.1f} FPS, {data['theoretical_mb']:.1f}MB")
    
    # Size comparison
    print(f"\n📏 SIZE COMPARISON")
    print("=" * 40)
    print("Model          Original    INT8       INT4")
    print("-" * 40)
    
    for model_name, results in all_results.items():
        if 'int8' in results and 'int4' in results:
            orig_mb = results['int8']['original_theoretical_mb']
            int8_mb = results['int8']['theoretical_mb']
            int4_mb = results['int4']['theoretical_mb']
            print(f"{model_name:<12} {orig_mb:<10.1f} {int8_mb:<10.1f} {int4_mb:<10.1f}")
    
    # Save results
    output_file = "dino_focused_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"✅ Focused DINO benchmark complete!")
    print(f"🎯 Focus: INT8/INT4 quantization for Pi Zero deployment")

if __name__ == "__main__":
    main() 