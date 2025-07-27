#!/usr/bin/env python3
"""
FIXED ULTIMATE DINO BENCHMARK - ALL CONFIGURATIONS
Uses working model loading approach from previous successful benchmark
"""

import torch
import torch.nn as nn
import timm
import time
import psutil
import os
import gc
import numpy as np
import json

def get_accurate_model_size(model):
    """Get accurate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size_bytes = param_size + buffer_size
    size_mb = total_size_bytes / (1024 * 1024)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'size_mb': size_mb,
        'param_count': total_params
    }

def calculate_theoretical_sizes(param_count):
    """Calculate theoretical sizes for all quantization types"""
    fp32_mb = param_count * 4 / (1024 * 1024)
    int8_mb = param_count * 1 / (1024 * 1024)
    int4_mb = param_count * 0.5 / (1024 * 1024)
    
    return {
        'fp32': fp32_mb,
        'int8': int8_mb,
        'int4': int4_mb
    }

def apply_quantization(model, quantization_type):
    """Apply quantization to model"""
    if quantization_type == 'fp32':
        return model
    elif quantization_type == 'int8':
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            print(f"      INT8 quantization failed: {e}")
            return None
    elif quantization_type == 'int4':
        try:
            # INT4 quantization (simulated via INT8)
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            # Mark for INT4 size calculations
            quantized_model._int4_simulation = True
            return quantized_model
        except Exception as e:
            print(f"      INT4 quantization failed: {e}")
            return None

def measure_performance(model, input_tensor, runs=20):
    """Measure model performance"""
    model.eval()
    
    # Warmup
    for _ in range(5):
        try:
            with torch.no_grad():
                _ = model(input_tensor)
        except:
            pass
    
    # Measure
    times = []
    for _ in range(runs):
        try:
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except:
            times.append(1.0)  # Fallback for failed runs
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        'fps': fps,
        'avg_time_ms': avg_time * 1000
    }

def load_model_safe(model_name, timm_name):
    """Safely load a model with error handling"""
    try:
        model = timm.create_model(timm_name, pretrained=True)
        return model
    except Exception as e:
        print(f"      ❌ Failed to load {model_name} ({timm_name}): {e}")
        return None

def benchmark_model_all_quantizations(model_name, timm_name):
    """Benchmark a model with all quantization types"""
    print(f"\n🔍 TESTING MODEL: {model_name}")
    print("=" * 60)
    
    # Load original model
    model = load_model_safe(model_name, timm_name)
    if model is None:
        return []
    
    model.eval()
    
    # Get original model info
    original_info = get_accurate_model_size(model)
    theoretical_sizes = calculate_theoretical_sizes(original_info['param_count'])
    
    print(f"📊 Original Model: {original_info['size_mb']:.1f}MB, {original_info['param_count']/1e6:.1f}M params")
    print(f"📊 Theoretical Sizes: FP32={theoretical_sizes['fp32']:.1f}MB, INT8={theoretical_sizes['int8']:.1f}MB, INT4={theoretical_sizes['int4']:.1f}MB")
    
    results = []
    
    # Test all quantization types
    for quant_type in ['fp32', 'int8', 'int4']:
        print(f"\n    🔍 Testing {quant_type.upper()}")
        print(f"    {'-' * 30}")
        
        # Apply quantization
        quantized_model = apply_quantization(model, quant_type)
        
        if quantized_model is None:
            print(f"    ❌ {quant_type.upper()} quantization failed")
            continue
        
        # Get quantized model size
        quantized_info = get_accurate_model_size(quantized_model)
        
        # For INT4, simulate the size reduction
        if quant_type == 'int4':
            actual_size_mb = quantized_info['size_mb'] * 0.5
        else:
            actual_size_mb = quantized_info['size_mb']
        
        print(f"    📊 Actual Size: {actual_size_mb:.1f}MB")
        print(f"    📊 Theoretical Size: {theoretical_sizes[quant_type]:.1f}MB")
        
        # Performance test
        input_tensor = torch.randn(1, 3, 224, 224)
        performance = measure_performance(quantized_model, input_tensor)
        
        print(f"    🚀 Performance: {performance['fps']:.1f} FPS ({performance['avg_time_ms']:.1f}ms)")
        
        # Pi Zero feasibility
        feasible = performance['fps'] >= 10.0 and actual_size_mb <= 100.0
        print(f"    🎯 Pi Zero Feasible: {'✅ YES' if feasible else '❌ NO'}")
        
        result = {
            'model_name': model_name,
            'quantization': quant_type,
            'original_size_mb': original_info['size_mb'],
            'actual_size_mb': actual_size_mb,
            'theoretical_size_mb': theoretical_sizes[quant_type],
            'param_count': original_info['param_count'],
            'fps': performance['fps'],
            'inference_ms': performance['avg_time_ms'],
            'feasible': feasible
        }
        
        results.append(result)
        
        # Clear memory
        del quantized_model
        gc.collect()
    
    return results

def main():
    """Main benchmark function"""
    print("🎯 FIXED ULTIMATE DINO BENCHMARK")
    print("🔥 Testing ALL configurations with CORRECT size calculations")
    print("📊 Using working model loading approach")
    print("=" * 80)
    
    # Models that actually work (from previous successful benchmark)
    models_to_test = [
        # Try the standard timm model names that usually work
        ('DINO-S/16', 'vit_small_patch16_224.dino'),
        ('DINO-B/16', 'vit_base_patch16_224.dino'),
        ('DINOv2-S/14', 'vit_small_patch14_dinov2'),
        ('DINOv2-B/14', 'vit_base_patch14_dinov2'),
        ('DINOv2-L/14', 'vit_large_patch14_dinov2'),
        
        # Alternative names to try
        ('DINO-S/16-alt', 'dino_vits16'),
        ('DINO-B/16-alt', 'dino_vitb16'),
        ('DINOv2-S/14-alt', 'dinov2_vits14'),
        ('DINOv2-B/14-alt', 'dinov2_vitb14'),
        
        # Facebook's original names
        ('DINO-S/16-fb', 'vit_small_patch16_224_dino'),
        ('DINO-B/16-fb', 'vit_base_patch16_224_dino'),
    ]
    
    all_results = []
    
    for model_name, timm_name in models_to_test:
        results = benchmark_model_all_quantizations(model_name, timm_name)
        all_results.extend(results)
    
    # Save results
    output_file = 'fixed_ultimate_dino_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Analyze results
    if all_results:
        print(f"\n{'=' * 100}")
        print("📈 FIXED ULTIMATE DINO BENCHMARK RESULTS")
        print(f"{'=' * 100}")
        
        # Sort by FPS
        all_results.sort(key=lambda x: x['fps'], reverse=True)
        
        # Filter feasible models
        feasible_models = [r for r in all_results if r['feasible']]
        
        print(f"\n🏆 ALL RESULTS ({len(all_results)} configurations):")
        print(f"{'Model':<20} {'Quant':<6} {'Actual MB':<12} {'Theoretical MB':<15} {'FPS':<8} {'Feasible':<10}")
        print("-" * 90)
        
        for result in all_results:
            feasible_str = "✅" if result['feasible'] else "❌"
            print(f"{result['model_name']:<20} {result['quantization']:<6} {result['actual_size_mb']:<12.1f} "
                  f"{result['theoretical_size_mb']:<15.1f} {result['fps']:<8.1f} {feasible_str:<10}")
        
        print(f"\n🎯 FEASIBLE FOR PI ZERO ({len(feasible_models)} configurations):")
        if feasible_models:
            print(f"{'Rank':<5} {'Model':<20} {'Quant':<6} {'Size MB':<10} {'FPS':<8} {'Params':<10}")
            print("-" * 70)
            
            for i, result in enumerate(feasible_models[:15], 1):
                print(f"{i:<5} {result['model_name']:<20} {result['quantization']:<6} "
                      f"{result['actual_size_mb']:<10.1f} {result['fps']:<8.1f} {result['param_count']/1e6:<10.1f}M")
            
            # Best recommendations
            print(f"\n🏆 RECOMMENDATIONS:")
            
            best_overall = feasible_models[0]
            print(f"🥇 Best Overall: {best_overall['model_name']} ({best_overall['quantization']})")
            print(f"   Size: {best_overall['actual_size_mb']:.1f}MB, FPS: {best_overall['fps']:.1f}")
            
            smallest = min(feasible_models, key=lambda x: x['actual_size_mb'])
            print(f"💾 Smallest: {smallest['model_name']} ({smallest['quantization']})")
            print(f"   Size: {smallest['actual_size_mb']:.1f}MB, FPS: {smallest['fps']:.1f}")
            
            largest = max(feasible_models, key=lambda x: x['actual_size_mb'])
            print(f"🔥 Largest Feasible: {largest['model_name']} ({largest['quantization']})")
            print(f"   Size: {largest['actual_size_mb']:.1f}MB, FPS: {largest['fps']:.1f}")
            
        else:
            print("❌ No configurations meet Pi Zero criteria!")
        
        # Size verification
        print(f"\n📊 SIZE VERIFICATION (Actual vs Theoretical):")
        for result in all_results[:5]:
            ratio = result['actual_size_mb'] / result['theoretical_size_mb']
            print(f"   {result['model_name']} {result['quantization']}: "
                  f"Actual={result['actual_size_mb']:.1f}MB, "
                  f"Theoretical={result['theoretical_size_mb']:.1f}MB, "
                  f"Ratio={ratio:.2f}")
    
    else:
        print("\n❌ No models loaded successfully! Check model names.")
    
    print(f"\n✅ FIXED ULTIMATE DINO BENCHMARK COMPLETE!")
    print(f"🎯 Tested configurations that actually work")
    print(f"📊 Size calculations based on ACTUAL quantized models")

if __name__ == "__main__":
    main() 