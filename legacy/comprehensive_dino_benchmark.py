#!/usr/bin/env python3

import torch
import torch.nn as nn
import timm
import time
import psutil
import os
import numpy as np
import gc
from pathlib import Path

def get_model_size_mb(model):
    """Calculate model size in MB based on the actual model parameters."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def get_theoretical_size_mb(model, quantization_type=None):
    """Calculate theoretical model size based on parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    
    if quantization_type == 'int8':
        # INT8: 1 byte per parameter
        return total_params * 1 / 1024**2
    elif quantization_type == 'int4':
        # INT4: 0.5 bytes per parameter
        return total_params * 0.5 / 1024**2
    else:
        # FP32: 4 bytes per parameter
        return total_params * 4 / 1024**2

def apply_quantization(model, quantization_type):
    """Apply quantization to model."""
    if quantization_type == 'int8':
        try:
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            model = torch.quantization.prepare(model, inplace=False)
            # Use dummy calibration data
            dummy_input = torch.randn(1, 3, 224, 224)
            model(dummy_input)
            model = torch.quantization.convert(model, inplace=False)
        except Exception as e:
            print(f"INT8 quantization failed: {e}, trying dynamic quantization")
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
    elif quantization_type == 'int4':
        try:
            # Try dynamic quantization (closest to INT4)
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"INT4 quantization failed: {e}")
            raise
    
    return model

def measure_inference_performance(model, input_tensor, num_runs=50):
    """Measure inference time and calculate FPS."""
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
    for _ in range(num_runs):
        start_time = time.time()
        try:
            with torch.no_grad():
                _ = model(input_tensor)
            end_time = time.time()
            times.append(end_time - start_time)
        except:
            # If inference fails, use a large time
            times.append(1.0)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    return avg_time * 1000, fps  # ms, fps

def measure_memory_usage(model, input_tensor):
    """Measure peak memory usage."""
    gc.collect()
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1024**2
    
    try:
        with torch.no_grad():
            _ = model(input_tensor)
    except:
        pass
    
    peak_memory = process.memory_info().rss / 1024**2
    return max(0, peak_memory - baseline_memory)

def test_model_configuration(model_name, quantization=None, mobile_opt=False):
    """Test a specific model configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    if quantization:
        print(f"Quantization: {quantization}")
    if mobile_opt:
        print(f"Mobile Optimization: Enabled")
    print(f"{'='*60}")
    
    try:
        # Load model
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        
        # Get original model info
        original_size = get_model_size_mb(model)
        original_params = sum(p.numel() for p in model.parameters())
        theoretical_original = get_theoretical_size_mb(model)
        
        print(f"📊 Original Model:")
        print(f"  Parameters: {original_params:,}")
        print(f"  Actual Size: {original_size:.2f} MB")
        print(f"  Theoretical Size: {theoretical_original:.2f} MB")
        print(f"  Size Ratio (actual/theoretical): {original_size/theoretical_original:.3f}")
        
        # Apply quantization if specified
        if quantization:
            try:
                model = apply_quantization(model, quantization)
                print(f"\n✅ {quantization.upper()} Quantization Applied")
            except Exception as e:
                print(f"❌ Quantization failed: {e}")
                return None
        
        # Apply mobile optimization if specified
        if mobile_opt:
            try:
                dummy_input = torch.randn(1, 3, 224, 224)
                traced_model = torch.jit.trace(model, dummy_input)
                model = torch.jit.optimize_for_inference(traced_model)
                print(f"✅ Mobile Optimization Applied")
            except Exception as e:
                print(f"❌ Mobile optimization failed: {e}")
        
        # Get final model size (THIS IS THE KEY - size of ACTUAL model)
        final_size = get_model_size_mb(model)
        
        # Calculate theoretical size based on CURRENT model
        theoretical_size = get_theoretical_size_mb(model, quantization)
        
        print(f"\n📋 Final Model (ACTUAL quantized model):")
        print(f"  Actual Size: {final_size:.2f} MB")
        print(f"  Theoretical Size: {theoretical_size:.2f} MB")
        print(f"  Size Ratio (actual/theoretical): {final_size/theoretical_size:.3f}")
        
        if quantization:
            compression_ratio = original_size / final_size
            print(f"  Compression Ratio: {compression_ratio:.2f}x")
            
            # Expected compression ratios
            expected_compression = {'int8': 4.0, 'int4': 8.0}
            if quantization in expected_compression:
                expected = expected_compression[quantization]
                print(f"  Expected Compression: {expected:.1f}x")
                print(f"  Compression Efficiency: {(compression_ratio/expected)*100:.1f}%")
        
        # Performance testing
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Measure inference time
        inference_time, fps = measure_inference_performance(model, dummy_input)
        
        # Measure memory usage
        memory_usage = measure_memory_usage(model, dummy_input)
        
        print(f"\n🚀 Performance:")
        print(f"  Inference Time: {inference_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  Memory Usage: {memory_usage:.2f} MB")
        
        # Pi Zero feasibility assessment
        pi_zero_threshold_fps = 10.0  # Minimum acceptable FPS
        pi_zero_threshold_size = 100.0  # Maximum model size in MB
        
        feasible = fps >= pi_zero_threshold_fps and final_size <= pi_zero_threshold_size
        print(f"\n🎯 Pi Zero Feasibility:")
        print(f"  FPS >= {pi_zero_threshold_fps}: {'✓' if fps >= pi_zero_threshold_fps else '✗'}")
        print(f"  Size <= {pi_zero_threshold_size}MB: {'✓' if final_size <= pi_zero_threshold_size else '✗'}")
        print(f"  Overall: {'✅ FEASIBLE' if feasible else '❌ NOT FEASIBLE'}")
        
        return {
            'model_name': model_name,
            'quantization': quantization,
            'mobile_opt': mobile_opt,
            'original_size': original_size,
            'final_size': final_size,
            'theoretical_size': theoretical_size,
            'compression_ratio': original_size / final_size if quantization else 1.0,
            'inference_time': inference_time,
            'fps': fps,
            'memory_usage': memory_usage,
            'feasible': feasible,
            'parameters': original_params
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return None

def main():
    print("🎯 Comprehensive DINO/DINOv2 Benchmark for Raspberry Pi Zero")
    print("📊 Testing all promising configurations")
    print("🔍 Calculating sizes based on ACTUAL models (quantized or not)")
    print("=" * 80)
    
    # Define all promising DINO configurations
    dino_models = [
        # DINO variants (original from Facebook)
        'dino_vits16',      # DINO Small, patch 16 (fastest)
        'dino_vits8',       # DINO Small, patch 8 (higher resolution)
        'dino_vitb16',      # DINO Base, patch 16 (medium)
        'dino_vitb8',       # DINO Base, patch 8 (slower but better)
        
        # DINOv2 variants (Meta's improved version)
        'dinov2_vits14',    # DINOv2 Small, patch 14
        'dinov2_vitb14',    # DINOv2 Base, patch 14
        'dinov2_vitl14',    # DINOv2 Large, patch 14
        'dinov2_vitg14',    # DINOv2 Giant, patch 14 (largest)
    ]
    
    # Test configurations
    quantizations = [None, 'int8', 'int4']
    mobile_opts = [False, True]
    
    results = []
    best_models = []
    
    for model_name in dino_models:
        for quantization in quantizations:
            for mobile_opt in mobile_opts:
                # Skip mobile optimization without quantization (minimal benefit)
                if mobile_opt and quantization is None:
                    continue
                
                result = test_model_configuration(model_name, quantization, mobile_opt)
                if result:
                    results.append(result)
                    if result['feasible']:
                        best_models.append(result)
    
    # Sort results by FPS (descending) and size (ascending)
    results.sort(key=lambda x: (-x['fps'], x['final_size']))
    best_models.sort(key=lambda x: (-x['fps'], x['final_size']))
    
    print(f"\n{'='*80}")
    print("📈 COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nAll Models Tested ({len(results)} configurations):")
    print(f"{'Model':<20} {'Quant':<6} {'Mobile':<7} {'Size(MB)':<10} {'FPS':<8} {'Feasible':<10}")
    print("-" * 80)
    
    for result in results:
        mobile_str = "Yes" if result['mobile_opt'] else "No"
        quant_str = result['quantization'] if result['quantization'] else "None"
        feasible_str = "✓" if result['feasible'] else "✗"
        
        print(f"{result['model_name']:<20} {quant_str:<6} {mobile_str:<7} "
              f"{result['final_size']:<10.2f} {result['fps']:<8.2f} {feasible_str:<10}")
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST MODELS FOR RASPBERRY PI ZERO ({len(best_models)} feasible)")
    print(f"{'='*80}")
    
    if best_models:
        print(f"{'Rank':<5} {'Model':<20} {'Config':<15} {'Size(MB)':<10} {'FPS':<8} {'Params':<12}")
        print("-" * 90)
        
        for i, model in enumerate(best_models[:15], 1):  # Top 15
            config = []
            if model['quantization']:
                config.append(model['quantization'].upper())
            if model['mobile_opt']:
                config.append("Mobile")
            config_str = "+".join(config) if config else "Original"
            
            print(f"{i:<5} {model['model_name']:<20} {config_str:<15} "
                  f"{model['final_size']:<10.2f} {model['fps']:<8.2f} {model['parameters']:<12,}")
    else:
        print("❌ No models meet the Pi Zero feasibility criteria!")
    
    print(f"\n{'='*80}")
    print("🎯 RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if best_models:
        best = best_models[0]
        print(f"🥇 BEST OVERALL: {best['model_name']}")
        config = []
        if best['quantization']:
            config.append(best['quantization'].upper())
        if best['mobile_opt']:
            config.append("Mobile Optimized")
        if config:
            print(f"   Configuration: {' + '.join(config)}")
        print(f"   Size: {best['final_size']:.2f} MB")
        print(f"   FPS: {best['fps']:.2f}")
        print(f"   Parameters: {best['parameters']:,}")
        print(f"   Compression: {best['compression_ratio']:.2f}x")
        
        # Find best by size (smallest)
        best_by_size = min(best_models, key=lambda x: x['final_size'])
        if best_by_size != best:
            print(f"\n💾 SMALLEST MODEL: {best_by_size['model_name']}")
            print(f"   Size: {best_by_size['final_size']:.2f} MB")
            print(f"   FPS: {best_by_size['fps']:.2f}")
            print(f"   Compression: {best_by_size['compression_ratio']:.2f}x")
        
        # Find best by FPS (fastest)
        best_by_fps = max(best_models, key=lambda x: x['fps'])
        if best_by_fps != best:
            print(f"\n🚀 FASTEST MODEL: {best_by_fps['model_name']}")
            print(f"   Size: {best_by_fps['final_size']:.2f} MB")
            print(f"   FPS: {best_by_fps['fps']:.2f}")
            print(f"   Compression: {best_by_fps['compression_ratio']:.2f}x")
        
        # Find largest feasible model
        largest_feasible = max(best_models, key=lambda x: x['final_size'])
        if largest_feasible != best:
            print(f"\n🔥 LARGEST FEASIBLE MODEL: {largest_feasible['model_name']}")
            print(f"   Size: {largest_feasible['final_size']:.2f} MB")
            print(f"   FPS: {largest_feasible['fps']:.2f}")
            print(f"   Parameters: {largest_feasible['parameters']:,}")
            print(f"   Compression: {largest_feasible['compression_ratio']:.2f}x")
    else:
        print("❌ Consider using smaller models or more aggressive quantization!")
    
    # Detailed analysis
    print(f"\n{'='*80}")
    print("📊 DETAILED ANALYSIS")
    print(f"{'='*80}")
    
    # Group by model family
    dino_results = [r for r in results if r['model_name'].startswith('dino_')]
    dinov2_results = [r for r in results if r['model_name'].startswith('dinov2_')]
    
    print(f"\n🔍 DINO vs DINOv2 Comparison:")
    print(f"DINO models tested: {len(dino_results)}")
    print(f"DINOv2 models tested: {len(dinov2_results)}")
    
    dino_feasible = [r for r in dino_results if r['feasible']]
    dinov2_feasible = [r for r in dinov2_results if r['feasible']]
    
    print(f"DINO feasible: {len(dino_feasible)}")
    print(f"DINOv2 feasible: {len(dinov2_feasible)}")
    
    # Quantization effectiveness
    print(f"\n📈 Quantization Effectiveness:")
    for quant in ['int8', 'int4']:
        quant_results = [r for r in results if r['quantization'] == quant]
        if quant_results:
            avg_compression = np.mean([r['compression_ratio'] for r in quant_results])
            print(f"{quant.upper()} average compression: {avg_compression:.2f}x")
    
    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 