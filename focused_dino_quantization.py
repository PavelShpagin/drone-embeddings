#!/usr/bin/env python3
"""
FOCUSED DINO QUANTIZATION BENCHMARK
Targeting the most promising quantization approaches for realistic sizes
"""

import torch
import torch.nn as nn
import timm
import time
import numpy as np
import json
import gc
import warnings
warnings.filterwarnings('ignore')

def get_model_info(model):
    """Get detailed model information"""
    param_count = sum(p.numel() for p in model.parameters())
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        'total_size_mb': total_size_mb,
        'param_count': param_count,
        'param_size_mb': param_size / (1024 * 1024),
        'buffer_size_mb': buffer_size / (1024 * 1024)
    }

def calculate_ideal_quantization_sizes(param_count, buffer_size_mb=0):
    """Calculate what the sizes SHOULD be with perfect quantization"""
    fp32_params_mb = param_count * 4 / (1024 * 1024)
    int8_params_mb = param_count * 1 / (1024 * 1024)
    int4_params_mb = param_count * 0.5 / (1024 * 1024)
    
    return {
        'fp32_ideal': fp32_params_mb + buffer_size_mb,
        'int8_ideal': int8_params_mb + buffer_size_mb,
        'int4_ideal': int4_params_mb + buffer_size_mb,
        'compression_ratio_int8': fp32_params_mb / int8_params_mb,
        'compression_ratio_int4': fp32_params_mb / int4_params_mb
    }

def apply_enhanced_dynamic_quantization(model):
    """Apply enhanced dynamic quantization targeting all relevant layers"""
    try:
        print(f"      🔧 Enhanced dynamic quantization...")
        
        # Target all layer types that could be quantized in a ViT
        quantization_targets = {
            torch.nn.Linear,           # Most weights in ViTs
            torch.nn.Conv2d,           # Patch embedding
            torch.nn.LayerNorm,        # Normalization layers
            torch.nn.MultiheadAttention, # Self-attention (if supported)
            torch.nn.Embedding,       # Position embeddings
        }
        
        # Apply quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            quantization_targets, 
            dtype=torch.qint8
        )
        
        return quantized_model, 'enhanced_dynamic'
        
    except Exception as e:
        print(f"      ❌ Enhanced dynamic quantization failed: {e}")
        return None, None

def apply_static_quantization_with_calibration(model):
    """Apply static quantization with proper calibration"""
    try:
        print(f"      🔧 Static quantization with calibration...")
        
        # Prepare model
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare for quantization
        prepared_model = torch.quantization.prepare(model, inplace=False)
        
        # Calibration with multiple samples
        print(f"      📊 Calibrating with representative data...")
        calibration_samples = [
            torch.randn(1, 3, 224, 224) for _ in range(20)
        ]
        
        with torch.no_grad():
            for i, sample in enumerate(calibration_samples):
                try:
                    _ = prepared_model(sample)
                except Exception as e:
                    print(f"      ⚠️ Calibration sample {i} failed: {e}")
                    continue
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        return quantized_model, 'static_calibrated'
        
    except Exception as e:
        print(f"      ❌ Static quantization failed: {e}")
        return None, None

def apply_selective_quantization(model):
    """Apply selective quantization - only Linear layers (most conservative)"""
    try:
        print(f"      🔧 Selective quantization (Linear only)...")
        
        # Only quantize Linear layers (most weights in ViTs)
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        return quantized_model, 'selective_linear'
        
    except Exception as e:
        print(f"      ❌ Selective quantization failed: {e}")
        return None, None

def measure_performance_robust(model, method_name):
    """Robust performance measurement"""
    try:
        model.eval() if hasattr(model, 'eval') else None
        
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Warmup
        print(f"      🔥 Warming up {method_name}...")
        for _ in range(5):
            with torch.no_grad():
                try:
                    _ = model(input_tensor)
                except Exception as e:
                    print(f"      ⚠️ Warmup failed: {e}")
                    return {'fps': 0, 'inference_ms': 1000, 'success': False}
        
        # Benchmark
        print(f"      ⏱️ Benchmarking {method_name}...")
        times = []
        
        for i in range(30):
            try:
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(input_tensor)
                end = time.perf_counter()
                
                times.append(end - start)
                
            except Exception as e:
                print(f"      ❌ Run {i} failed: {e}")
                times.append(1.0)  # Penalty
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'fps': fps,
            'inference_ms': avg_time * 1000,
            'success': True
        }
        
    except Exception as e:
        print(f"      ❌ Performance measurement failed: {e}")
        return {'fps': 0, 'inference_ms': 1000, 'success': False}

def benchmark_dino_quantization_focused(model_name, timm_name):
    """Focused benchmark of DINO quantization approaches"""
    print(f"\n🎯 FOCUSED QUANTIZATION: {model_name}")
    print("=" * 60)
    
    # Load model
    try:
        model = timm.create_model(timm_name, pretrained=True)
        model.eval()
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return []
    
    # Get model info
    original_info = get_model_info(model)
    ideal_sizes = calculate_ideal_quantization_sizes(
        original_info['param_count'], 
        original_info['buffer_size_mb']
    )
    
    print(f"📊 Original: {original_info['total_size_mb']:.1f}MB ({original_info['param_count']/1e6:.1f}M params)")
    print(f"📊 Ideal INT8: {ideal_sizes['int8_ideal']:.1f}MB ({ideal_sizes['compression_ratio_int8']:.1f}x smaller)")
    print(f"📊 Ideal INT4: {ideal_sizes['int4_ideal']:.1f}MB ({ideal_sizes['compression_ratio_int4']:.1f}x smaller)")
    
    # Test quantization methods
    quantization_methods = [
        ('Original FP32', lambda x: (x, 'fp32')),
        ('Enhanced Dynamic', apply_enhanced_dynamic_quantization),
        ('Static Calibrated', apply_static_quantization_with_calibration),
        ('Selective Linear', apply_selective_quantization),
    ]
    
    results = []
    
    for method_name, method_func in quantization_methods:
        print(f"\n  🔍 Testing: {method_name}")
        print(f"  {'-' * 50}")
        
        try:
            # Apply quantization
            if method_name == 'Original FP32':
                quantized_model, quant_type = model, 'fp32'
            else:
                quantized_model, quant_type = method_func(model)
            
            if quantized_model is None:
                print(f"  ❌ {method_name} failed")
                continue
            
            # Get quantized model info
            quantized_info = get_model_info(quantized_model)
            
            # Calculate effectiveness
            if method_name == 'Original FP32':
                size_effectiveness = 100.0
                target_size = ideal_sizes['fp32_ideal']
            else:
                size_effectiveness = min(100.0, (ideal_sizes['int8_ideal'] / quantized_info['total_size_mb']) * 100)
                target_size = ideal_sizes['int8_ideal']
            
            # Performance test
            performance = measure_performance_robust(quantized_model, method_name)
            
            print(f"  📊 Actual Size: {quantized_info['total_size_mb']:.1f}MB")
            print(f"  📊 Target Size: {target_size:.1f}MB")
            print(f"  📊 Size Effectiveness: {size_effectiveness:.1f}%")
            print(f"  🚀 Performance: {performance['fps']:.1f} FPS ({performance['inference_ms']:.1f}ms)")
            
            # Pi Zero feasibility
            feasible = (performance['fps'] >= 5.0 and 
                       quantized_info['total_size_mb'] <= 150.0 and
                       size_effectiveness >= 60.0)
            
            print(f"  🎯 Pi Zero Feasible: {'✅ YES' if feasible else '❌ NO'}")
            
            # Store results
            result = {
                'model_name': model_name,
                'method': method_name,
                'quantization_type': quant_type,
                'original_size_mb': float(original_info['total_size_mb']),
                'quantized_size_mb': float(quantized_info['total_size_mb']),
                'ideal_int8_size_mb': float(ideal_sizes['int8_ideal']),
                'ideal_int4_size_mb': float(ideal_sizes['int4_ideal']),
                'size_effectiveness_pct': float(size_effectiveness),
                'param_count': int(original_info['param_count']),
                'fps': float(performance['fps']),
                'inference_ms': float(performance['inference_ms']),
                'feasible': bool(feasible),
                'success': bool(performance['success'])
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ {method_name} failed with error: {e}")
            
        # Memory cleanup
        if 'quantized_model' in locals():
            del quantized_model
        gc.collect()
    
    return results

def main():
    """Main benchmark function"""
    print("🎯 FOCUSED DINO QUANTIZATION BENCHMARK")
    print("🔧 Testing the most promising quantization approaches")
    print("📊 Targeting theoretical size effectiveness")
    print("=" * 80)
    
    # Test key models
    models_to_test = [
        ('DINO-S/16', 'vit_small_patch16_224.dino'),
        ('DINO-B/16', 'vit_base_patch16_224.dino'),
        ('DINOv2-S/14', 'vit_small_patch14_dinov2'),
        ('DINOv2-B/14', 'vit_base_patch14_dinov2'),
    ]
    
    all_results = []
    
    for model_name, timm_name in models_to_test:
        results = benchmark_dino_quantization_focused(model_name, timm_name)
        all_results.extend(results)
        
        print(f"\n{'='*80}")
    
    # Save results
    output_file = 'focused_dino_quantization_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Analysis
    if all_results:
        print(f"\n📈 FOCUSED QUANTIZATION ANALYSIS")
        print("=" * 80)
        
        # Sort by effectiveness
        all_results.sort(key=lambda x: x['size_effectiveness_pct'], reverse=True)
        
        print(f"\n🏆 QUANTIZATION EFFECTIVENESS RANKING:")
        print(f"{'Model':<15} {'Method':<20} {'Actual MB':<10} {'Ideal MB':<10} {'Effectiveness':<12} {'FPS':<8}")
        print("-" * 90)
        
        for result in all_results[:10]:  # Top 10
            print(f"{result['model_name']:<15} {result['method']:<20} {result['quantized_size_mb']:<10.1f} "
                  f"{result['ideal_int8_size_mb']:<10.1f} {result['size_effectiveness_pct']:<12.1f}% {result['fps']:<8.1f}")
        
        # Find best approaches
        high_effectiveness = [r for r in all_results if r['size_effectiveness_pct'] >= 70.0]
        feasible_configs = [r for r in all_results if r['feasible']]
        
        print(f"\n🎯 HIGH EFFECTIVENESS (≥70% of ideal):")
        if high_effectiveness:
            for result in high_effectiveness:
                print(f"  ✅ {result['model_name']} - {result['method']}: "
                      f"{result['quantized_size_mb']:.1f}MB ({result['size_effectiveness_pct']:.1f}%)")
        else:
            print("  ❌ No methods achieved ≥70% effectiveness")
        
        print(f"\n🎯 FEASIBLE FOR PI ZERO:")
        if feasible_configs:
            for result in feasible_configs:
                print(f"  ✅ {result['model_name']} - {result['method']}: "
                      f"{result['quantized_size_mb']:.1f}MB, {result['fps']:.1f} FPS")
        else:
            print("  ❌ No configurations are feasible for Pi Zero")
            
        print(f"\n🔍 KEY INSIGHTS:")
        if all_results:
            best_method = max(all_results, key=lambda x: x['size_effectiveness_pct'])
            print(f"  🏆 Best method: {best_method['method']} ({best_method['size_effectiveness_pct']:.1f}% effective)")
            
            avg_effectiveness = np.mean([r['size_effectiveness_pct'] for r in all_results if r['method'] != 'Original FP32'])
            print(f"  📊 Average effectiveness: {avg_effectiveness:.1f}%")
            
            if avg_effectiveness < 50:
                print(f"  ⚠️  Current quantization methods are not achieving theoretical sizes")
                print(f"  💡 Consider alternative approaches like ONNX or TensorRT")
    
    print(f"\n✅ FOCUSED DINO QUANTIZATION COMPLETE!")
    print(f"🎯 This identifies the most promising quantization approaches")
    print(f"📊 Shows realistic expectations for DINO model deployment")

if __name__ == "__main__":
    main() 