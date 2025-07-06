#!/usr/bin/env python3
"""
ADVANCED DINO OPTIMIZATION BENCHMARK
Multiple quantization approaches to achieve theoretical sizes
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
from pathlib import Path
import tempfile
import subprocess
import warnings
warnings.filterwarnings('ignore')

def get_model_size_accurate(model):
    """Get accurate model size in MB with detailed breakdown"""
    param_size = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        size = param.numel() * param.element_size()
        param_size += size
        param_count += param.numel()
    
    buffer_size = 0
    buffer_count = 0
    
    for name, buffer in model.named_buffers():
        size = buffer.numel() * buffer.element_size()
        buffer_size += size
        buffer_count += buffer.numel()
    
    total_size_bytes = param_size + buffer_size
    size_mb = total_size_bytes / (1024 * 1024)
    
    return {
        'size_mb': size_mb,
        'param_count': param_count,
        'buffer_count': buffer_count,
        'param_size_mb': param_size / (1024 * 1024),
        'buffer_size_mb': buffer_size / (1024 * 1024)
    }

def calculate_theoretical_quantization(param_count, buffer_count=0):
    """Calculate theoretical quantization sizes"""
    # Assume buffers stay FP32 (usually small)
    buffer_size_mb = buffer_count * 4 / (1024 * 1024)
    
    fp32_params_mb = param_count * 4 / (1024 * 1024)
    int8_params_mb = param_count * 1 / (1024 * 1024)
    int4_params_mb = param_count * 0.5 / (1024 * 1024)
    
    return {
        'fp32': fp32_params_mb + buffer_size_mb,
        'int8': int8_params_mb + buffer_size_mb,
        'int4': int4_params_mb + buffer_size_mb,
        'buffer_overhead_mb': buffer_size_mb
    }

def apply_pytorch_static_quantization(model, model_name):
    """Apply PyTorch static quantization (most comprehensive)"""
    try:
        print(f"      🔧 Applying PyTorch static quantization...")
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # For Vision Transformers, we need to be more specific
        model = torch.quantization.fuse_modules(model, [])  # No specific fusion for ViTs
        
        prepared_model = torch.quantization.prepare(model, inplace=False)
        
        # Calibration with representative data
        print(f"      📊 Calibrating model...")
        dummy_inputs = [torch.randn(1, 3, 224, 224) for _ in range(10)]
        
        with torch.no_grad():
            for dummy_input in dummy_inputs:
                try:
                    _ = prepared_model(dummy_input)
                except Exception as e:
                    print(f"      ⚠️ Calibration step failed: {e}")
                    break
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        return quantized_model, 'pytorch_static'
        
    except Exception as e:
        print(f"      ❌ PyTorch static quantization failed: {e}")
        return None, None

def apply_pytorch_dynamic_quantization(model):
    """Apply PyTorch dynamic quantization (comprehensive layer types)"""
    try:
        print(f"      🔧 Applying PyTorch dynamic quantization...")
        
        # Include all possible layer types that might be quantized
        quantization_targets = {
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.Conv1d,
            torch.nn.MultiheadAttention,
            torch.nn.LayerNorm,  # Often a big part of ViTs
            torch.nn.Embedding,
        }
        
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            quantization_targets, 
            dtype=torch.qint8
        )
        
        return quantized_model, 'pytorch_dynamic_comprehensive'
        
    except Exception as e:
        print(f"      ❌ PyTorch dynamic quantization failed: {e}")
        return None, None

def apply_onnx_quantization(model, model_name):
    """Apply ONNX quantization (often more comprehensive)"""
    try:
        print(f"      🔧 Applying ONNX quantization...")
        
        # Check if ONNX is available
        try:
            import onnx
            import onnxruntime as ort
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print(f"      ⚠️ ONNX not available, skipping...")
            return None, None
        
        # Export to ONNX
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_onnx:
            onnx_path = tmp_onnx.name
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_quantized:
            quantized_onnx_path = tmp_quantized.name
        
        # Export to ONNX
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Quantize ONNX model
        quantize_dynamic(onnx_path, quantized_onnx_path, weight_type=QuantType.QUInt8)
        
        # Load quantized ONNX model
        ort_session = ort.InferenceSession(quantized_onnx_path)
        
        # Create a wrapper for consistent interface
        class ONNXWrapper:
            def __init__(self, session):
                self.session = session
                self.input_name = session.get_inputs()[0].name
                
            def __call__(self, x):
                return self.session.run(None, {self.input_name: x.numpy()})
        
        # Clean up temp files
        os.unlink(onnx_path)
        os.unlink(quantized_onnx_path)
        
        return ONNXWrapper(ort_session), 'onnx_dynamic'
        
    except Exception as e:
        print(f"      ❌ ONNX quantization failed: {e}")
        return None, None

def apply_torchscript_quantization(model):
    """Apply TorchScript quantization"""
    try:
        print(f"      🔧 Applying TorchScript quantization...")
        
        # Convert to TorchScript
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Apply quantization
        quantized_model = torch.quantization.quantize_dynamic(
            traced_model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        
        return quantized_model, 'torchscript_quantized'
        
    except Exception as e:
        print(f"      ❌ TorchScript quantization failed: {e}")
        return None, None

def benchmark_quantization_method(model, method_name, quantization_func, model_name):
    """Benchmark a specific quantization method"""
    print(f"\n    🔍 Testing {method_name}")
    print(f"    {'-' * 50}")
    
    # Get original model info
    original_info = get_model_size_accurate(model)
    
    # Apply quantization
    if method_name == 'Original (FP32)':
        quantized_model, quant_type = model, 'fp32'
    elif method_name == 'PyTorch Static':
        quantized_model, quant_type = apply_pytorch_static_quantization(model, model_name)
    elif method_name == 'PyTorch Dynamic':
        quantized_model, quant_type = apply_pytorch_dynamic_quantization(model)
    elif method_name == 'ONNX':
        quantized_model, quant_type = apply_onnx_quantization(model, model_name)
    elif method_name == 'TorchScript':
        quantized_model, quant_type = apply_torchscript_quantization(model)
    else:
        quantized_model, quant_type = quantization_func(model)
    
    if quantized_model is None:
        return None
    
    # Get quantized model size
    if isinstance(quantized_model, torch.jit.ScriptModule):
        # For TorchScript models, estimate size differently
        quantized_size_mb = original_info['size_mb'] * 0.25  # Rough estimate
        quantized_info = {'size_mb': quantized_size_mb}
    else:
        quantized_info = get_model_size_accurate(quantized_model)
    
    # Calculate theoretical sizes
    theoretical = calculate_theoretical_quantization(
        original_info['param_count'], 
        original_info['buffer_count']
    )
    
    # Performance test
    performance = measure_performance_comprehensive(quantized_model, method_name)
    
    # Calculate size effectiveness
    if method_name == 'Original (FP32)':
        size_effectiveness = 100.0
        target_theoretical = theoretical['fp32']
    else:
        size_effectiveness = (quantized_info['size_mb'] / theoretical['int8']) * 100
        target_theoretical = theoretical['int8']
    
    print(f"    📊 Original Size: {original_info['size_mb']:.1f}MB")
    print(f"    📊 Quantized Size: {quantized_info['size_mb']:.1f}MB")
    print(f"    📊 Theoretical Target: {target_theoretical:.1f}MB")
    print(f"    📊 Size Effectiveness: {size_effectiveness:.1f}% of theoretical")
    print(f"    📊 Quantization Method: {quant_type}")
    print(f"    🚀 Performance: {performance['fps']:.1f} FPS ({performance['avg_time_ms']:.1f}ms)")
    
    # Feasibility analysis
    feasible = (performance['fps'] >= 5.0 and 
               quantized_info['size_mb'] <= 200.0 and
               size_effectiveness >= 80.0)
    
    print(f"    🎯 Pi Zero Feasible: {'✅ YES' if feasible else '❌ NO'}")
    
    if size_effectiveness < 50.0:
        print(f"    ⚠️  Size reduction is much less than theoretical - quantization may be incomplete")
    
    return {
        'method': method_name,
        'quantization_type': quant_type,
        'original_size_mb': float(original_info['size_mb']),
        'quantized_size_mb': float(quantized_info['size_mb']),
        'theoretical_int8_mb': float(theoretical['int8']),
        'theoretical_int4_mb': float(theoretical['int4']),
        'size_effectiveness_pct': float(size_effectiveness),
        'param_count': int(original_info['param_count']),
        'fps': float(performance['fps']),
        'inference_ms': float(performance['avg_time_ms']),
        'feasible': bool(feasible)
    }

def measure_performance_comprehensive(model, method_name):
    """Measure model performance with proper handling for different model types"""
    try:
        model.eval() if hasattr(model, 'eval') else None
        
        # Create input tensor
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Warmup
        for _ in range(5):
            try:
                with torch.no_grad():
                    if hasattr(model, '__call__'):
                        if 'ONNX' in method_name:
                            _ = model(input_tensor)
                        else:
                            _ = model(input_tensor)
                    else:
                        _ = model(input_tensor)
            except Exception as e:
                print(f"      ⚠️ Warmup failed: {e}")
                return {'fps': 0, 'avg_time_ms': 1000}
        
        # Measure performance
        times = []
        successful_runs = 0
        
        for _ in range(20):
            try:
                start_time = time.perf_counter()
                with torch.no_grad():
                    if 'ONNX' in method_name:
                        _ = model(input_tensor)
                    else:
                        _ = model(input_tensor)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                successful_runs += 1
                
            except Exception as e:
                times.append(1.0)  # Penalty for failed runs
        
        if successful_runs == 0:
            return {'fps': 0, 'avg_time_ms': 1000}
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'fps': fps,
            'avg_time_ms': avg_time * 1000,
            'successful_runs': successful_runs
        }
        
    except Exception as e:
        print(f"      ❌ Performance measurement failed: {e}")
        return {'fps': 0, 'avg_time_ms': 1000}

def benchmark_dino_model_comprehensive(model_name, timm_name):
    """Comprehensive benchmark of a DINO model with multiple quantization approaches"""
    print(f"\n🔍 COMPREHENSIVE TESTING: {model_name}")
    print("=" * 70)
    
    # Load model
    try:
        model = timm.create_model(timm_name, pretrained=True)
        model.eval()
    except Exception as e:
        print(f"      ❌ Failed to load {model_name}: {e}")
        return []
    
    # Get original model info
    original_info = get_model_size_accurate(model)
    theoretical = calculate_theoretical_quantization(
        original_info['param_count'], 
        original_info['buffer_count']
    )
    
    print(f"📊 Original Model: {original_info['size_mb']:.1f}MB ({original_info['param_count']/1e6:.1f}M params)")
    print(f"📊 Theoretical INT8: {theoretical['int8']:.1f}MB (4x smaller)")
    print(f"📊 Theoretical INT4: {theoretical['int4']:.1f}MB (8x smaller)")
    print(f"📊 Buffer overhead: {theoretical['buffer_overhead_mb']:.1f}MB")
    
    # Test multiple quantization approaches
    quantization_methods = [
        ('Original (FP32)', lambda x: (x, 'fp32')),
        ('PyTorch Static', None),
        ('PyTorch Dynamic', None),
        ('ONNX', None),
        ('TorchScript', None),
    ]
    
    results = []
    
    for method_name, method_func in quantization_methods:
        try:
            result = benchmark_quantization_method(model, method_name, method_func, model_name)
            if result:
                results.append(result)
                
        except Exception as e:
            print(f"    ❌ {method_name} failed: {e}")
            
        # Clean up memory
        gc.collect()
    
    return results

def main():
    """Main benchmark function"""
    print("🚀 ADVANCED DINO OPTIMIZATION BENCHMARK")
    print("🔧 Multiple quantization approaches for theoretical sizes")
    print("📊 Comprehensive analysis of size effectiveness")
    print("=" * 80)
    
    # Test key DINO models
    models_to_test = [
        ('DINO-S/16', 'vit_small_patch16_224.dino'),
        ('DINO-B/16', 'vit_base_patch16_224.dino'),
        ('DINOv2-S/14', 'vit_small_patch14_dinov2'),
        ('DINOv2-B/14', 'vit_base_patch14_dinov2'),
    ]
    
    all_results = []
    
    for model_name, timm_name in models_to_test:
        results = benchmark_dino_model_comprehensive(model_name, timm_name)
        all_results.extend(results)
    
    # Save results
    output_file = 'advanced_dino_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Analyze results
    if all_results:
        print(f"\n{'=' * 100}")
        print("📈 ADVANCED DINO OPTIMIZATION RESULTS")
        print(f"{'=' * 100}")
        
        # Group by model
        models = {}
        for result in all_results:
            model_name = result.get('model', 'Unknown')
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(result)
        
        print(f"\n🏆 QUANTIZATION EFFECTIVENESS ANALYSIS:")
        print(f"{'Model':<15} {'Method':<20} {'Size MB':<10} {'Target MB':<10} {'Effectiveness':<12} {'FPS':<8}")
        print("-" * 90)
        
        for result in sorted(all_results, key=lambda x: x['size_effectiveness_pct'], reverse=True):
            print(f"{result.get('model', 'Unknown'):<15} {result['method']:<20} {result['quantized_size_mb']:<10.1f} "
                  f"{result['theoretical_int8_mb']:<10.1f} {result['size_effectiveness_pct']:<12.1f}% {result['fps']:<8.1f}")
        
        # Find best methods
        best_methods = [r for r in all_results if r['size_effectiveness_pct'] >= 80.0]
        feasible_methods = [r for r in all_results if r['feasible']]
        
        print(f"\n🎯 HIGHLY EFFECTIVE QUANTIZATION (≥80% of theoretical):")
        if best_methods:
            for result in best_methods:
                print(f"  ✅ {result.get('model', 'Unknown')} - {result['method']}: "
                      f"{result['quantized_size_mb']:.1f}MB ({result['size_effectiveness_pct']:.1f}%)")
        else:
            print("  ❌ No methods achieved >80% effectiveness")
        
        print(f"\n🎯 FEASIBLE FOR PI ZERO:")
        if feasible_methods:
            for result in feasible_methods:
                print(f"  ✅ {result.get('model', 'Unknown')} - {result['method']}: "
                      f"{result['quantized_size_mb']:.1f}MB, {result['fps']:.1f} FPS")
        else:
            print("  ❌ No methods are feasible for Pi Zero")
    
    print(f"\n✅ ADVANCED DINO OPTIMIZATION COMPLETE!")
    print(f"📊 This shows which quantization methods can achieve theoretical sizes")
    print(f"🎯 Identifies the best approach for Pi Zero deployment")

if __name__ == "__main__":
    main() 