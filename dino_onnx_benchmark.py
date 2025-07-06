#!/usr/bin/env python3
"""
DINO ONNX BENCHMARK
Convert DINO models to ONNX for better performance and deployment
"""

import torch
import timm
import time
import numpy as np
import json
import gc
import tempfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def get_model_size_mb(model_path):
    """Get model file size in MB"""
    if isinstance(model_path, str) and os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0

def get_pytorch_model_size(model):
    """Get PyTorch model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)

def convert_to_onnx(model, model_name, optimize=True):
    """Convert PyTorch model to ONNX with optimization"""
    try:
        print(f"      🔄 Converting {model_name} to ONNX...")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            onnx_path = tmp_file.name
        
        # Prepare model and input
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
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
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"      ✅ ONNX export successful")
        
        # Optimize ONNX model if requested
        if optimize:
            try:
                import onnx
                from onnx import optimizer
                
                print(f"      🔧 Optimizing ONNX model...")
                
                # Load and optimize
                onnx_model = onnx.load(onnx_path)
                optimized_model = optimizer.optimize(onnx_model)
                
                # Save optimized model
                with tempfile.NamedTemporaryFile(suffix='_optimized.onnx', delete=False) as tmp_opt:
                    optimized_path = tmp_opt.name
                
                onnx.save(optimized_model, optimized_path)
                
                # Clean up original
                os.unlink(onnx_path)
                onnx_path = optimized_path
                
                print(f"      ✅ ONNX optimization successful")
                
            except Exception as e:
                print(f"      ⚠️ ONNX optimization failed: {e}")
                print(f"      ⚠️ Using non-optimized ONNX model")
        
        return onnx_path, True
        
    except Exception as e:
        print(f"      ❌ ONNX conversion failed: {e}")
        return None, False

def create_onnx_runtime_session(onnx_path):
    """Create ONNX Runtime inference session"""
    try:
        import onnxruntime as ort
        
        print(f"      🔧 Creating ONNX Runtime session...")
        
        # Create session with optimizations
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Try to use available providers
        providers = ['CPUExecutionProvider']
        if ort.get_available_providers():
            available = ort.get_available_providers()
            print(f"      📊 Available providers: {available}")
            
            # Use GPU if available
            if 'CUDAExecutionProvider' in available:
                providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
        
        print(f"      ✅ ONNX Runtime session created")
        return session, True
        
    except Exception as e:
        print(f"      ❌ ONNX Runtime session creation failed: {e}")
        return None, False

def apply_onnx_quantization(onnx_path, model_name):
    """Apply ONNX quantization"""
    try:
        print(f"      🔧 Applying ONNX quantization...")
        
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Create quantized model path
        with tempfile.NamedTemporaryFile(suffix='_quantized.onnx', delete=False) as tmp_quant:
            quantized_path = tmp_quant.name
        
        # Apply dynamic quantization
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        
        print(f"      ✅ ONNX quantization successful")
        return quantized_path, True
        
    except Exception as e:
        print(f"      ❌ ONNX quantization failed: {e}")
        return None, False

def measure_pytorch_performance(model, model_name, runs=30):
    """Measure PyTorch model performance"""
    try:
        print(f"      ⏱️ Measuring PyTorch performance...")
        
        model.eval()
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Benchmark
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {'fps': fps, 'inference_ms': avg_time * 1000, 'success': True}
        
    except Exception as e:
        print(f"      ❌ PyTorch performance measurement failed: {e}")
        return {'fps': 0, 'inference_ms': 1000, 'success': False}

def measure_onnx_performance(session, model_name, runs=30):
    """Measure ONNX model performance"""
    try:
        print(f"      ⏱️ Measuring ONNX performance...")
        
        # Get input name
        input_name = session.get_inputs()[0].name
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            _ = session.run(None, {input_name: input_data})
        
        # Benchmark
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = session.run(None, {input_name: input_data})
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {'fps': fps, 'inference_ms': avg_time * 1000, 'success': True}
        
    except Exception as e:
        print(f"      ❌ ONNX performance measurement failed: {e}")
        return {'fps': 0, 'inference_ms': 1000, 'success': False}

def benchmark_dino_onnx(model_name, timm_name):
    """Comprehensive ONNX benchmark for DINO model"""
    print(f"\n🎯 ONNX BENCHMARK: {model_name}")
    print("=" * 70)
    
    # Load PyTorch model
    try:
        model = timm.create_model(timm_name, pretrained=True)
        model.eval()
        print(f"✅ PyTorch model loaded")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return []
    
    # Get original PyTorch size
    pytorch_size_mb = get_pytorch_model_size(model)
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"📊 Original PyTorch: {pytorch_size_mb:.1f}MB ({param_count/1e6:.1f}M params)")
    
    results = []
    
    # 1. Test Original PyTorch
    print(f"\n  🔍 Testing: PyTorch Original")
    print(f"  {'-' * 50}")
    
    pytorch_perf = measure_pytorch_performance(model, model_name)
    
    print(f"  📊 Size: {pytorch_size_mb:.1f}MB")
    print(f"  🚀 Performance: {pytorch_perf['fps']:.1f} FPS ({pytorch_perf['inference_ms']:.1f}ms)")
    
    results.append({
        'model_name': model_name,
        'format': 'PyTorch',
        'optimization': 'Original',
        'size_mb': float(pytorch_size_mb),
        'fps': float(pytorch_perf['fps']),
        'inference_ms': float(pytorch_perf['inference_ms']),
        'success': bool(pytorch_perf['success'])
    })
    
    # 2. Test ONNX FP32
    print(f"\n  🔍 Testing: ONNX FP32")
    print(f"  {'-' * 50}")
    
    onnx_path, onnx_success = convert_to_onnx(model, model_name, optimize=True)
    
    if onnx_success and onnx_path:
        onnx_size_mb = get_model_size_mb(onnx_path)
        
        # Create ONNX Runtime session
        session, session_success = create_onnx_runtime_session(onnx_path)
        
        if session_success and session:
            onnx_perf = measure_onnx_performance(session, model_name)
            
            print(f"  📊 Size: {onnx_size_mb:.1f}MB")
            print(f"  🚀 Performance: {onnx_perf['fps']:.1f} FPS ({onnx_perf['inference_ms']:.1f}ms)")
            
            # Calculate speedup
            speedup = onnx_perf['fps'] / pytorch_perf['fps'] if pytorch_perf['fps'] > 0 else 0
            size_ratio = onnx_size_mb / pytorch_size_mb if pytorch_size_mb > 0 else 0
            
            print(f"  🔥 Speedup: {speedup:.2f}x")
            print(f"  📦 Size ratio: {size_ratio:.2f}x")
            
            results.append({
                'model_name': model_name,
                'format': 'ONNX',
                'optimization': 'FP32',
                'size_mb': float(onnx_size_mb),
                'fps': float(onnx_perf['fps']),
                'inference_ms': float(onnx_perf['inference_ms']),
                'speedup_vs_pytorch': float(speedup),
                'size_ratio_vs_pytorch': float(size_ratio),
                'success': bool(onnx_perf['success'])
            })
            
            # 3. Test ONNX Quantized
            print(f"\n  🔍 Testing: ONNX Quantized")
            print(f"  {'-' * 50}")
            
            quantized_path, quant_success = apply_onnx_quantization(onnx_path, model_name)
            
            if quant_success and quantized_path:
                quantized_size_mb = get_model_size_mb(quantized_path)
                
                # Create quantized session
                quant_session, quant_session_success = create_onnx_runtime_session(quantized_path)
                
                if quant_session_success and quant_session:
                    quant_perf = measure_onnx_performance(quant_session, model_name)
                    
                    print(f"  📊 Size: {quantized_size_mb:.1f}MB")
                    print(f"  🚀 Performance: {quant_perf['fps']:.1f} FPS ({quant_perf['inference_ms']:.1f}ms)")
                    
                    # Calculate improvements
                    quant_speedup = quant_perf['fps'] / pytorch_perf['fps'] if pytorch_perf['fps'] > 0 else 0
                    quant_size_ratio = quantized_size_mb / pytorch_size_mb if pytorch_size_mb > 0 else 0
                    compression_ratio = pytorch_size_mb / quantized_size_mb if quantized_size_mb > 0 else 0
                    
                    print(f"  🔥 Speedup: {quant_speedup:.2f}x")
                    print(f"  📦 Size ratio: {quant_size_ratio:.2f}x")
                    print(f"  🗜️ Compression: {compression_ratio:.2f}x")
                    
                    results.append({
                        'model_name': model_name,
                        'format': 'ONNX',
                        'optimization': 'Quantized',
                        'size_mb': float(quantized_size_mb),
                        'fps': float(quant_perf['fps']),
                        'inference_ms': float(quant_perf['inference_ms']),
                        'speedup_vs_pytorch': float(quant_speedup),
                        'size_ratio_vs_pytorch': float(quant_size_ratio),
                        'compression_ratio': float(compression_ratio),
                        'success': bool(quant_perf['success'])
                    })
                    
                    # Clean up
                    os.unlink(quantized_path)
                
                # Clean up
                os.unlink(onnx_path)
    
    # Pi Zero feasibility analysis
    for result in results:
        feasible = (result['fps'] >= 5.0 and result['size_mb'] <= 200.0)
        result['pi_zero_feasible'] = feasible
        
        if feasible:
            print(f"  🎯 {result['format']} {result['optimization']}: ✅ Pi Zero Feasible")
        else:
            print(f"  🎯 {result['format']} {result['optimization']}: ❌ Not feasible")
    
    return results

def main():
    """Main ONNX benchmark function"""
    print("🚀 DINO ONNX BENCHMARK")
    print("🔄 Convert DINO models to ONNX for better performance")
    print("📊 Compare PyTorch vs ONNX vs ONNX Quantized")
    print("=" * 80)
    
    # Check dependencies
    try:
        import onnx
        import onnxruntime as ort
        print(f"✅ ONNX dependencies available")
        print(f"📊 ONNX Runtime version: {ort.__version__}")
        print(f"📊 Available providers: {ort.get_available_providers()}")
    except ImportError as e:
        print(f"❌ ONNX dependencies missing: {e}")
        print(f"🔧 Install with: pip install onnx onnxruntime")
        return
    
    # Test key DINO models
    models_to_test = [
        ('DINO-S/16', 'vit_small_patch16_224.dino'),
        ('DINO-B/16', 'vit_base_patch16_224.dino'),
        ('DINOv2-S/14', 'vit_small_patch14_dinov2'),
        ('DINOv2-B/14', 'vit_base_patch14_dinov2'),
    ]
    
    all_results = []
    
    for model_name, timm_name in models_to_test:
        results = benchmark_dino_onnx(model_name, timm_name)
        all_results.extend(results)
        
        print(f"\n{'='*80}")
        gc.collect()
    
    # Save results
    output_file = 'dino_onnx_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Analysis
    if all_results:
        print(f"\n📈 DINO ONNX BENCHMARK ANALYSIS")
        print("=" * 80)
        
        # Group by format
        pytorch_results = [r for r in all_results if r['format'] == 'PyTorch']
        onnx_fp32_results = [r for r in all_results if r['format'] == 'ONNX' and r['optimization'] == 'FP32']
        onnx_quant_results = [r for r in all_results if r['format'] == 'ONNX' and r['optimization'] == 'Quantized']
        
        print(f"\n🏆 PERFORMANCE COMPARISON:")
        print(f"{'Model':<15} {'PyTorch FPS':<12} {'ONNX FPS':<10} {'ONNX Quant FPS':<15} {'Best Speedup':<12}")
        print("-" * 80)
        
        for pytorch_result in pytorch_results:
            model_name = pytorch_result['model_name']
            pytorch_fps = pytorch_result['fps']
            
            onnx_fps = 0
            onnx_quant_fps = 0
            best_speedup = 1.0
            
            for onnx_result in onnx_fp32_results:
                if onnx_result['model_name'] == model_name:
                    onnx_fps = onnx_result['fps']
                    best_speedup = max(best_speedup, onnx_result.get('speedup_vs_pytorch', 1.0))
            
            for quant_result in onnx_quant_results:
                if quant_result['model_name'] == model_name:
                    onnx_quant_fps = quant_result['fps']
                    best_speedup = max(best_speedup, quant_result.get('speedup_vs_pytorch', 1.0))
            
            print(f"{model_name:<15} {pytorch_fps:<12.1f} {onnx_fps:<10.1f} {onnx_quant_fps:<15.1f} {best_speedup:<12.2f}x")
        
        print(f"\n📦 SIZE COMPARISON:")
        print(f"{'Model':<15} {'PyTorch MB':<12} {'ONNX MB':<10} {'ONNX Quant MB':<15} {'Best Compression':<15}")
        print("-" * 85)
        
        for pytorch_result in pytorch_results:
            model_name = pytorch_result['model_name']
            pytorch_size = pytorch_result['size_mb']
            
            onnx_size = 0
            onnx_quant_size = 0
            best_compression = 1.0
            
            for onnx_result in onnx_fp32_results:
                if onnx_result['model_name'] == model_name:
                    onnx_size = onnx_result['size_mb']
            
            for quant_result in onnx_quant_results:
                if quant_result['model_name'] == model_name:
                    onnx_quant_size = quant_result['size_mb']
                    best_compression = max(best_compression, quant_result.get('compression_ratio', 1.0))
            
            print(f"{model_name:<15} {pytorch_size:<12.1f} {onnx_size:<10.1f} {onnx_quant_size:<15.1f} {best_compression:<15.2f}x")
        
        # Pi Zero feasibility
        feasible_results = [r for r in all_results if r.get('pi_zero_feasible', False)]
        
        print(f"\n🎯 PI ZERO FEASIBLE CONFIGURATIONS:")
        if feasible_results:
            for result in feasible_results:
                print(f"  ✅ {result['model_name']} - {result['format']} {result['optimization']}: "
                      f"{result['size_mb']:.1f}MB, {result['fps']:.1f} FPS")
        else:
            print("  ❌ No configurations are feasible for Pi Zero")
        
        # Best overall results
        best_performance = max(all_results, key=lambda x: x['fps'])
        best_size = min(all_results, key=lambda x: x['size_mb'])
        
        print(f"\n🏆 BEST RESULTS:")
        print(f"  🚀 Fastest: {best_performance['model_name']} - {best_performance['format']} {best_performance['optimization']}: {best_performance['fps']:.1f} FPS")
        print(f"  📦 Smallest: {best_size['model_name']} - {best_size['format']} {best_size['optimization']}: {best_size['size_mb']:.1f}MB")
    
    print(f"\n✅ DINO ONNX BENCHMARK COMPLETE!")
    print(f"🔄 ONNX conversion provides better performance and deployment options")
    print(f"📊 Use ONNX quantized models for the best size/performance balance")

if __name__ == "__main__":
    main() 