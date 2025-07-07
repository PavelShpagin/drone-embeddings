#!/usr/bin/env python3
"""
Pi Zero Performance Benchmark
Lightweight script to test pre-created quantized models
"""

import os
import gc
import psutil
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Only import torch if needed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available")
    TORCH_AVAILABLE = False

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_system_resources():
    """Check Pi Zero resources"""
    print("🔍 SYSTEM RESOURCE CHECK")
    print("=" * 50)
    
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print(f"📊 Total RAM: {memory.total / 1024**2:.0f} MB")
    print(f"📊 Available RAM: {memory.available / 1024**2:.0f} MB") 
    print(f"📊 Used RAM: {memory.used / 1024**2:.0f} MB")
    print(f"💾 Swap Total: {swap.total / 1024**2:.0f} MB")
    print(f"💾 Swap Used: {swap.used / 1024**2:.0f} MB")
    
    # Warnings
    if memory.available < 100:
        print(f"⚠️ Low available RAM ({memory.available / 1024**2:.0f}MB)")
    
    if swap.total == 0:
        print(f"⚠️ No swap memory - consider adding 1GB swap")
    
    return memory.available / 1024**2  # Return available MB

def load_model_info():
    """Load model creation summary"""
    summary_path = Path("model_creation_summary.json")
    
    if not summary_path.exists():
        print(f"❌ Model summary not found: {summary_path}")
        return []
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    return data.get('models', [])

def benchmark_model(model_info, available_ram_mb):
    """Benchmark a single model"""
    model_path = Path(model_info['file_path'])
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print(f"\n🎯 BENCHMARKING: {model_info['name']} ({model_info['quantization']})")
    print(f"📁 File: {model_path.name}")
    print(f"📊 File size: {model_info['file_size_mb']:.1f}MB")
    print(f"📊 Theoretical: {model_info['theoretical_size_mb']:.1f}MB")
    
    # Check if model fits in RAM
    if model_info['file_size_mb'] > available_ram_mb * 0.8:  # Use 80% of available RAM
        print(f"⚠️ Model too large for available RAM ({available_ram_mb:.0f}MB)")
        return {
            'name': model_info['name'],
            'quantization': model_info['quantization'],
            'status': 'too_large',
            'file_size_mb': model_info['file_size_mb'],
            'available_ram_mb': available_ram_mb
        }
    
    if not TORCH_AVAILABLE:
        print(f"⚠️ PyTorch not available - skipping actual loading")
        return {
            'name': model_info['name'],
            'quantization': model_info['quantization'],
            'status': 'pytorch_unavailable',
            'file_size_mb': model_info['file_size_mb']
        }
    
    # Memory before loading
    baseline_memory = get_memory_usage()
    print(f"📊 Baseline memory: {baseline_memory:.1f}MB")
    
    try:
        # Load model
        print(f"🔄 Loading model...")
        start_load = time.perf_counter()
        
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        
        load_time = time.perf_counter() - start_load
        after_load_memory = get_memory_usage()
        model_memory = after_load_memory - baseline_memory
        
        print(f"✅ Model loaded in {load_time:.2f}s")
        print(f"📊 Model memory usage: {model_memory:.1f}MB")
        
        # Prepare input
        input_size = model_info['input_size']
        sample_input = torch.randn(1, 3, input_size, input_size)
        
        # Warmup
        print(f"🔥 Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Benchmark inference
        print(f"⏱️ Benchmarking inference...")
        times = []
        num_runs = 20  # Reduced for Pi Zero
        
        for i in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(sample_input)
            end = time.perf_counter()
            times.append(end - start)
            
            if i % 5 == 0:
                print(f"    Run {i+1}/{num_runs} complete")
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        peak_memory = get_memory_usage()
        
        print(f"🚀 Average inference time: {avg_time*1000:.1f}ms")
        print(f"🚀 FPS: {fps:.2f}")
        print(f"📊 Peak memory: {peak_memory:.1f}MB")
        
        # Clean up
        del model
        del sample_input
        gc.collect()
        
        return {
            'name': model_info['name'],
            'quantization': model_info['quantization'],
            'status': 'success',
            'file_size_mb': model_info['file_size_mb'],
            'theoretical_size_mb': model_info['theoretical_size_mb'],
            'load_time_s': load_time,
            'model_memory_mb': model_memory,
            'peak_memory_mb': peak_memory,
            'avg_inference_ms': avg_time * 1000,
            'fps': fps,
            'input_size': input_size,
            'compression_ratio': model_info.get('compression_ratio', 1.0)
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return {
            'name': model_info['name'],
            'quantization': model_info['quantization'],
            'status': 'failed',
            'error': str(e),
            'file_size_mb': model_info['file_size_mb']
        }

def main():
    """Main benchmark function"""
    print("🤖 PI ZERO PERFORMANCE BENCHMARK")
    print("🔧 Testing pre-created quantized models")
    print("📊 Progressive loading from smallest to largest")
    print("=" * 60)
    
    # Check system resources
    available_ram = check_system_resources()
    
    # Load model information
    print(f"\n📂 Loading model information...")
    model_infos = load_model_info()
    
    if not model_infos:
        print(f"❌ No models found. Run create_tflite_models.py first!")
        return
    
    print(f"✅ Found {len(model_infos)} models")
    
    # Sort by file size (progressive loading)
    model_infos.sort(key=lambda x: x['file_size_mb'])
    
    print(f"\n🎯 PROGRESSIVE BENCHMARK ORDER:")
    for i, info in enumerate(model_infos, 1):
        print(f"  {i}. {info['name']} ({info['quantization']}) - {info['file_size_mb']:.1f}MB")
    
    # Benchmark each model
    results = []
    
    for i, model_info in enumerate(model_infos, 1):
        print(f"\n{'='*60}")
        print(f"📈 Progress: {i}/{len(model_infos)}")
        
        result = benchmark_model(model_info, available_ram)
        if result:
            results.append(result)
        
        # Force garbage collection between models
        gc.collect()
        
        # Check if we should continue
        current_memory = get_memory_usage()
        if current_memory > available_ram * 0.9:
            print(f"⚠️ Memory usage too high ({current_memory:.1f}MB), stopping")
            break
    
    # Save results
    results_path = Path("perf_benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_ram_mb': psutil.virtual_memory().total / 1024**2,
            'available_ram_mb': available_ram,
            'results': results
        }, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📈 BENCHMARK COMPLETE!")
    print(f"💾 Results saved: {results_path}")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        print(f"\n🏆 SUCCESSFUL BENCHMARKS ({len(successful)}):")
        print(f"{'Model':<20} {'Quant':<6} {'Size MB':<8} {'FPS':<8} {'Memory MB':<10}")
        print("-" * 60)
        
        for result in successful:
            print(f"{result['name']:<20} {result['quantization']:<6} {result['file_size_mb']:<8.1f} "
                  f"{result['fps']:<8.2f} {result['model_memory_mb']:<10.1f}")
        
        # Best performers
        best_fps = max(successful, key=lambda x: x['fps'])
        smallest = min(successful, key=lambda x: x['file_size_mb'])
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"  🚀 Fastest: {best_fps['name']} ({best_fps['quantization']}) - {best_fps['fps']:.2f} FPS")
        print(f"  📦 Smallest: {smallest['name']} ({smallest['quantization']}) - {smallest['file_size_mb']:.1f}MB")
    
    failed = [r for r in results if r['status'] != 'success']
    if failed:
        print(f"\n❌ FAILED/SKIPPED ({len(failed)}):")
        for result in failed:
            status = result['status'].replace('_', ' ').title()
            print(f"  • {result['name']} ({result['quantization']}): {status}")
    
    print(f"\n✅ Pi Zero benchmark complete!")

if __name__ == "__main__":
    main() 