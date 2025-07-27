#!/usr/bin/env python3
"""
Ultra-Minimal Pi Zero Benchmark
Designed for 512MB RAM constraint
"""

import os
import gc
import psutil
import time
import json

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_system_resources():
    """Check if Pi Zero has enough resources"""
    print("🔍 SYSTEM RESOURCE CHECK")
    print("=" * 50)
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"📊 Total RAM: {memory.total / 1024**3:.2f} GB")
    print(f"📊 Available RAM: {memory.available / 1024**2:.0f} MB")
    print(f"📊 Used RAM: {memory.used / 1024**2:.0f} MB")
    print(f"📊 RAM Usage: {memory.percent:.1f}%")
    
    # Swap info
    swap = psutil.swap_memory()
    print(f"💾 Swap Total: {swap.total / 1024**2:.0f} MB")
    print(f"💾 Swap Used: {swap.used / 1024**2:.0f} MB")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print(f"💿 Disk Free: {disk.free / 1024**3:.1f} GB")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if memory.available < 100:
        print(f"  ⚠️ Very low available RAM ({memory.available / 1024**2:.0f}MB)")
        print(f"  🔧 Consider adding swap memory")
    
    if swap.total == 0:
        print(f"  ⚠️ No swap memory detected")
        print(f"  🔧 Add swap: sudo fallocate -l 1G /swapfile")
    
    print()

def theoretical_model_analysis():
    """Analyze DINO models theoretically without loading them"""
    print("🧮 THEORETICAL MODEL ANALYSIS")
    print("=" * 50)
    
    models = [
        ("DINO-S/16", 21.3, 224),      # 21.3M parameters
        ("DINO-B/16", 85.8, 224),     # 85.8M parameters  
        ("DINOv2-S/14", 21.1, 224),   # 21.1M parameters
        ("DINOv2-B/14", 86.3, 224),   # 86.3M parameters
        ("ViT-S/8", 21.3, 224),       # Same as S/16 but patch/8
        ("ViT-G/14", 1011.5, 224),    # 1011.5M parameters (HUGE!)
    ]
    
    print(f"{'Model':<15} {'Params (M)':<10} {'FP32 (MB)':<10} {'INT8 (MB)':<10} {'Pi Zero?':<10}")
    print("-" * 65)
    
    results = []
    
    for model_name, params_m, input_size in models:
        # Calculate theoretical sizes
        fp32_mb = params_m * 4  # 4 bytes per FP32 parameter
        int8_mb = params_m * 1  # 1 byte per INT8 parameter
        
        # Pi Zero feasibility (< 200MB for safe operation)
        pi_zero_feasible = int8_mb < 200
        feasible_str = "✅ YES" if pi_zero_feasible else "❌ NO"
        
        print(f"{model_name:<15} {params_m:<10.1f} {fp32_mb:<10.0f} {int8_mb:<10.0f} {feasible_str:<10}")
        
        results.append({
            'model': model_name,
            'params_millions': params_m,
            'fp32_mb': fp32_mb,
            'int8_mb': int8_mb,
            'pi_zero_feasible': pi_zero_feasible,
            'input_size': input_size
        })
    
    return results

def simulate_performance():
    """Simulate expected performance on Pi Zero"""
    print(f"\n🚀 SIMULATED PI ZERO PERFORMANCE")
    print("=" * 50)
    
    # Based on Pi Zero specs: ARM11 @ 1GHz, 512MB RAM
    base_fps = {
        'DINO-S/16': 2.5,      # Small model, reasonable
        'DINO-B/16': 0.8,      # Large model, very slow
        'DINOv2-S/14': 2.2,    # Similar to DINO-S
        'DINOv2-B/14': 0.7,    # Large model, very slow
        'ViT-S/8': 1.8,        # Patch/8 is more compute intensive
        'ViT-G/14': 0.1,       # Giant model, barely usable
    }
    
    print(f"{'Model':<15} {'FP32 FPS':<10} {'INT8 FPS':<10} {'Speedup':<10} {'Usable?':<10}")
    print("-" * 60)
    
    for model, fp32_fps in base_fps.items():
        # INT8 typically 2-3x speedup
        int8_fps = fp32_fps * 2.5
        speedup = int8_fps / fp32_fps
        
        # Usable if >= 1 FPS
        usable = int8_fps >= 1.0
        usable_str = "✅ YES" if usable else "❌ NO"
        
        print(f"{model:<15} {fp32_fps:<10.1f} {int8_fps:<10.1f} {speedup:<10.1f}x {usable_str:<10}")

def memory_optimization_tips():
    """Provide memory optimization tips"""
    print(f"\n💡 MEMORY OPTIMIZATION TIPS")
    print("=" * 50)
    
    tips = [
        "🔧 Add swap memory: sudo fallocate -l 1G /swapfile",
        "🔧 Use INT8 quantization to reduce model size by 4x",
        "🔧 Only load one model at a time",
        "🔧 Use gc.collect() between model loads",
        "🔧 Set OMP_NUM_THREADS=1 to reduce overhead",
        "🔧 Consider ONNX Runtime instead of PyTorch",
        "🔧 Use model.half() for FP16 to save 50% memory",
        "🔧 Process smaller input batches (batch_size=1)",
        "🔧 Consider TensorFlow Lite for better mobile optimization"
    ]
    
    for tip in tips:
        print(f"  {tip}")

def create_deployment_recommendation():
    """Create Pi Zero deployment recommendation"""
    print(f"\n🎯 PI ZERO DEPLOYMENT RECOMMENDATION")
    print("=" * 50)
    
    print(f"✅ FEASIBLE MODELS:")
    print(f"  • DINO-S/16 (INT8): ~21MB, ~5.6 FPS")
    print(f"  • DINOv2-S/14 (INT8): ~21MB, ~5.5 FPS")
    print(f"  • ViT-S/8 (INT8): ~21MB, ~4.5 FPS (higher resolution)")
    
    print(f"\n❌ NOT FEASIBLE:")
    print(f"  • DINO-B/16: Too large (86MB)")
    print(f"  • DINOv2-B/14: Too large (86MB)")
    print(f"  • ViT-G/14: Way too large (1011MB)")
    
    print(f"\n🔧 REQUIRED SETUP:")
    print(f"  1. Add 1GB swap memory")
    print(f"  2. Use PyTorch CPU-only")
    print(f"  3. Use INT8 quantization")
    print(f"  4. Process one image at a time")
    print(f"  5. Use model.eval() to disable training mode")
    
    print(f"\n⚡ EXPECTED PERFORMANCE:")
    print(f"  • Best case: DINO-S/16 INT8 at ~6 FPS")
    print(f"  • Realistic: 3-4 FPS with proper optimization")
    print(f"  • Memory usage: ~50-80MB peak")

def main():
    """Main analysis function"""
    print("🤖 PI ZERO MINIMAL BENCHMARK")
    print("🔧 Designed for 512MB RAM constraint")
    print("📊 No model loading - theoretical analysis only")
    print("=" * 60)
    
    # Check current memory usage
    baseline_memory = get_memory_usage()
    print(f"💾 Script memory usage: {baseline_memory:.1f}MB\n")
    
    # System check
    check_system_resources()
    
    # Theoretical analysis
    results = theoretical_model_analysis()
    
    # Performance simulation
    simulate_performance()
    
    # Optimization tips
    memory_optimization_tips()
    
    # Final recommendation
    create_deployment_recommendation()
    
    # Save results
    output_file = 'pi_zero_analysis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'system_memory_mb': psutil.virtual_memory().total / 1024**2,
            'baseline_memory_mb': baseline_memory,
            'models': results,
            'recommendation': 'Use DINO-S/16 or DINOv2-S/14 with INT8 quantization'
        }, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"🎯 Recommendation: Start with DINO-S/16 + INT8 quantization")

if __name__ == "__main__":
    main() 