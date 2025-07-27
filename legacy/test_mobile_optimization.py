#!/usr/bin/env python3
"""
Test PyTorch Mobile Optimization
Verify that mobile optimizations are actually working
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.mobile_optimizer import optimize_for_mobile
import time
import os

def test_mobile_optimization():
    print("Testing PyTorch Mobile Optimization")
    print("=" * 50)
    
    # Load a simple model
    print("Loading MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(1, 3, 224, 224)
    
    print("\n1. Testing original model...")
    with torch.no_grad():
        original_output = model(sample_input)
        print(f"   Original output shape: {original_output.shape}")
    
    print("\n2. Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, sample_input)
        traced_output = traced_model(sample_input)
        print(f"   Traced output shape: {traced_output.shape}")
        
        # Check if outputs match
        output_diff = torch.max(torch.abs(original_output - traced_output)).item()
        print(f"   Output difference: {output_diff:.2e}")
    
    print("\n3. Applying mobile optimization...")
    mobile_model = optimize_for_mobile(traced_model)
    
    with torch.no_grad():
        mobile_output = mobile_model(sample_input)
        print(f"   Mobile output shape: {mobile_output.shape}")
        
        # Check if outputs match
        mobile_diff = torch.max(torch.abs(original_output - mobile_output)).item()
        print(f"   Mobile vs original difference: {mobile_diff:.2e}")
    
    print("\n4. Comparing model sizes...")
    
    # Save models to check file sizes
    os.makedirs('temp_models', exist_ok=True)
    
    # Original model (state dict)
    torch.save(model.state_dict(), 'temp_models/original.pt')
    original_size = os.path.getsize('temp_models/original.pt') / (1024 * 1024)
    
    # Traced model
    traced_model.save('temp_models/traced.pt')
    traced_size = os.path.getsize('temp_models/traced.pt') / (1024 * 1024)
    
    # Mobile optimized model
    mobile_model.save('temp_models/mobile.pt')
    mobile_size = os.path.getsize('temp_models/mobile.pt') / (1024 * 1024)
    
    print(f"   Original model (state_dict): {original_size:.2f} MB")
    print(f"   Traced model: {traced_size:.2f} MB")
    print(f"   Mobile optimized model: {mobile_size:.2f} MB")
    
    if mobile_size < traced_size:
        compression_ratio = traced_size / mobile_size
        print(f"   Mobile optimization achieved {compression_ratio:.2f}x compression!")
    else:
        print(f"   Warning: Mobile model is not smaller than traced model")
    
    print("\n5. Performance comparison...")
    
    def benchmark_model(model, name, num_runs=100):
        times = []
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(sample_input)
            
            # Benchmark
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(sample_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        std_time = (sum([(t - avg_time) ** 2 for t in times]) / len(times)) ** 0.5
        
        print(f"   {name}: {avg_time:.2f} ± {std_time:.2f} ms")
        return avg_time
    
    original_time = benchmark_model(model, "Original model")
    traced_time = benchmark_model(traced_model, "Traced model  ")
    mobile_time = benchmark_model(mobile_model, "Mobile model  ")
    
    print(f"\n6. Results summary:")
    print(f"   Mobile vs Original: {mobile_time/original_time:.2f}x time")
    print(f"   Mobile vs Traced:   {mobile_time/traced_time:.2f}x time")
    
    if mobile_time < traced_time:
        speedup = traced_time / mobile_time
        print(f"   ✓ Mobile optimization achieved {speedup:.2f}x speedup!")
    else:
        print(f"   ⚠ Mobile optimization did not improve performance")
    
    # Cleanup
    import shutil
    shutil.rmtree('temp_models')
    
    print(f"\n{'✓ Mobile optimization test complete!' if mobile_size < traced_size else '⚠ Mobile optimization may not be working properly'}")

if __name__ == "__main__":
    test_mobile_optimization() 