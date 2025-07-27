#!/usr/bin/env python3

import torch
import timm
import time
import psutil
import os
import numpy as np
import gc

def get_model_size_mb(model):
    """Calculate model size in MB based on the actual model parameters."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def test_single_model(model_name):
    """Test a single model configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print("Loading model...")
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        
        # Get model info
        original_size = get_model_size_mb(model)
        original_params = sum(p.numel() for p in model.parameters())
        
        print(f"📊 Model Info:")
        print(f"  Parameters: {original_params:,}")
        print(f"  Actual Size: {original_size:.2f} MB")
        
        # Theoretical size (FP32: 4 bytes per parameter)
        theoretical_size = original_params * 4 / 1024**2
        print(f"  Theoretical Size: {theoretical_size:.2f} MB")
        print(f"  Size Ratio: {original_size/theoretical_size:.3f}")
        
        # Performance test
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Measure inference time
        model.eval()
        times = []
        print("Measuring performance...")
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measure
        for _ in range(20):
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        print(f"🚀 Performance:")
        print(f"  Inference Time: {avg_time * 1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        
        # Pi Zero feasibility
        feasible = fps >= 10.0 and original_size <= 100.0
        print(f"\n🎯 Pi Zero Feasible: {'✅ YES' if feasible else '❌ NO'}")
        
        return {
            'model_name': model_name,
            'size': original_size,
            'params': original_params,
            'fps': fps,
            'feasible': feasible
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🎯 Simple DINO Model Test")
    print("=" * 60)
    
    # Test a few key models
    test_models = [
        'dino_vits16',      # DINO Small
        'dino_vitb16',      # DINO Base
        'dinov2_vits14',    # DINOv2 Small
        'dinov2_vitb14',    # DINOv2 Base
    ]
    
    results = []
    
    for model_name in test_models:
        result = test_single_model(model_name)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Model':<20} {'Size(MB)':<10} {'FPS':<8} {'Feasible':<10}")
    print("-" * 50)
    
    for result in results:
        feasible_str = "✅" if result['feasible'] else "❌"
        print(f"{result['model_name']:<20} {result['size']:<10.2f} {result['fps']:<8.2f} {feasible_str:<10}")
    
    print("\n✅ Simple test complete!")

if __name__ == "__main__":
    main() 