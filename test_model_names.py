#!/usr/bin/env python3

import timm
import torch

print("🔍 Testing DINO model names...")
print("=" * 50)

# Test model names
model_names = {
    'DINO-S/16': 'vit_small_patch16_224.dino',
    'DINO-S/8': 'vit_small_patch8_224.dino', 
    'DINO-B/16': 'vit_base_patch16_224.dino',
    'DINO-B/8': 'vit_base_patch8_224.dino',
    'DINOv2-S/14': 'vit_small_patch14_dinov2.lvd142m',
    'DINOv2-B/14': 'vit_base_patch14_dinov2.lvd142m',
    'DINOv2-L/14': 'vit_large_patch14_dinov2.lvd142m',
    'DINOv2-G/14': 'vit_giant_patch14_dinov2.lvd142m',
}

for display_name, model_name in model_names.items():
    try:
        print(f"Testing {display_name}: {model_name}")
        model = timm.create_model(model_name, pretrained=True)
        
        # Get basic info
        total_params = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        size_mb = param_size / (1024 * 1024)
        
        print(f"  ✅ SUCCESS: {total_params/1e6:.1f}M params, {size_mb:.1f}MB")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  ✅ Inference works: output shape {output.shape}")
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
    
    print()

print("✅ Model name test complete!") 