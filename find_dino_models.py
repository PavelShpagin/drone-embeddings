#!/usr/bin/env python3

import timm
import torch

print("🔍 Finding all working DINO model names...")
print("=" * 60)

# Get all models
all_models = timm.list_models()

# Search patterns
search_patterns = ['dino', 'dinov2', 'vit']

working_models = []

for pattern in search_patterns:
    matching_models = [m for m in all_models if pattern.lower() in m.lower()]
    
    print(f"\n📊 Testing models containing '{pattern}' ({len(matching_models)} found):")
    print("-" * 50)
    
    for model_name in matching_models[:15]:  # Test first 15 of each pattern
        try:
            model = timm.create_model(model_name, pretrained=False)  # Don't download weights yet
            
            # Get model info
            total_params = sum(p.numel() for p in model.parameters())
            
            # Check if it looks like a DINO model (Vision Transformer with reasonable size)
            if 10e6 <= total_params <= 2e9:  # Between 10M and 2B parameters
                
                # Try to load with pretrained weights
                try:
                    model_pretrained = timm.create_model(model_name, pretrained=True)
                    param_size = sum(p.numel() * p.element_size() for p in model_pretrained.parameters())
                    size_mb = param_size / (1024 * 1024)
                    
                    print(f"  ✅ {model_name}: {total_params/1e6:.1f}M params, {size_mb:.1f}MB")
                    working_models.append({
                        'name': model_name,
                        'params': total_params,
                        'size_mb': size_mb
                    })
                    
                except Exception as e:
                    print(f"  ⚠️  {model_name}: Works but no pretrained weights ({total_params/1e6:.1f}M params)")
                    
        except Exception as e:
            continue

print(f"\n🏆 WORKING DINO MODELS ({len(working_models)} found):")
print("=" * 60)

# Sort by parameter count
working_models.sort(key=lambda x: x['params'])

for model in working_models:
    size_category = "Small" if model['params'] < 50e6 else "Medium" if model['params'] < 200e6 else "Large"
    print(f"  {model['name']:<40} | {model['params']/1e6:>6.1f}M | {model['size_mb']:>6.1f}MB | {size_category}")

print(f"\n✅ Search complete! Found {len(working_models)} working DINO models.")

# Generate updated model dict
print(f"\n🔧 SUGGESTED MODEL DICTIONARY:")
print("=" * 60)
print("models = {")

# Try to categorize the models
for model in working_models:
    name = model['name']
    if 'small' in name and 'patch16' in name and 'dino' in name and 'dinov2' not in name:
        print(f"    'DINO-S/16': lambda: timm.create_model('{name}', pretrained=True),")
    elif 'small' in name and 'patch8' in name and 'dino' in name and 'dinov2' not in name:
        print(f"    'DINO-S/8': lambda: timm.create_model('{name}', pretrained=True),")
    elif 'base' in name and 'patch16' in name and 'dino' in name and 'dinov2' not in name:
        print(f"    'DINO-B/16': lambda: timm.create_model('{name}', pretrained=True),")
    elif 'small' in name and 'dinov2' in name:
        print(f"    'DINOv2-S/14': lambda: timm.create_model('{name}', pretrained=True),")
    elif 'base' in name and 'dinov2' in name:
        print(f"    'DINOv2-B/14': lambda: timm.create_model('{name}', pretrained=True),")
    elif 'large' in name and 'dinov2' in name:
        print(f"    'DINOv2-L/14': lambda: timm.create_model('{name}', pretrained=True),")
    elif 'giant' in name and 'dinov2' in name:
        print(f"    'DINOv2-G/14': lambda: timm.create_model('{name}', pretrained=True),")

print("}")

print(f"\n🎯 Copy the working model names into the benchmark script!") 