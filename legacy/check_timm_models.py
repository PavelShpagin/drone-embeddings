#!/usr/bin/env python3

import timm

print("🔍 Checking available DINO models in timm...")
print("=" * 60)

# Get all available models
all_models = timm.list_models()

# Filter for DINO models
dino_models = [model for model in all_models if 'dino' in model.lower()]

print(f"📊 Found {len(dino_models)} DINO models:")
print("-" * 40)

for model in sorted(dino_models):
    print(f"  {model}")

print("\n🔍 Checking for specific patterns...")

# Check for specific patterns
patterns = ['vit', 'dino', 'dinov2']
for pattern in patterns:
    matching = [model for model in all_models if pattern in model.lower()]
    print(f"\n📊 Models containing '{pattern}': {len(matching)}")
    for model in sorted(matching)[:10]:  # Show first 10
        print(f"  {model}")
    if len(matching) > 10:
        print(f"  ... and {len(matching) - 10} more")

print("\n✅ Model check complete!") 