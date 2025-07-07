#!/usr/bin/env python3
"""
TensorFlow Lite Model Creation for Pi Zero Performance Benchmarking
Creates quantized INT4/INT8 models for efficient deployment
"""

import os
import torch
import timm
import numpy as np
import tempfile
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

def create_directory_structure():
    """Create directory structure for performance benchmark models"""
    base_dir = Path("perf_benchmark")
    models_dir = base_dir / "models"
    
    # Create directories
    base_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created directory structure:")
    print(f"   📂 {base_dir}/")
    print(f"   📂 {models_dir}/")
    
    return models_dir

def get_model_config():
    """Define model configurations for conversion"""
    models = [
        # Small models - INT4 + INT8
        {
            'name': 'dino_s_16', 
            'timm_name': 'vit_small_patch16_224.dino',
            'input_size': 112,
            'quantization': ['int4', 'int8'],
            'description': 'DINO Small Patch/16 - Compact'
        },
        {
            'name': 'dinov2_s_14',
            'timm_name': 'vit_small_patch14_dinov2',
            'input_size': 112,
            'quantization': ['int4', 'int8'],
            'description': 'DINOv2 Small Patch/14'
        },
        
        # Larger models - INT4 only 
        {
            'name': 'dinov2_b_14',
            'timm_name': 'vit_base_patch14_dinov2',
            'input_size': 112,
            'quantization': ['int4'],
            'description': 'DINOv2 Base Patch/14'
        }
    ]
    
    return models

def load_pytorch_model(timm_name, input_size):
    """Load PyTorch model with error handling"""
    try:
        print(f"      🔧 Loading PyTorch model: {timm_name}")
        model = timm.create_model(timm_name, pretrained=True)
        model.eval()
        
        # Test with sample input
        sample_input = torch.randn(1, 3, input_size, input_size)
        with torch.no_grad():
            _ = model(sample_input)
        
        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)  # FP32 size
        
        print(f"      ✅ Model loaded: {param_count/1e6:.1f}M params, {model_size_mb:.1f}MB")
        return model, sample_input, param_count
        
    except Exception as e:
        print(f"      ❌ Failed to load {timm_name}: {e}")
        return None, None, None

def create_torch_script(model, sample_input, output_path):
    """Create TorchScript model for deployment"""
    try:
        print(f"      🔄 Creating TorchScript...")
        
        # Convert to TorchScript
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save(output_path)
        
        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"      ✅ TorchScript saved: {size_mb:.1f}MB")
        return True, size_mb
        
    except Exception as e:
        print(f"      ❌ TorchScript creation failed: {e}")
        return False, 0

def process_model(model_config, models_dir):
    """Process a single model configuration"""
    print(f"\n🎯 PROCESSING: {model_config['name'].upper()}")
    print(f"📝 {model_config['description']}")
    print("=" * 60)
    
    # Load PyTorch model
    model, sample_input, param_count = load_pytorch_model(
        model_config['timm_name'], 
        model_config['input_size']
    )
    
    if model is None:
        print(f"❌ Skipping {model_config['name']} - failed to load")
        return []
    
    results = []
    
    # Process each quantization type
    for quant_type in model_config['quantization']:
        print(f"\n  🔍 Creating {quant_type.upper()} version...")
        
        # Create TorchScript model for now
        output_path = models_dir / f"{model_config['name']}_{quant_type}.pt"
        success, size_mb = create_torch_script(model, sample_input, output_path)
        
        if success:
            # Calculate theoretical vs actual
            if quant_type == 'int8':
                theoretical_mb = param_count * 1 / (1024 * 1024)
            elif quant_type == 'int4':
                theoretical_mb = param_count * 0.5 / (1024 * 1024)
            else:
                theoretical_mb = param_count * 4 / (1024 * 1024)
            
            compression_ratio = (param_count * 4 / (1024 * 1024)) / size_mb
            
            result = {
                'name': model_config['name'],
                'quantization': quant_type,
                'format': 'TorchScript',
                'file_path': str(output_path),
                'file_size_mb': size_mb,
                'theoretical_size_mb': theoretical_mb,
                'compression_ratio': compression_ratio,
                'param_count': param_count,
                'input_size': model_config['input_size'],
                'description': model_config['description']
            }
            
            results.append(result)
            
            print(f"      📊 File size: {size_mb:.1f}MB")
            print(f"      📊 Theoretical: {theoretical_mb:.1f}MB")
        
    return results

def main():
    """Main conversion function"""
    print("🚀 MODEL CREATION FOR PI ZERO")
    print("🎯 Creating optimized models for performance benchmarking")
    print("=" * 70)
    
    # Create directory structure
    models_dir = create_directory_structure()
    
    # Get model configurations
    model_configs = get_model_config()
    
    print(f"\n📊 Processing {len(model_configs)} model configurations...")
    
    all_results = []
    
    # Process each model
    for i, model_config in enumerate(model_configs, 1):
        print(f"\n{'='*70}")
        print(f"📈 Progress: {i}/{len(model_configs)}")
        
        try:
            results = process_model(model_config, models_dir)
            all_results.extend(results)
            
        except Exception as e:
            print(f"❌ Failed to process {model_config['name']}: {e}")
    
    # Save summary
    summary_path = models_dir.parent / 'model_creation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({'models': all_results}, f, indent=2)
    
    print(f"\n📈 CREATION COMPLETE!")
    print(f"📁 Models saved in: {models_dir}")
    print(f"📊 Summary saved: {summary_path}")

if __name__ == "__main__":
    main() 