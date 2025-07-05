#!/usr/bin/env python3
"""
Optimized Model Profiler for Raspberry Pi Zero
==============================================

This script profiles both regular PyTorch models and optimized formats:
- Regular PyTorch models
- ONNX models (with and without quantization)
- Quantized PyTorch models
- PyTorch Mobile models

Measures:
- Model size (MB)
- Inference time (ms)
- Memory footprint (MB)
- FLOPs (GFLOPs)
- Parameters (M)
- Throughput (FPS)

Requirements:
    pip install torch torchvision timm onnx onnxruntime pandas matplotlib seaborn thop psutil tqdm
"""

import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import psutil
import thop
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import seaborn as sns
import sys
import json
import logging

# Optional imports for optimized models
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX/ONNXRuntime not available. Install with: pip install onnx onnxruntime")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedModelProfiler:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models dictionary
        self.models = {}
        self.onnx_models = {}
        self.quantized_models = {}
        self.mobile_models = {}
        
        # Load all available models
        self.load_all_models()
        
    def load_all_models(self):
        """Load all models for profiling."""
        logger.info("Loading models...")
        
        # Original PyTorch models
        self.load_pytorch_models()
        
        # Load optimized models if available
        self.load_optimized_models()
        
    def load_pytorch_models(self):
        """Load regular PyTorch models."""
        logger.info("Loading PyTorch models...")
        
        try:
            # EfficientNet-B0
            self.models['efficientnet_b0'] = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            
            # MobileNetV2
            self.models['mobilenet_v2'] = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            
            # ResNet-50
            self.models['resnet50'] = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            # MobileNetV4 (using timm)
            try:
                self.models['mobilenet_v4'] = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)
            except Exception as e:
                logger.warning(f"Could not load MobileNetV4: {e}")
            
            # DINOv2 (using timm)
            try:
                self.models['dinov2'] = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
            except Exception as e:
                logger.warning(f"Could not load DINOv2: {e}")
            
            # DINO (using timm)
            try:
                self.models['dino'] = timm.create_model('dino_vitbase16_224', pretrained=True)
            except Exception as e:
                logger.warning(f"Could not load DINO: {e}")
                
        except Exception as e:
            logger.error(f"Error loading PyTorch models: {e}")
        
        # Move all models to device and set to eval mode
        for name, model in self.models.items():
            self.models[name] = model.to(self.device).eval()
            
    def load_optimized_models(self):
        """Load optimized models from the optimized_models directory."""
        logger.info("Loading optimized models...")
        
        optimized_dir = Path("optimized_models")
        if not optimized_dir.exists():
            logger.warning("Optimized models directory not found. Run convert_models_for_pi.py first.")
            return
        
        # Load ONNX models
        self.load_onnx_models(optimized_dir / "onnx")
        
        # Load quantized PyTorch models
        self.load_quantized_pytorch_models(optimized_dir / "quantized_pytorch")
        
        # Load PyTorch Mobile models
        self.load_mobile_models(optimized_dir / "pytorch_mobile")
        
    def load_onnx_models(self, onnx_dir: Path):
        """Load ONNX models."""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available, skipping ONNX models")
            return
            
        if not onnx_dir.exists():
            logger.warning(f"ONNX directory not found: {onnx_dir}")
            return
            
        logger.info("Loading ONNX models...")
        
        for onnx_file in onnx_dir.glob("*.onnx"):
            try:
                # Create inference session
                session = ort.InferenceSession(str(onnx_file), providers=['CPUExecutionProvider'])
                model_name = onnx_file.stem
                self.onnx_models[model_name] = session
                logger.info(f"Loaded ONNX model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading ONNX model {onnx_file}: {e}")
                
    def load_quantized_pytorch_models(self, quantized_dir: Path):
        """Load quantized PyTorch models."""
        if not quantized_dir.exists():
            logger.warning(f"Quantized PyTorch directory not found: {quantized_dir}")
            return
            
        logger.info("Loading quantized PyTorch models...")
        
        for quantized_file in quantized_dir.glob("*.pt"):
            try:
                # We need to reconstruct the model architecture and load weights
                model_name = quantized_file.stem.replace("_quantized", "")
                
                # Skip for now - would need to save the quantized model properly
                logger.info(f"Found quantized model: {model_name} (reconstruction needed)")
                
            except Exception as e:
                logger.error(f"Error loading quantized model {quantized_file}: {e}")
                
    def load_mobile_models(self, mobile_dir: Path):
        """Load PyTorch Mobile models."""
        if not mobile_dir.exists():
            logger.warning(f"PyTorch Mobile directory not found: {mobile_dir}")
            return
            
        logger.info("Loading PyTorch Mobile models...")
        
        for mobile_file in mobile_dir.glob("*.pt"):
            try:
                # Load mobile model
                mobile_model = torch.jit.load(str(mobile_file))
                mobile_model.eval()
                model_name = mobile_file.stem.replace("_mobile", "")
                self.mobile_models[model_name] = mobile_model
                logger.info(f"Loaded PyTorch Mobile model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading mobile model {mobile_file}: {e}")

    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
        
    def get_file_size(self, file_path: str) -> float:
        """Get file size in MB."""
        return os.path.getsize(file_path) / (1024 ** 2)
        
    def measure_inference_time(self, model: Any, input_tensor: torch.Tensor, 
                             num_runs: int = 100, model_type: str = 'pytorch') -> Tuple[float, float]:
        """Measure average and std of inference time."""
        times = []
        
        if model_type == 'pytorch':
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(input_tensor)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
        elif model_type == 'onnx':
            # Convert tensor to numpy for ONNX
            input_np = input_tensor.cpu().numpy()
            input_name = model.get_inputs()[0].name
            
            for _ in range(num_runs):
                start_time = time.time()
                _ = model.run(None, {input_name: input_np})
                end_time = time.time()
                times.append(end_time - start_time)
                
        elif model_type == 'mobile':
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(input_tensor)
                    end_time = time.time()
                    times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)
        
    def measure_memory_usage(self, model: Any, input_tensor: torch.Tensor, model_type: str = 'pytorch') -> float:
        """Measure peak memory usage during inference."""
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        # Get baseline memory
        baseline_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        
        # Run inference
        if model_type == 'pytorch':
            with torch.no_grad():
                _ = model(input_tensor)
        elif model_type == 'onnx':
            input_np = input_tensor.cpu().numpy()
            input_name = model.get_inputs()[0].name
            _ = model.run(None, {input_name: input_np})
        elif model_type == 'mobile':
            with torch.no_grad():
                _ = model(input_tensor)
            
        # Get peak memory
        if self.device == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2) - baseline_memory
            
        return peak_memory
        
    def calculate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> Tuple[float, float]:
        """Calculate FLOPs and parameters for PyTorch models."""
        try:
            flops, params = thop.profile(model, inputs=(input_tensor,), verbose=False)
            return flops / 1e9, params / 1e6  # Convert to GFLOPs and M params
        except Exception as e:
            logger.warning(f"Could not calculate FLOPs: {e}")
            return 0.0, 0.0
        
    def profile_pytorch_model(self, model_name: str, model: nn.Module, sample_input: torch.Tensor) -> Dict:
        """Profile a PyTorch model."""
        # Basic measurements
        model_size = self.get_model_size(model)
        inference_time, time_std = self.measure_inference_time(model, sample_input, model_type='pytorch')
        memory_usage = self.measure_memory_usage(model, sample_input, model_type='pytorch')
        flops, params = self.calculate_flops(model, sample_input)
        
        return {
            'Model Type': 'PyTorch',
            'Model Size (MB)': model_size,
            'Inference Time (ms)': inference_time * 1000,
            'Time Std (ms)': time_std * 1000,
            'Peak Memory (MB)': memory_usage,
            'GFLOPs': flops,
            'Parameters (M)': params,
            'Throughput (FPS)': 1.0 / inference_time if inference_time > 0 else 0
        }
        
    def profile_onnx_model(self, model_name: str, model: Any, sample_input: torch.Tensor) -> Dict:
        """Profile an ONNX model."""
        # Get model file size
        onnx_dir = Path("optimized_models/onnx")
        model_file = onnx_dir / f"{model_name}.onnx"
        model_size = self.get_file_size(str(model_file)) if model_file.exists() else 0
        
        # Measure inference time
        inference_time, time_std = self.measure_inference_time(model, sample_input, model_type='onnx')
        memory_usage = self.measure_memory_usage(model, sample_input, model_type='onnx')
        
        return {
            'Model Type': 'ONNX',
            'Model Size (MB)': model_size,
            'Inference Time (ms)': inference_time * 1000,
            'Time Std (ms)': time_std * 1000,
            'Peak Memory (MB)': memory_usage,
            'GFLOPs': 0,  # Hard to calculate for ONNX
            'Parameters (M)': 0,  # Hard to calculate for ONNX
            'Throughput (FPS)': 1.0 / inference_time if inference_time > 0 else 0
        }
        
    def profile_mobile_model(self, model_name: str, model: Any, sample_input: torch.Tensor) -> Dict:
        """Profile a PyTorch Mobile model."""
        # Get model file size
        mobile_dir = Path("optimized_models/pytorch_mobile")
        model_file = mobile_dir / f"{model_name}_mobile.pt"
        model_size = self.get_file_size(str(model_file)) if model_file.exists() else 0
        
        # Measure inference time
        inference_time, time_std = self.measure_inference_time(model, sample_input, model_type='mobile')
        memory_usage = self.measure_memory_usage(model, sample_input, model_type='mobile')
        
        return {
            'Model Type': 'PyTorch Mobile',
            'Model Size (MB)': model_size,
            'Inference Time (ms)': inference_time * 1000,
            'Time Std (ms)': time_std * 1000,
            'Peak Memory (MB)': memory_usage,
            'GFLOPs': 0,  # Hard to calculate for mobile
            'Parameters (M)': 0,  # Hard to calculate for mobile
            'Throughput (FPS)': 1.0 / inference_time if inference_time > 0 else 0
        }
        
    def profile_all_models(self, sample_input: torch.Tensor) -> pd.DataFrame:
        """Profile all models and return results as DataFrame."""
        results = {}
        
        # Profile PyTorch models
        logger.info("Profiling PyTorch models...")
        for model_name, model in tqdm(self.models.items(), desc="PyTorch models"):
            try:
                results[f"{model_name}_pytorch"] = self.profile_pytorch_model(model_name, model, sample_input)
            except Exception as e:
                logger.error(f"Error profiling PyTorch model {model_name}: {e}")
                
        # Profile ONNX models
        logger.info("Profiling ONNX models...")
        for model_name, model in tqdm(self.onnx_models.items(), desc="ONNX models"):
            try:
                results[f"{model_name}_onnx"] = self.profile_onnx_model(model_name, model, sample_input)
            except Exception as e:
                logger.error(f"Error profiling ONNX model {model_name}: {e}")
                
        # Profile PyTorch Mobile models
        logger.info("Profiling PyTorch Mobile models...")
        for model_name, model in tqdm(self.mobile_models.items(), desc="Mobile models"):
            try:
                results[f"{model_name}_mobile"] = self.profile_mobile_model(model_name, model, sample_input)
            except Exception as e:
                logger.error(f"Error profiling Mobile model {model_name}: {e}")
                
        return pd.DataFrame.from_dict(results, orient='index')
        
    def plot_results(self, results: pd.DataFrame, save_path: str = 'optimized_model_profiling_results.png'):
        """Plot profiling results with comparison between model types."""
        if results.empty:
            logger.warning("No results to plot")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Optimized Model Profiling Results', fontsize=16)
        
        # Plot 1: Model Size Comparison
        ax1 = axes[0, 0]
        size_data = results[['Model Size (MB)', 'Model Type']].copy()
        size_data['Model Name'] = [name.split('_')[0] for name in size_data.index]
        
        # Group by model type
        for model_type in size_data['Model Type'].unique():
            subset = size_data[size_data['Model Type'] == model_type]
            ax1.bar(subset['Model Name'], subset['Model Size (MB)'], 
                   alpha=0.7, label=model_type)
        
        ax1.set_title('Model Size Comparison (MB)')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Size (MB)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Inference Time Comparison
        ax2 = axes[0, 1]
        time_data = results[['Inference Time (ms)', 'Model Type']].copy()
        time_data['Model Name'] = [name.split('_')[0] for name in time_data.index]
        
        for model_type in time_data['Model Type'].unique():
            subset = time_data[time_data['Model Type'] == model_type]
            ax2.bar(subset['Model Name'], subset['Inference Time (ms)'], 
                   alpha=0.7, label=model_type)
        
        ax2.set_title('Inference Time Comparison (ms)')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Memory Usage Comparison
        ax3 = axes[1, 0]
        memory_data = results[['Peak Memory (MB)', 'Model Type']].copy()
        memory_data['Model Name'] = [name.split('_')[0] for name in memory_data.index]
        
        for model_type in memory_data['Model Type'].unique():
            subset = memory_data[memory_data['Model Type'] == model_type]
            ax3.bar(subset['Model Name'], subset['Peak Memory (MB)'], 
                   alpha=0.7, label=model_type)
        
        ax3.set_title('Peak Memory Usage Comparison (MB)')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Peak Memory (MB)')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Throughput Comparison
        ax4 = axes[1, 1]
        throughput_data = results[['Throughput (FPS)', 'Model Type']].copy()
        throughput_data['Model Name'] = [name.split('_')[0] for name in throughput_data.index]
        
        for model_type in throughput_data['Model Type'].unique():
            subset = throughput_data[throughput_data['Model Type'] == model_type]
            ax4.bar(subset['Model Name'], subset['Throughput (FPS)'], 
                   alpha=0.7, label=model_type)
        
        ax4.set_title('Throughput Comparison (FPS)')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Throughput (FPS)')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to: {save_path}")
        
    def create_summary_report(self, results: pd.DataFrame, save_path: str = 'optimization_summary.json'):
        """Create a summary report of optimization results."""
        if results.empty:
            logger.warning("No results to summarize")
            return
            
        summary = {
            'total_models': len(results),
            'model_types': results['Model Type'].value_counts().to_dict(),
            'best_performers': {},
            'optimization_gains': {}
        }
        
        # Find best performers in each category
        metrics = ['Model Size (MB)', 'Inference Time (ms)', 'Peak Memory (MB)', 'Throughput (FPS)']
        
        for metric in metrics:
            if metric in results.columns:
                if metric == 'Throughput (FPS)':
                    # Higher is better
                    best_idx = results[metric].idxmax()
                    best_value = results[metric].max()
                else:
                    # Lower is better
                    best_idx = results[metric].idxmin()
                    best_value = results[metric].min()
                
                summary['best_performers'][metric] = {
                    'model': best_idx,
                    'value': best_value
                }
        
        # Calculate optimization gains
        base_models = [name for name in results.index if name.endswith('_pytorch')]
        
        for base_model in base_models:
            base_name = base_model.replace('_pytorch', '')
            optimized_variants = [name for name in results.index if name.startswith(base_name) and not name.endswith('_pytorch')]
            
            if optimized_variants:
                base_stats = results.loc[base_model]
                gains = {}
                
                for variant in optimized_variants:
                    variant_stats = results.loc[variant]
                    variant_type = variant.split('_')[-1]
                    
                    # Calculate percentage improvements
                    size_improvement = (base_stats['Model Size (MB)'] - variant_stats['Model Size (MB)']) / base_stats['Model Size (MB)'] * 100
                    time_improvement = (base_stats['Inference Time (ms)'] - variant_stats['Inference Time (ms)']) / base_stats['Inference Time (ms)'] * 100
                    memory_improvement = (base_stats['Peak Memory (MB)'] - variant_stats['Peak Memory (MB)']) / base_stats['Peak Memory (MB)'] * 100
                    
                    gains[variant_type] = {
                        'size_improvement_percent': size_improvement,
                        'time_improvement_percent': time_improvement,
                        'memory_improvement_percent': memory_improvement
                    }
                
                summary['optimization_gains'][base_name] = gains
        
        # Save summary
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to: {save_path}")
        return summary

def process_satellite_image(image_path: str, crop_size: int = 100) -> List[Image.Image]:
    """Process satellite image into crops."""
    image = Image.open(image_path)
    crops = []
    
    width, height = image.size
    for i in range(0, height - crop_size, crop_size):
        for j in range(0, width - crop_size, crop_size):
            crop = image.crop((j, i, j + crop_size, i + crop_size))
            crops.append(crop)
            
    return crops

def main():
    """Main function to run the profiling."""
    logger.info("Starting optimized model profiling...")
    
    # Initialize profiler
    profiler = OptimizedModelProfiler()
    
    # Define the path to the specific test image
    test_image_path = Path('data/test/test.jpg')
    
    if not test_image_path.exists():
        logger.error(f"Test image not found at: {test_image_path}")
        # Create a dummy input instead
        logger.info("Creating dummy input for profiling...")
        dummy_input = torch.randn(1, 3, 224, 224)
        if profiler.device == 'cuda':
            dummy_input = dummy_input.cuda()
        sample_input = dummy_input
    else:
        logger.info(f"Processing image: {test_image_path}")
        
        # Process the test image into crops
        crops = process_satellite_image(str(test_image_path), crop_size=100)
        
        if not crops:
            logger.error("No crops were generated from the image.")
            return
            
        logger.info(f"Generated {len(crops)} crops of size 100x100 pixels.")
        
        # Use the first crop as the sample input
        sample_input = profiler.transform(crops[0]).unsqueeze(0)
        if profiler.device == 'cuda':
            sample_input = sample_input.cuda()
    
    # Profile all models
    logger.info("Starting model profiling...")
    results = profiler.profile_all_models(sample_input)
    
    if results.empty:
        logger.error("No models were profiled successfully.")
        return
    
    # Save results
    results.to_csv('optimized_model_profiling_results.csv')
    logger.info("Results saved to optimized_model_profiling_results.csv")
    
    # Create plots
    profiler.plot_results(results)
    
    # Create summary report
    summary = profiler.create_summary_report(results)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZED MODEL PROFILING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total models profiled: {len(results)}")
    logger.info(f"Model types: {dict(results['Model Type'].value_counts())}")
    
    if summary and 'best_performers' in summary:
        logger.info("\nBest performers:")
        for metric, info in summary['best_performers'].items():
            logger.info(f"  {metric}: {info['model']} ({info['value']:.2f})")
    
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS FOR RASPBERRY PI ZERO:")
    logger.info("1. Focus on models with lowest inference time and memory usage")
    logger.info("2. ONNX quantized models typically offer best size/speed trade-off")
    logger.info("3. PyTorch Mobile models are easiest to integrate")
    logger.info("4. Consider model accuracy vs. speed trade-offs for your use case")
    logger.info("="*80)

if __name__ == "__main__":
    main()
