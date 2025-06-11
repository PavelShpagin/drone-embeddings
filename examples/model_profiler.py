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
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns
import sys

class ModelProfiler:
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
        self.load_all_models()
        
    def load_all_models(self):
        """Load all models for profiling."""
        print("Loading models...")
        
        # ShuffleNet
        self.models['shufflenet'] = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        
        # MobileNetV2
        self.models['mobilenetv2'] = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # MobileNetV3
        self.models['mobilenetv3'] = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # EfficientNet-B0
        self.models['efficientnet'] = timm.create_model('efficientnet_b0', pretrained=True)
        
        # ResNet-50
        self.models['resnet50'] = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # AnyLoc
        try:
            from models.anyloc.configs.default import get_default_config
            from models.anyloc.models.model_factory import create_model
            
            cfg = get_default_config()
            cfg.model_name = 'dinov2' # AnyLoc's best model uses DINOv2
            self.models['anyloc'] = create_model(cfg).to(self.device)
            print("Loaded AnyLoc with DINOv2 backbone.")
        except ImportError:
            print("Could not import AnyLoc, skipping.")
        except Exception as e:
            print(f"Error loading AnyLoc: {str(e)}, skipping.")

        # TransGEO
        try:
            from models.transgeo.model import GeoLocalizationNet
            transgeo_model = GeoLocalizationNet(backbone='resnet50', fc_output_dim=2048)
            # TransGEO requires manual weight loading, which is out of scope for this profiler.
            # We will profile the architecture without pre-trained weights.
            self.models['transgeo'] = transgeo_model
            print("Loaded TransGEO with ResNet-50 backbone (uninitialized).")
        except ImportError:
            print("Could not import TransGEO, skipping.")
        except Exception as e:
            print(f"Error loading TransGEO: {str(e)}, skipping.")
        
        # Move all models to device and set to eval mode
        for name, model in self.models.items():
            self.models[name] = model.to(self.device).eval()
            
    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
        
    def measure_inference_time(self, model: nn.Module, input_tensor: torch.Tensor, 
                             num_runs: int = 100) -> Tuple[float, float]:
        """Measure average and std of inference time."""
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)
        
    def measure_memory_usage(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """Measure peak memory usage during inference."""
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        # Get baseline memory
        baseline_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        
        # Run inference
        with torch.no_grad():
            _ = model(input_tensor)
            
        # Get peak memory
        if self.device == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2 - baseline_memory
            
        return peak_memory
        
    def calculate_flops(self, model: nn.Module, input_tensor: torch.Tensor) -> Tuple[float, float]:
        """Calculate FLOPs and parameters."""
        flops, params = thop.profile(model, inputs=(input_tensor,))
        return flops / 1e9, params / 1e6  # Convert to GFLOPs and M params
        
    def profile_model(self, model_name: str, sample_input: torch.Tensor) -> Dict:
        """Profile a single model."""
        model = self.models[model_name]
        
        # Basic measurements
        model_size = self.get_model_size(model)
        inference_time, time_std = self.measure_inference_time(model, sample_input)
        memory_usage = self.measure_memory_usage(model, sample_input)
        flops, params = self.calculate_flops(model, sample_input)
        
        return {
            'Model Size (MB)': model_size,
            'Inference Time (ms)': inference_time * 1000,  # Convert to ms
            'Time Std (ms)': time_std * 1000,
            'Peak Memory (MB)': memory_usage,
            'GFLOPs': flops,
            'Parameters (M)': params
        }
        
    def profile_all_models(self, sample_input: torch.Tensor) -> pd.DataFrame:
        """Profile all models and return results as DataFrame."""
        results = {}
        for model_name in tqdm(self.models.keys(), desc="Profiling models"):
            try:
                results[model_name] = self.profile_model(model_name, sample_input)
            except Exception as e:
                print(f"Error profiling {model_name}: {str(e)}")
                
        return pd.DataFrame.from_dict(results, orient='index')
        
    def plot_results(self, results: pd.DataFrame, save_path: str = 'model_profiling_results.png'):
        """Plot profiling results."""
        metrics = list(results.columns)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        fig.suptitle('Model Profiling Results', fontsize=16)
        
        for i, metric in enumerate(metrics):
            sns.barplot(x=results.index, y=results[metric], ax=axes[i])
            axes[i].set_title(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

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
    # Initialize profiler
    profiler = ModelProfiler()
    
    # Define the path to the specific test image
    test_image_path = Path('data/test/test.jpg')
    
    if not test_image_path.exists():
        print(f"Test image not found at: {test_image_path}")
        return
        
    print(f"Processing image: {test_image_path}")
    
    # Process the test image into 100x100 crops
    crops = process_satellite_image(str(test_image_path), crop_size=100)
    
    if not crops:
        print("No crops were generated from the image. Please check image dimensions and crop size.")
        return
        
    print(f"Generated {len(crops)} crops of size 100x100 pixels.")
    
    # Use the first crop as the sample input for profiling
    # The profiler's transform will resize it to 224x224 as required by the models
    sample_input = profiler.transform(crops[0]).unsqueeze(0).to(profiler.device)
    
    # Profile all loaded models
    results = profiler.profile_all_models(sample_input)
    
    # Save results
    results.to_csv('model_profiling_results.csv')
    print("\nResults saved to model_profiling_results.csv")
    
    # Plot results
    profiler.plot_results(results)
    print("Plots saved to model_profiling_results.png")
    
    # Print summary
    print("\nProfiling Summary:")
    print(results.round(2))

if __name__ == "__main__":
    main()