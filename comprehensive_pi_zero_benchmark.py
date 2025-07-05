#!/usr/bin/env python3
"""
Comprehensive Pi Zero Model Benchmark
Measures actual performance metrics including FPS, inference speed, and RAM usage
Compares quantized vs non-quantized models with real measurements
"""

import torch
import torchvision.models as models
import timm
import time
import psutil
import os
import gc
import numpy as np
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import threading
import onnx
import onnxruntime as ort

class MemoryMonitor:
    """Monitor actual RAM usage in real-time"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = 0
        self.peak_memory = 0
        self.monitoring = False
        self.memory_samples = []
        
    def start_monitoring(self):
        """Start monitoring memory usage"""
        self.baseline_memory = self.get_current_memory()
        self.peak_memory = self.baseline_memory
        self.monitoring = True
        self.memory_samples = []
        
        def monitor():
            while self.monitoring:
                current = self.get_current_memory()
                self.memory_samples.append(current)
                if current > self.peak_memory:
                    self.peak_memory = current
                time.sleep(0.01)  # Sample every 10ms
                
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return stats"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
            
        return {
            'baseline_mb': self.baseline_memory,
            'peak_mb': self.peak_memory,
            'delta_mb': self.peak_memory - self.baseline_memory,
            'avg_mb': np.mean(self.memory_samples) if self.memory_samples else self.baseline_memory,
            'samples': len(self.memory_samples)
        }
        
    def get_current_memory(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

class ModelBenchmark:
    """Comprehensive model benchmarking with actual measurements"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.memory_monitor = MemoryMonitor()
        self.results = {}
        
    def calculate_theoretical_memory(self, model):
        """Calculate theoretical memory usage"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        total_bytes = param_size + buffer_size
        return {
            'params_mb': param_size / 1024 / 1024,
            'buffers_mb': buffer_size / 1024 / 1024,
            'total_mb': total_bytes / 1024 / 1024,
            'total_params': sum(p.numel() for p in model.parameters())
        }
    
    def measure_model_loading_memory(self, model_loader_func, model_name):
        """Measure actual memory usage during model loading"""
        print(f"📊 Measuring memory for {model_name}...")
        
        # Clear memory and get baseline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.memory_monitor.start_monitoring()
        
        # Load model
        start_time = time.time()
        model = model_loader_func()
        model = model.to(self.device).eval()
        load_time = time.time() - start_time
        
        # Let memory stabilize
        time.sleep(0.5)
        memory_stats = self.memory_monitor.stop_monitoring()
        
        # Calculate theoretical memory
        theoretical = self.calculate_theoretical_memory(model)
        
        return {
            'model': model,
            'load_time': load_time,
            'actual_memory': memory_stats,
            'theoretical_memory': theoretical,
            'memory_efficiency': theoretical['total_mb'] / memory_stats['delta_mb'] if memory_stats['delta_mb'] > 0 else 0
        }
    
    def measure_inference_performance(self, model, model_name, input_tensor, num_warmup=10, num_runs=100):
        """Measure actual inference performance and FPS"""
        print(f"🚀 Benchmarking inference for {model_name}...")
        
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)
        
        # Measure inference times
        inference_times = []
        
        self.memory_monitor.start_monitoring()
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                output = model(input_tensor)
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
        
        inference_memory = self.memory_monitor.stop_monitoring()
        
        # Calculate statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        
        fps = 1.0 / avg_inference_time
        
        return {
            'avg_inference_ms': avg_inference_time * 1000,
            'std_inference_ms': std_inference_time * 1000,
            'min_inference_ms': min_inference_time * 1000,
            'max_inference_ms': max_inference_time * 1000,
            'fps': fps,
            'throughput_samples_per_second': fps,
            'inference_memory': inference_memory,
            'total_runs': num_runs
        }
    
    def create_quantized_model(self, model, model_name):
        """Create quantized version of the model"""
        print(f"⚡ Creating quantized version of {model_name}...")
        
        try:
            # Prepare model for quantization
            model.eval()
            
            # Create quantized model
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear, torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
            
            return quantized_model
        except Exception as e:
            print(f"❌ Quantization failed for {model_name}: {e}")
            return None
    
    def create_mobile_optimized_model(self, model, model_name, input_tensor):
        """Create mobile-optimized version of the model"""
        print(f"📱 Creating mobile-optimized version of {model_name}...")
        
        try:
            model.eval()
            
            # Trace the model
            traced_model = torch.jit.trace(model, input_tensor)
            
            # Optimize for mobile
            mobile_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            
            return mobile_model
        except Exception as e:
            print(f"❌ Mobile optimization failed for {model_name}: {e}")
            return None
    
    def benchmark_model_variants(self, base_model, model_name, input_tensor):
        """Benchmark all variants of a model"""
        print(f"\n🔬 Comprehensive benchmark for {model_name}")
        print("=" * 60)
        
        variants = {}
        
        # 1. Original model
        print("1️⃣ Original model...")
        original_stats = self.measure_model_loading_memory(lambda: base_model, f"{model_name}_original")
        original_perf = self.measure_inference_performance(
            original_stats['model'], f"{model_name}_original", input_tensor
        )
        
        variants['original'] = {
            'loading': original_stats,
            'inference': original_perf
        }
        
        # 2. Quantized model
        print("2️⃣ Quantized model...")
        quantized_model = self.create_quantized_model(original_stats['model'], model_name)
        if quantized_model:
            quantized_stats = self.measure_model_loading_memory(lambda: quantized_model, f"{model_name}_quantized")
            quantized_perf = self.measure_inference_performance(
                quantized_stats['model'], f"{model_name}_quantized", input_tensor
            )
            
            variants['quantized'] = {
                'loading': quantized_stats,
                'inference': quantized_perf
            }
        
        # 3. Mobile optimized model
        print("3️⃣ Mobile optimized model...")
        mobile_model = self.create_mobile_optimized_model(original_stats['model'], model_name, input_tensor)
        if mobile_model:
            # Note: Mobile models need special handling for memory measurement
            mobile_perf = self.measure_inference_performance(
                mobile_model, f"{model_name}_mobile", input_tensor
            )
            
            variants['mobile'] = {
                'loading': {'theoretical_memory': {'total_mb': 'mobile_optimized'}},
                'inference': mobile_perf
            }
        
        # 4. Quantized + Mobile optimized
        print("4️⃣ Quantized + Mobile optimized model...")
        if quantized_model:
            quantized_mobile_model = self.create_mobile_optimized_model(quantized_model, f"{model_name}_quantized", input_tensor)
            if quantized_mobile_model:
                quantized_mobile_perf = self.measure_inference_performance(
                    quantized_mobile_model, f"{model_name}_quantized_mobile", input_tensor
                )
                
                variants['quantized_mobile'] = {
                    'loading': {'theoretical_memory': {'total_mb': 'quantized_mobile_optimized'}},
                    'inference': quantized_mobile_perf
                }
        
        return variants
    
    def print_comparison_table(self, model_name, variants):
        """Print detailed comparison table"""
        print(f"\n📊 Performance Comparison for {model_name}")
        print("=" * 100)
        
        # Header
        print(f"{'Variant':<20} {'FPS':<8} {'Inf(ms)':<10} {'RAM(MB)':<10} {'Theoretical(MB)':<15} {'Efficiency':<12}")
        print("-" * 100)
        
        for variant_name, data in variants.items():
            fps = data['inference']['fps']
            inf_ms = data['inference']['avg_inference_ms']
            
            if 'loading' in data and 'actual_memory' in data['loading']:
                ram_mb = data['loading']['actual_memory']['delta_mb']
                theoretical_mb = data['loading']['theoretical_memory']['total_mb']
                efficiency = data['loading']['memory_efficiency']
            else:
                ram_mb = "N/A"
                theoretical_mb = "N/A"
                efficiency = "N/A"
            
            print(f"{variant_name:<20} {fps:<8.1f} {inf_ms:<10.2f} {ram_mb:<10} {theoretical_mb:<15} {efficiency:<12}")
        
        # Performance improvements
        if 'original' in variants and 'quantized' in variants:
            orig_fps = variants['original']['inference']['fps']
            quant_fps = variants['quantized']['inference']['fps']
            fps_improvement = ((quant_fps - orig_fps) / orig_fps) * 100
            print(f"\n🚀 Quantization FPS improvement: {fps_improvement:+.1f}%")
        
        if 'original' in variants and 'mobile' in variants:
            orig_fps = variants['original']['inference']['fps']
            mobile_fps = variants['mobile']['inference']['fps']
            mobile_improvement = ((mobile_fps - orig_fps) / orig_fps) * 100
            print(f"📱 Mobile optimization FPS improvement: {mobile_improvement:+.1f}%")

def main():
    """Main benchmarking function"""
    print("🎯 Comprehensive Pi Zero Model Benchmark")
    print("📊 Measuring actual FPS, inference speed, and RAM usage")
    print("⚡ Comparing quantized vs non-quantized performance")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = ModelBenchmark(device='cpu')  # Pi Zero is CPU only
    
    # Create input tensor (typical mobile input size)
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # Models to benchmark
    models_to_test = [
        ("SuperPoint", lambda: torch.hub.load('magicleap/SuperGluePretrainedNetwork', 'superpoint_v1', pretrained=True)),
        ("MobileNetV2", lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)),
        ("MobileNetV3", lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)),
        ("EfficientNet-B0", lambda: timm.create_model('efficientnet_b0', pretrained=True)),
    ]
    
    all_results = {}
    
    for model_name, model_loader in models_to_test:
        try:
            print(f"\n🔍 Testing {model_name}")
            base_model = model_loader()
            
            # Adjust input size for SuperPoint
            if model_name == "SuperPoint":
                test_input = torch.randn(1, 1, 224, 224)  # Grayscale for SuperPoint
            else:
                test_input = input_tensor
            
            variants = benchmark.benchmark_model_variants(base_model, model_name, test_input)
            all_results[model_name] = variants
            
            # Print comparison table
            benchmark.print_comparison_table(model_name, variants)
            
        except Exception as e:
            print(f"❌ Failed to benchmark {model_name}: {e}")
            continue
    
    # Save results
    results_file = "pi_zero_comprehensive_benchmark_results.json"
    print(f"\n💾 Saving detailed results to {results_file}")
    
    # Convert results to JSON-serializable format
    json_results = {}
    for model_name, variants in all_results.items():
        json_results[model_name] = {}
        for variant_name, data in variants.items():
            json_results[model_name][variant_name] = {
                'fps': data['inference']['fps'],
                'avg_inference_ms': data['inference']['avg_inference_ms'],
                'std_inference_ms': data['inference']['std_inference_ms'],
                'min_inference_ms': data['inference']['min_inference_ms'],
                'max_inference_ms': data['inference']['max_inference_ms'],
            }
            
            if 'loading' in data and 'actual_memory' in data['loading']:
                json_results[model_name][variant_name].update({
                    'actual_memory_mb': data['loading']['actual_memory']['delta_mb'],
                    'theoretical_memory_mb': data['loading']['theoretical_memory']['total_mb'],
                    'memory_efficiency': data['loading']['memory_efficiency'],
                    'total_params': data['loading']['theoretical_memory']['total_params']
                })
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("✅ Benchmark complete!")
    print(f"📈 Results saved to: {results_file}")
    
    # Summary for Pi Zero
    print("\n🎯 Pi Zero Deployment Recommendations:")
    print("=" * 50)
    
    for model_name, variants in all_results.items():
        if 'original' in variants:
            orig_fps = variants['original']['inference']['fps']
            if 'loading' in variants['original'] and 'actual_memory' in variants['original']['loading']:
                orig_memory = variants['original']['loading']['actual_memory']['delta_mb']
                
                pi_zero_suitable = orig_memory < 100 and orig_fps > 5  # Basic thresholds
                status = "✅ Suitable" if pi_zero_suitable else "❌ Too heavy"
                
                print(f"{model_name}: {status} ({orig_fps:.1f} FPS, {orig_memory:.1f}MB RAM)")

if __name__ == "__main__":
    main() 