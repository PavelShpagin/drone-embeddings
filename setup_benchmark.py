#!/usr/bin/env python3
"""
Setup script for DINO Benchmark on Raspberry Pi Zero Environment
================================================================

This script installs necessary dependencies and configures the environment
for running DINO benchmarks with Raspberry Pi Zero constraints simulation.
"""

import os
import sys
import subprocess
import platform
import warnings
warnings.filterwarnings('ignore')

def check_python_version():
    """Check if Python version is compatible"""
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_package(package_name, pip_name=None):
    """Install a Python package with error handling"""
    install_name = pip_name or package_name
    
    try:
        __import__(package_name)
        print(f"✅ {package_name} already installed")
        return True
    except ImportError:
        print(f"📦 Installing {package_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", install_name,
                "--quiet", "--disable-pip-version-check"
            ])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False

def install_torch_cpu_only():
    """Install PyTorch CPU-only version for edge simulation"""
    try:
        import torch
        print(f"✅ PyTorch already installed: {torch.__version__}")
        
        # Check if CUDA is available (should be disabled for Pi Zero simulation)
        if torch.cuda.is_available():
            print("⚠️  CUDA detected. For accurate Pi Zero simulation, consider CPU-only PyTorch")
        else:
            print("✅ CPU-only PyTorch detected (suitable for Pi Zero simulation)")
        
        return True
    except ImportError:
        print("📦 Installing PyTorch CPU-only version...")
        try:
            # Install CPU-only PyTorch for Pi Zero simulation
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "--index-url", 
                "https://download.pytorch.org/whl/cpu",
                "--quiet", "--disable-pip-version-check"
            ])
            print("✅ PyTorch CPU-only installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install PyTorch: {e}")
            print("📝 Benchmark will run in dummy mode without PyTorch")
            return False

def check_memory_constraints():
    """Check if system has enough memory and warn about constraints"""
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"💾 System Memory: {total_memory_gb:.1f} GB")
        
        if total_memory_gb < 1:
            print("⚠️  Low memory system detected")
            print("   This is suitable for Pi Zero simulation")
        elif total_memory_gb > 4:
            print("⚠️  High memory system detected")
            print("   Benchmark will simulate 512MB Pi Zero constraints")
        
        return True
    except ImportError:
        print("❌ Cannot check memory - psutil not available")
        return False

def detect_platform():
    """Detect platform and provide specific recommendations"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"🖥️  Platform: {platform.platform()}")
    print(f"🏗️  Architecture: {machine}")
    
    # Provide platform-specific recommendations
    if "arm" in machine:
        print("✅ ARM architecture detected - similar to Raspberry Pi")
        if "armv6" in machine:
            print("✅ ARMv6 detected - matches Raspberry Pi Zero exactly")
        elif "armv7" in machine:
            print("ℹ️  ARMv7 detected - more powerful than Pi Zero")
        elif "aarch64" in machine or "arm64" in machine:
            print("ℹ️  ARM64 detected - much more powerful than Pi Zero")
    else:
        print("ℹ️  x86/x64 architecture - will simulate Pi Zero constraints")
    
    return system, machine

def create_benchmark_config():
    """Create configuration file for benchmark"""
    config = {
        "raspberry_pi_zero": {
            "memory_mb": 512,
            "cpu_cores": 1,
            "cpu_freq_mhz": 1000,
            "architecture": "ARMv6",
            "gpu": None
        },
        "benchmark_settings": {
            "batch_size": 1,
            "max_test_samples": 50,
            "quantization_methods": ["none", "8bit", "4bit", "binary"],
            "model_configs": {
                "micro": {"dim": 128, "depth": 3, "heads": 2, "mlp_dim": 512},
                "tiny": {"dim": 192, "depth": 4, "heads": 3, "mlp_dim": 768},
                "small": {"dim": 384, "depth": 6, "heads": 6, "mlp_dim": 1536}
            }
        }
    }
    
    try:
        import json
        with open('benchmark_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Benchmark configuration saved to benchmark_config.json")
        return True
    except Exception as e:
        print(f"❌ Failed to create config file: {e}")
        return False

def run_quick_test():
    """Run a quick test to verify installation"""
    print("\n🧪 Running quick installation test...")
    
    try:
        # Test basic imports
        import numpy as np
        print("✅ NumPy import successful")
        
        import psutil
        print("✅ psutil import successful")
        
        try:
            import torch
            print(f"✅ PyTorch import successful: {torch.__version__}")
            
            # Test basic tensor operations
            x = torch.randn(2, 3)
            y = torch.sum(x)
            print("✅ PyTorch tensor operations working")
            
        except ImportError:
            print("⚠️  PyTorch not available - will run in dummy mode")
        
        try:
            import matplotlib.pyplot as plt
            print("✅ Matplotlib import successful")
        except ImportError:
            print("⚠️  Matplotlib not available - plots will be skipped")
        
        print("✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def print_usage_instructions():
    """Print instructions for running the benchmark"""
    print("\n" + "="*60)
    print("🚀 SETUP COMPLETE - USAGE INSTRUCTIONS")
    print("="*60)
    print()
    print("To run the DINO benchmark:")
    print("  python dino_benchmark.py")
    print()
    print("The benchmark will:")
    print("  • Simulate Raspberry Pi Zero constraints (512MB RAM)")
    print("  • Test multiple DINO model configurations")
    print("  • Apply various quantization techniques")
    print("  • Generate performance reports and visualizations")
    print()
    print("Output files:")
    print("  • dino_benchmark_results.json - Detailed results")
    print("  • dino_benchmark_plots.png - Performance visualizations")
    print()
    print("For help with the benchmark script:")
    print("  python dino_benchmark.py --help")
    print()

def main():
    """Main setup function"""
    print("🔧 DINO Benchmark Setup for Raspberry Pi Zero Simulation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect platform
    system, machine = detect_platform()
    
    print("\n📋 Installing required packages...")
    
    # Install core packages
    success = True
    
    # Essential packages
    essential_packages = [
        ("numpy", "numpy>=1.19.0"),
        ("psutil", "psutil>=5.8.0"),
        ("matplotlib", "matplotlib>=3.3.0"),
        ("tqdm", "tqdm>=4.60.0")
    ]
    
    for package, pip_name in essential_packages:
        if not install_package(package, pip_name):
            success = False
    
    # Optional packages
    print("\n📋 Installing optional packages...")
    
    # PyTorch (CPU-only for Pi Zero simulation)
    install_torch_cpu_only()
    
    # PIL for image processing
    install_package("PIL", "Pillow>=8.0.0")
    
    # Check memory constraints
    print("\n💾 Checking system constraints...")
    check_memory_constraints()
    
    # Create configuration
    print("\n⚙️  Creating benchmark configuration...")
    create_benchmark_config()
    
    # Run quick test
    if run_quick_test():
        print_usage_instructions()
    else:
        print("\n❌ Setup completed with errors. Some features may not work.")
    
    print("\n✅ Setup completed!")

if __name__ == "__main__":
    main()