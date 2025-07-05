#!/bin/bash

# Minimal Installation Script for PyTorch Model Optimization
# This script installs only essential packages to avoid disk space issues

echo "🚀 Installing minimal PyTorch optimization requirements..."
echo "📦 This will install only essential packages (~500MB instead of 2GB+)"

# Create/activate virtual environment if it doesn't exist
if [ ! -d "venv_minimal" ]; then
    echo "Creating minimal virtual environment..."
    python3 -m venv venv_minimal
fi

source venv_minimal/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install minimal requirements
echo "📥 Installing essential packages only..."
pip install -r requirements_optimization.txt

echo "✅ Installation complete!"
echo ""
echo "🎯 What's installed:"
echo "   - PyTorch (CPU-only for Pi Zero compatibility)"
echo "   - TorchVision (for image processing)"
echo "   - TIMM (for pre-trained models)"
echo "   - ONNX (for model conversion)"
echo "   - ONNXRuntime (for optimized inference)"
echo "   - Analysis tools (thop, psutil, pandas)"
echo "   - Basic utilities (PIL, numpy, tqdm)"
echo ""
echo "💾 Disk space saved: ~1.5GB (no TensorFlow, no visualization libs)"
echo ""
echo "🔧 To use: source venv_minimal/bin/activate"
echo "📊 To install visualization (optional): pip install matplotlib seaborn" 