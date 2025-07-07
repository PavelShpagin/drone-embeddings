#!/bin/bash
# Install Dependencies for Pi Zero Performance Benchmarks
# Minimal installation for dino_tflite_fixed.py and related benchmarks

echo "🚀 Installing Pi Zero Benchmark Dependencies"
echo "📦 Installing only essential packages to save space"
echo "=" | tr ' ' '='
echo

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  WARNING: Not in a virtual environment!"
    echo "   Consider running: python3 -m venv venv && source venv/bin/activate"
    echo
fi

# Update pip
echo "🔄 Updating pip..."
python3 -m pip install --upgrade pip

# Install core PyTorch (CPU-only for Pi Zero)
echo "🔥 Installing PyTorch (CPU-only)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install timm for model loading
echo "🤖 Installing timm..."
pip install timm>=0.6.0

# Install monitoring and analysis
echo "📊 Installing analysis tools..."
pip install psutil>=5.9.0 numpy>=1.21.0 tqdm>=4.64.0

# Optional: Install pandas for data analysis
echo "📈 Installing pandas (optional)..."
pip install pandas>=1.5.0

# Optional: Install Pillow for image processing
echo "🖼️ Installing Pillow (optional)..."
pip install Pillow>=9.0.0

# Optional: Install thop for FLOPs calculation
echo "🧮 Installing thop (optional)..."
pip install thop>=0.1.0

echo
echo "✅ INSTALLATION COMPLETE!"
echo "🎯 Ready to run: python dino_tflite_fixed.py"
echo "💾 Total installed packages: ~6-8 essential packages"
echo "📊 Disk usage: ~500MB (much smaller than full requirements.txt)"
echo
echo "🚀 To run benchmark:"
echo "   python dino_tflite_fixed.py"
echo
echo "📦 Optional additions (install if needed):"
echo "   pip install onnx onnxruntime  # For ONNX benchmarks"
echo "   pip install tensorflow        # For TensorFlow Lite benchmarks"
echo "   pip install matplotlib seaborn # For visualization" 