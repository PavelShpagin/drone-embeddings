#!/bin/bash
# Run Extended DINO + ViT TensorFlow Lite Benchmark
# Includes ViT-S/8 and ViT-G/14 models

echo "🚀 DINO + ViT TENSORFLOW LITE EXTENDED BENCHMARK"
echo "🔧 Including ViT-S/8 and ViT-G/14 models"
echo "📊 Proper error handling and alternative model loading"
echo "=" | tr ' ' '='
echo

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "📦 Installing required packages..."
pip install -q torch torchvision timm psutil numpy

# Run the extended benchmark
echo "🎯 Running extended benchmark..."
python dino_tflite_extended.py

echo
echo "✅ Extended benchmark complete!"
echo "📊 Results saved to: dino_tflite_extended_results.json"
echo "🎯 ViT-S/8 and ViT-G/14 models included" 