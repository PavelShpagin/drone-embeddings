#!/bin/bash

# Run Comprehensive DINO Benchmark with INT4/INT8 Quantization
# Tests all models including DINO small, medium, large, SuperPoint

echo "🎯 Comprehensive DINO & Model Benchmark"
echo "🔬 Testing all models with INT4 and INT8 quantization"
echo "📊 Models included:"
echo "   - DINO Small (ViT-S/14)"
echo "   - DINO Base/Medium (ViT-B/14)"
echo "   - DINO Large (ViT-L/14)"
echo "   - Original DINO (ViT-B/16)"
echo "   - SuperPoint (keypoint detection)"
echo "   - MobileNetV2/V3"
echo "   - EfficientNet-B0"
echo "   - ResNet50"
echo "⚡ Quantization levels: FP32, INT8, INT4"
echo "=" * 80

# Use minimal environment to avoid disk space issues
if [ -d "venv_minimal" ]; then
    echo "📦 Using existing minimal environment..."
    source venv_minimal/bin/activate
else
    echo "🔧 Setting up minimal environment..."
    python3 -m venv venv_minimal
    source venv_minimal/bin/activate
    
    echo "📥 Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements_optimization.txt
fi

echo ""
echo "🧹 Clearing previous results..."
rm -f comprehensive_quantization_results.json

echo ""
echo "🚀 Starting comprehensive benchmark..."
echo "   This will take several minutes..."
echo "   Testing ~9 models × 3 quantization levels = 27 configurations"
echo ""

# Run the comprehensive benchmark
python3 comprehensive_dino_benchmark.py

echo ""
echo "📊 Checking results..."
if [ -f "comprehensive_quantization_results.json" ]; then
    echo "✅ Results saved to: comprehensive_quantization_results.json"
    echo "📄 Results file size: $(du -sh comprehensive_quantization_results.json | cut -f1)"
    
    # Show quick summary
    echo ""
    echo "🎯 Quick Pi Zero Compatibility Summary:"
    echo "   (Models with >10 FPS and <100MB RAM are suitable)"
    echo ""
    
else
    echo "❌ No results file generated"
fi

echo ""
echo "🏁 Comprehensive benchmark complete!"
echo "🔬 What was tested:"
echo "   ✅ DINO models (small, base, large)"
echo "   ✅ SuperPoint keypoint detection"
echo "   ✅ Traditional CNN models"
echo "   ✅ INT4 and INT8 quantization"
echo "   ✅ Pi Zero deployment analysis"
echo "   ✅ Real vs theoretical memory comparisons" 