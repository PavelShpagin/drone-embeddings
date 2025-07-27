#!/bin/bash

echo "🚀 ADVANCED DINO OPTIMIZATION BENCHMARK"
echo "🔧 Multiple quantization approaches for theoretical sizes"
echo "📊 Comprehensive analysis of size effectiveness"
echo "================================================================="

echo "🎯 Installing additional dependencies..."
pip install -q onnx onnxruntime

echo ""
echo "🚀 Running advanced DINO optimization benchmark..."
python advanced_dino_optimization.py

echo ""
echo "📊 Checking results..."
if [ -f "advanced_dino_optimization_results.json" ]; then
    echo "✅ Results saved to: advanced_dino_optimization_results.json"
    echo "📄 Results file size: $(du -h advanced_dino_optimization_results.json | cut -f1)"
else
    echo "❌ No results file found!"
fi

echo ""
echo "🏁 Advanced DINO optimization complete!"
echo "🎯 This tests multiple quantization approaches to achieve theoretical sizes"
echo "📊 Shows which methods can get close to 4x/8x size reduction"
echo "🔧 Identifies the best quantization approach for Pi Zero deployment" 