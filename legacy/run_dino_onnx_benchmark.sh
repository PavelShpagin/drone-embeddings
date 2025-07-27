#!/bin/bash

echo "🚀 DINO ONNX BENCHMARK"
echo "🔄 Convert DINO models to ONNX for better performance"
echo "📊 Compare PyTorch vs ONNX vs ONNX Quantized"
echo "================================================================="

echo "🔧 Installing ONNX dependencies..."
pip install -q onnx onnxruntime

echo ""
echo "🚀 Running DINO ONNX benchmark..."
python dino_onnx_benchmark.py

echo ""
echo "📊 Checking results..."
if [ -f "dino_onnx_benchmark_results.json" ]; then
    echo "✅ Results saved to: dino_onnx_benchmark_results.json"
    echo "📄 Results file size: $(du -h dino_onnx_benchmark_results.json | cut -f1)"
else
    echo "❌ No results file found!"
fi

echo ""
echo "🏁 DINO ONNX benchmark complete!"
echo "🔄 ONNX conversion provides better performance and deployment options"
echo "📊 Use ONNX quantized models for the best size/performance balance"
echo "🎯 ONNX Runtime is often 2-3x faster than PyTorch for inference" 