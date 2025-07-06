#!/bin/bash

echo "🚀 DINO TENSORFLOW LITE BENCHMARK"
echo "🔢 Full Integer Quantization with In-Memory RAM measurement"
echo "📊 Focus on realistic INT8 quantization sizes"
echo "================================================================="

echo "🔧 Installing TensorFlow Lite dependencies..."
pip install -q tensorflow psutil

echo ""
echo "🚀 Running DINO TensorFlow Lite benchmark..."
python dino_tflite_benchmark.py

echo ""
echo "📊 Checking results..."
if [ -f "dino_tflite_benchmark_results.json" ]; then
    echo "✅ Results saved to: dino_tflite_benchmark_results.json"
    echo "📄 Results file size: $(du -h dino_tflite_benchmark_results.json | cut -f1)"
else
    echo "❌ No results file found!"
fi

echo ""
echo "🏁 DINO TensorFlow Lite benchmark complete!"
echo "🔢 TensorFlow Lite Full Integer Quantization should achieve close to theoretical sizes"
echo "📊 In-memory RAM usage is more accurate than disk storage"
echo "🎯 TFLite INT8 is the best option for Pi Zero deployment"
echo "🚀 TFLite often provides 2-4x better performance than PyTorch" 