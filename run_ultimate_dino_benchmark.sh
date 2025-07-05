#!/bin/bash

echo "🎯 ULTIMATE DINO BENCHMARK"
echo "🔥 Testing ALL configurations with correct size calculations"
echo "📊 Including largest models: DINOv2-L and DINOv2-G"
echo "================================================================================"

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate || source venv_minimal/bin/activate
else
    echo "📦 Using existing virtual environment: $VIRTUAL_ENV"
fi

echo "🧹 Clearing previous results..."
rm -f ultimate_dino_benchmark_results.json

echo "🚀 Starting ultimate DINO benchmark..."
echo "   This will test ALL DINO configurations:"
echo "   - DINO-S/16, DINO-S/8, DINO-B/16, DINO-B/8"
echo "   - DINOv2-S/14, DINOv2-B/14, DINOv2-L/14, DINOv2-G/14"
echo "   - Each with FP32, INT8, and INT4 quantization"
echo "   - Total: 24 configurations"
echo "   - Proper size calculations based on ACTUAL quantized models"
echo ""

python3 ultimate_dino_benchmark.py

echo ""
echo "📊 Checking results..."
if [ -f "ultimate_dino_benchmark_results.json" ]; then
    echo "✅ Results saved to: ultimate_dino_benchmark_results.json"
    echo "📄 Results file size: $(du -h ultimate_dino_benchmark_results.json | cut -f1)"
else
    echo "❌ Results file not found!"
fi

echo ""
echo "🏆 Ultimate DINO benchmark complete!"
echo "🎯 This tested ALL configurations including the largest models"
echo "📊 Size calculations based on ACTUAL quantized models"
echo "🔥 Should answer why small models vs large quantized models" 