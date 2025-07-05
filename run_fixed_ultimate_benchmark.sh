#!/bin/bash

echo "🎯 FIXED ULTIMATE DINO BENCHMARK"
echo "🔧 Using working model names and correct size calculations"
echo "📊 Testing ALL quantization types: FP32, INT8, INT4"
echo "================================================================================"

echo "🚀 Running fixed ultimate DINO benchmark..."
python3 fixed_ultimate_dino_benchmark.py

echo ""
echo "📊 Checking results..."
if [ -f "fixed_ultimate_dino_results.json" ]; then
    echo "✅ Results saved to: fixed_ultimate_dino_results.json"
    echo "📄 Results file size: $(du -h fixed_ultimate_dino_results.json | cut -f1)"
else
    echo "❌ Results file not found!"
fi

echo ""
echo "🏁 Fixed ultimate DINO benchmark complete!"
echo "🔧 This should have working model names and correct size calculations"
echo "📊 Should show realistic MB sizes (not 0.2-0.7MB)" 