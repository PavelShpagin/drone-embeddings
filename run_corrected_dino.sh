#!/bin/bash

echo "🎯 CORRECTED DINO BENCHMARK"
echo "🔧 Fixed quantization and size calculation issues"
echo "📊 Proper theoretical vs actual size comparison"
echo "================================================================="

echo "🚀 Running corrected DINO benchmark..."
python corrected_dino_benchmark.py

echo ""
echo "📊 Checking results..."
if [ -f "corrected_dino_results.json" ]; then
    echo "✅ Results saved to: corrected_dino_results.json"
    echo "📄 Results file size: $(du -h corrected_dino_results.json | cut -f1)"
else
    echo "❌ No results file found!"
fi

echo ""
echo "🏁 Corrected DINO benchmark complete!"
echo "🔧 This should fix the suspicious 1.5MB quantized sizes"
echo "📊 Shows real quantization effectiveness vs theoretical"
echo "🎯 Identifies if PyTorch quantization actually works for ViTs" 