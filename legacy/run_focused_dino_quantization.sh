#!/bin/bash

echo "🎯 FOCUSED DINO QUANTIZATION BENCHMARK"
echo "🔧 Testing the most promising quantization approaches"
echo "📊 Targeting theoretical size effectiveness"
echo "================================================================="

echo "🚀 Running focused DINO quantization benchmark..."
python focused_dino_quantization.py

echo ""
echo "📊 Checking results..."
if [ -f "focused_dino_quantization_results.json" ]; then
    echo "✅ Results saved to: focused_dino_quantization_results.json"
    echo "📄 Results file size: $(du -h focused_dino_quantization_results.json | cut -f1)"
else
    echo "❌ No results file found!"
fi

echo ""
echo "🏁 Focused DINO quantization complete!"
echo "🎯 This identifies the most promising quantization approaches"
echo "📊 Shows realistic expectations for DINO model deployment"
echo "🔧 Targets methods that can achieve close to theoretical sizes" 