#!/bin/bash

echo "============================================="
echo "🚀 AnyLoc-Inspired Comprehensive Benchmark"
echo "============================================="
echo ""
echo "📋 This benchmark will test:"
echo "  • DINOv2 ViT-S/14 (21M params, ~84MB)"
echo "  • DINOv2 ViT-B/14 (86M params, ~344MB)"
echo "  • DINOv2 ViT-L/14 (300M params, ~1.2GB)"
echo "  • SuperPoint (keypoint detection)"
echo "  • MobileNetV2 & EfficientNet-B0"
echo ""
echo "🔧 Optimization techniques:"
echo "  • INT8 Quantization"
echo "  • PyTorch Mobile Optimization"
echo ""
echo "📊 Clear console logging of optimization benefits"
echo ""

# Clear previous results
rm -rf optimization_results/
mkdir -p optimization_results

echo "🧹 Cleared previous results"
echo ""

# Run the comprehensive benchmark
echo "🏁 Starting benchmark..."
python3 anyloc_comprehensive_benchmark.py

echo ""
echo "============================================="
echo "✅ ANYLOC BENCHMARK COMPLETE!"
echo "============================================="
echo ""
echo "📂 Results saved to optimization_results/ directory:"
echo "  • anyloc_benchmark_results.json (detailed)"
echo "  • anyloc_benchmark_summary.csv (summary)"
echo ""
echo "🎯 Key findings should be visible in console output above"
echo "💡 Look for optimization benefits and Pi Zero recommendations"
echo "" 