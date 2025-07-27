#!/bin/bash

echo "============================================="
echo "🚀 Comprehensive Model Optimization Benchmark"
echo "Including smallest DINO version & detailed logging"
echo "============================================="

# Clear previous results
rm -rf optimization_results/
mkdir -p optimization_results

# Run the comprehensive benchmark
python3 comprehensive_benchmark.py

echo ""
echo "============================================="
echo "✅ COMPREHENSIVE BENCHMARK COMPLETE!"
echo "============================================="
echo ""
echo "📊 Check optimization_results/ directory for:"
echo "  • Detailed JSON results with all metrics"
echo "  • CSV summary for easy analysis"
echo "  • Console output shows optimization benefits"
echo ""
echo "🎯 Key Features:"
echo "  ✅ Uses smallest DINOv2 version (vits14)"
echo "  ✅ Detailed optimization benefit logging"
echo "  ✅ SuperPoint keypoint detection model"
echo "  ✅ Mobile optimization verification"
echo "  ✅ Quantization with accuracy preservation"
echo "  ✅ Pi Zero deployment recommendations" 