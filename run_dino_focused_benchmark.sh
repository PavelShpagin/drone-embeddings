#!/bin/bash

# Run Focused DINO Benchmark - INT8/INT4 Only
# Tests specific DINO variants: S/16, S/8, B, DINOv2-small
# Compares only quantized metrics to save time

echo "🎯 Focused DINO Benchmark - INT8/INT4 Only"
echo "🔬 Testing specific DINO variants with quantization focus"
echo "📊 Models to test:"
echo "   - DINOv2-Small (patch/14)"
echo "   - DINO-S/16 (small, patch/16)"
echo "   - DINO-S/8 (small, patch/8)"
echo "   - DINO-B/16 (base, patch/16)"
echo "⚡ Quantization: INT8 and INT4 only (skipping FP32)"
echo "=" * 70

# Use minimal environment
if [ -d "venv_minimal" ]; then
    echo "📦 Using existing minimal environment..."
    source venv_minimal/bin/activate
else
    echo "🔧 Setting up minimal environment..."
    python3 -m venv venv_minimal
    source venv_minimal/bin/activate
    pip install -r requirements_optimization.txt
fi

echo ""
echo "🧹 Clearing previous results..."
rm -f dino_focused_benchmark_results.json

echo ""
echo "🚀 Starting focused DINO benchmark..."
echo "   This focuses only on quantized metrics"
echo "   Should avoid memory crashes by testing smaller models"
echo "   And skipping FP32 reduces memory usage"
echo ""

# Run the focused benchmark
python3 dino_focused_benchmark.py

echo ""
echo "📊 Checking results..."
if [ -f "dino_focused_benchmark_results.json" ]; then
    echo "✅ Results saved to: dino_focused_benchmark_results.json"
    echo "📄 Results file size: $(du -sh dino_focused_benchmark_results.json | cut -f1)"
    
    echo ""
    echo "🏆 Quick Summary:"
    echo "   Check output above for best INT8 and INT4 DINO models"
    echo "   Models with >10 FPS and <50MB are Pi Zero compatible"
else
    echo "❌ No results file generated"
fi

echo ""
echo "🏁 Focused DINO benchmark complete!"
echo "🎯 Tested only INT8/INT4 quantization for Pi Zero focus" 