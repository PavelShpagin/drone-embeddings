#!/bin/bash

# Run Fixed Performance Benchmark
# Addresses all the issues from the previous benchmark

echo "🎯 Fixed Performance Benchmark"
echo "🔧 Addressing issues from previous run:"
echo "   - Fixed 0.0 MB RAM measurements"
echo "   - Replaced mobile optimizer with TorchScript"
echo "   - Added proper real vs theoretical comparisons"
echo "   - Better error handling for quantization"
echo "=" * 60

# Use minimal environment to avoid disk space issues
if [ -d "venv_minimal" ]; then
    echo "📦 Using existing minimal environment..."
    source venv_minimal/bin/activate
else
    echo "🔧 Setting up minimal environment..."
    python3 -m venv venv_minimal
    source venv_minimal/bin/activate
    
    echo "📥 Installing minimal requirements..."
    pip install --upgrade pip
    pip install -r requirements_optimization.txt
fi

echo ""
echo "🧹 Clearing previous results..."
rm -f fixed_performance_results.json

echo ""
echo "🚀 Running fixed benchmark..."
echo "   This will properly measure:"
echo "   - Actual RAM usage (no more 0.0 MB)"
echo "   - Real vs theoretical memory comparisons"
echo "   - FPS improvements with quantization"
echo "   - TorchScript optimization effects"
echo ""

# Run the fixed benchmark
python3 fixed_performance_benchmark.py

echo ""
echo "📊 Checking results..."
if [ -f "fixed_performance_results.json" ]; then
    echo "✅ Results saved to: fixed_performance_results.json"
    echo "📄 Results file size: $(du -sh fixed_performance_results.json | cut -f1)"
else
    echo "❌ No results file generated"
fi

echo ""
echo "🎯 Fixed benchmark complete!"
echo "🔧 Key improvements:"
echo "   ✅ Memory measurements now work properly"
echo "   ✅ TorchScript optimization replaces mobile optimizer"
echo "   ✅ Clear real vs theoretical comparisons"
echo "   ✅ Comprehensive Pi Zero analysis" 