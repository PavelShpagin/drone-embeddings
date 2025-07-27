#!/bin/bash

# Run Lightweight Pi Zero Benchmark
# Tests only viable models to avoid crashes like DINO-L

echo "🎯 Lightweight Pi Zero Benchmark"
echo "📊 Testing only viable models to avoid memory crashes"
echo "🔬 Includes DINO-S but skips DINO-L (too large)"
echo "=" * 60

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
rm -f lightweight_pi_zero_results.json

echo ""
echo "🚀 Running lightweight benchmark..."
echo "   This will test:"
echo "   - MobileNetV2 (baseline)"
echo "   - MobileNetV3 (likely winner)"
echo "   - EfficientNet-B0 (accuracy)"
echo "   - DINO-S (vision transformer)"
echo "   - Skip DINO-L (too large for Pi)"
echo ""

# Run the lightweight benchmark
python3 lightweight_pi_zero_benchmark.py

echo ""
echo "📊 Benchmark complete!"
echo "🏆 Check results for the best Pi Zero model" 