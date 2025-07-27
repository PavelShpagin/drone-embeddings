#!/bin/bash

# Make All Benchmark Scripts Executable
echo "🔧 Making all benchmark scripts executable..."

# Make all scripts executable
chmod +x complete_pi_zero_analysis.sh
chmod +x push_superpoint_weights.sh
chmod +x run_dino_focused_benchmark.sh
chmod +x dino_focused_benchmark.py
chmod +x run_lightweight_benchmark.sh
chmod +x lightweight_pi_zero_benchmark.py
chmod +x run_fixed_benchmark.sh
chmod +x fixed_performance_benchmark.py
chmod +x comprehensive_dino_benchmark.py
chmod +x run_comprehensive_dino_benchmark.sh
chmod +x setup_and_run_all_benchmarks.sh

echo "✅ All scripts are now executable!"
echo ""
echo "🎯 Available benchmarks:"
echo "1. ./complete_pi_zero_analysis.sh (complete analysis & answers)"
echo "2. ./run_dino_focused_benchmark.sh (DINO INT8/INT4 only)"
echo "3. ./run_lightweight_benchmark.sh (lightweight test)"
echo "4. ./push_superpoint_weights.sh (push weights to GitHub)"
echo ""
echo "📊 Recommended: Start with ./complete_pi_zero_analysis.sh" 