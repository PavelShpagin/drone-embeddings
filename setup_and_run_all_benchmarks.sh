#!/bin/bash

# Setup and Run All Benchmarks
# Makes scripts executable and provides options

echo "🎯 Benchmark Setup and Runner"
echo "🔧 Setting up all benchmark scripts..."

# Make all benchmark scripts executable
chmod +x run_fixed_benchmark.sh
chmod +x run_comprehensive_dino_benchmark.sh
chmod +x comprehensive_dino_benchmark.py
chmod +x fixed_performance_benchmark.py

echo "✅ All scripts are now executable!"
echo ""

# Show available options
echo "📊 Available Benchmarks:"
echo "=" * 50
echo "1. Fixed Performance Benchmark"
echo "   - Fixes 0.0 MB RAM issue"
echo "   - Tests MobileNet, EfficientNet, ResNet"
echo "   - INT8 quantization only"
echo "   - Run with: ./run_fixed_benchmark.sh"
echo ""

echo "2. Comprehensive DINO Benchmark (RECOMMENDED)"
echo "   - DINO Small, Base, Large models"
echo "   - SuperPoint keypoint detection"
echo "   - All traditional models"
echo "   - INT4 AND INT8 quantization"
echo "   - Run with: ./run_comprehensive_dino_benchmark.sh"
echo ""

echo "🚀 Quick Start:"
echo "   ./run_comprehensive_dino_benchmark.sh"
echo ""

# Ask user what they want to run
echo "What would you like to run?"
echo "1) Fixed benchmark (faster)"
echo "2) Comprehensive DINO benchmark (complete)"
echo "3) Both (recommended)"
echo "4) Just setup (don't run anything)"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Running fixed benchmark..."
        ./run_fixed_benchmark.sh
        ;;
    2)
        echo "🚀 Running comprehensive DINO benchmark..."
        ./run_comprehensive_dino_benchmark.sh
        ;;
    3)
        echo "🚀 Running both benchmarks..."
        echo "📊 Starting with fixed benchmark..."
        ./run_fixed_benchmark.sh
        echo ""
        echo "📊 Now running comprehensive DINO benchmark..."
        ./run_comprehensive_dino_benchmark.sh
        ;;
    4)
        echo "✅ Setup complete! You can now run:"
        echo "   ./run_fixed_benchmark.sh"
        echo "   ./run_comprehensive_dino_benchmark.sh"
        ;;
    *)
        echo "❌ Invalid choice. Setup complete, but no benchmark run."
        echo "✅ You can now run:"
        echo "   ./run_fixed_benchmark.sh"
        echo "   ./run_comprehensive_dino_benchmark.sh"
        ;;
esac

echo ""
echo "🏁 Done!" 