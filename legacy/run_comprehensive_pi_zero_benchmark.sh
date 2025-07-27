#!/bin/bash

# Comprehensive Pi Zero Benchmark Runner
# Measures actual FPS, inference speed, and RAM usage

echo "🎯 Comprehensive Pi Zero Model Benchmark"
echo "📊 Measuring actual performance metrics..."
echo "⚡ Comparing quantized vs non-quantized models"
echo "💾 Using minimal requirements to avoid disk space issues"
echo "=" * 60

# Check if minimal environment exists
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

# Verify requirements
echo "🔍 Verifying installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import timm; print('✅ TIMM')"
python -c "import psutil; print('✅ psutil')"

# Clear any previous results
echo "🧹 Clearing previous results..."
rm -f pi_zero_comprehensive_benchmark_results.json

# Run comprehensive benchmark
echo "🚀 Starting comprehensive benchmark..."
echo "⏱️  This will measure:"
echo "   - Actual RAM usage during model loading"
echo "   - Real FPS and inference times"
echo "   - Quantized vs non-quantized performance"
echo "   - Mobile optimization effects"
echo ""

python comprehensive_pi_zero_benchmark.py

# Check if results were generated
if [ -f "pi_zero_comprehensive_benchmark_results.json" ]; then
    echo ""
    echo "✅ Benchmark completed successfully!"
    echo "📊 Results saved to: pi_zero_comprehensive_benchmark_results.json"
    echo ""
    echo "📈 Summary of results:"
    echo "=" * 40
    
    # Quick summary using Python
    python -c "
import json
try:
    with open('pi_zero_comprehensive_benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    print('Model Performance Summary:')
    print('-' * 40)
    
    for model_name, variants in results.items():
        if 'original' in variants:
            orig = variants['original']
            fps = orig['fps']
            ram = orig.get('actual_memory_mb', 'N/A')
            
            print(f'{model_name:15}: {fps:6.1f} FPS, {ram} MB RAM')
            
            # Show improvements if available
            if 'quantized' in variants:
                quant_fps = variants['quantized']['fps']
                improvement = ((quant_fps - fps) / fps) * 100
                print(f'  -> Quantized:  {quant_fps:6.1f} FPS ({improvement:+.1f}%)')
                
            if 'mobile' in variants:
                mobile_fps = variants['mobile']['fps']
                mobile_improvement = ((mobile_fps - fps) / fps) * 100
                print(f'  -> Mobile:     {mobile_fps:6.1f} FPS ({mobile_improvement:+.1f}%)')
            
            print()
            
except Exception as e:
    print(f'Error reading results: {e}')
"
    
    echo ""
    echo "🎯 Pi Zero Deployment Insights:"
    echo "   - Models with >5 FPS and <100MB RAM are Pi Zero suitable"
    echo "   - Quantization typically improves FPS by 20-50%"
    echo "   - Mobile optimization may improve or worsen performance on x86"
    echo "   - Best performance will be on actual ARM hardware"
    
else
    echo ""
    echo "❌ Benchmark failed or results not generated"
    echo "📋 Check console output above for error details"
fi

echo ""
echo "🏁 Benchmark session complete!" 