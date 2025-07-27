#!/bin/bash

# Simple Performance Measurement Runner
# Uses minimal requirements to avoid disk space issues

echo "🎯 Actual Performance Measurement"
echo "📊 Measuring real FPS and RAM usage (not hardcoded)"
echo "⚡ Comparing quantized vs non-quantized models"
echo "💾 Using minimal environment to save disk space"
echo ""

# Use minimal environment or create it
if [ -d "venv_minimal" ]; then
    echo "📦 Using existing minimal environment..."
    source venv_minimal/bin/activate
else
    echo "🔧 Creating minimal environment..."
    ./install_minimal_optimization.sh
    source venv_minimal/bin/activate
fi

# Clean up previous results
rm -f actual_performance_measurements.json

# Run the focused performance measurement
echo "🚀 Starting performance measurement..."
echo "   This will measure actual RAM usage and FPS"
echo "   All numbers are real measurements from your system"
echo ""

python measure_actual_performance.py

# Show results
if [ -f "actual_performance_measurements.json" ]; then
    echo ""
    echo "✅ Performance measurement completed!"
    echo "📁 Results saved to: actual_performance_measurements.json"
    echo ""
    echo "📊 Quick Summary:"
    echo "=================="
    
    # Show quick summary
    python -c "
import json
try:
    with open('actual_performance_measurements.json', 'r') as f:
        results = json.load(f)
    
    print('Model Performance (Actual Measurements):')
    print('-' * 45)
    
    for model_name, variants in results.items():
        if 'original' in variants:
            orig = variants['original']
            fps = orig['performance']['fps']
            ram = orig['ram']['actual_mb']
            theoretical = orig['ram']['theoretical_mb']
            
            print(f'{model_name:15}: {fps:6.1f} FPS')
            print(f'    RAM Actual: {ram:6.1f} MB')
            print(f'    RAM Theory: {theoretical:6.1f} MB')
            
            if 'quantized' in variants:
                quant_fps = variants['quantized']['performance']['fps']
                improvement = ((quant_fps - fps) / fps) * 100
                print(f'    Quantized:  {quant_fps:6.1f} FPS ({improvement:+.1f}%)')
            
            print()
            
except Exception as e:
    print(f'Error reading results: {e}')
"
    
    echo ""
    echo "🎯 Pi Zero Readiness:"
    echo "- Models with >5 FPS and <100MB RAM are suitable"
    echo "- Quantization typically improves performance"
    echo "- Check JSON file for detailed metrics"
    
else
    echo ""
    echo "❌ Performance measurement failed"
    echo "Check console output above for errors"
fi

echo ""
echo "🏁 Performance measurement complete!" 