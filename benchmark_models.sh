#!/bin/bash

# Model Optimization and Benchmarking Script for Raspberry Pi Zero
# This script runs comprehensive model optimization including DINO and DINOv2

set -e  # Exit on any error

echo "============================================="
echo "Model Optimization Benchmark for Pi Zero"
echo "Including SuperPoint, DINO, and DINOv2"
echo "============================================="

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to cleanup on exit
cleanup() {
    log "Cleaning up temporary files..."
    # Remove any temporary model files if they exist
    rm -f /tmp/model_*.pt
    log "Cleanup complete."
}

# Set trap for cleanup
trap cleanup EXIT

# Clear previous results (overwrite mode)
log "Clearing previous results..."
rm -rf optimization_results/
rm -rf logs/
mkdir -p optimization_results
mkdir -p logs

log "Starting model optimization benchmark..."

# Step 1: Install/upgrade dependencies
log "Installing required dependencies..."
pip install --upgrade --quiet torch torchvision timm tqdm numpy matplotlib seaborn pandas psutil thop || {
    log "Warning: Some dependencies failed to install"
}

# Step 2: Check system info
log "System Information:"
echo "  Python version: $(python3 --version)"
echo "  PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  Device: $(python3 -c 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")')"
echo "  CPU cores: $(nproc)"
echo "  Available RAM: $(free -h | grep Mem | awk '{print $7}')"

# Step 3: Run the optimization with detailed logging
log "Running model optimization and benchmarking..."
echo "Models to test: SuperPoint, MobileNetV2, MobileNetV3, EfficientNet-B0, ResNet50, DINO, DINOv2"
echo "Optimizations: Original, INT8 Quantization, PyTorch Mobile"
echo ""

# Run optimization with comprehensive error handling
python3 optimize_models.py 2>&1 | tee logs/optimization_detailed.log

# Check if optimization was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "Model optimization completed successfully!"
else
    log "Model optimization encountered errors, but continuing with analysis..."
fi

# Step 4: Generate comprehensive analysis report
log "Generating comprehensive analysis report..."
python3 -c "
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Find the latest results file
results_files = glob.glob('optimization_results/optimization_results_*.json')
if not results_files:
    print('No results files found!')
    exit(1)

latest_file = max(results_files)
print(f'Processing results from: {latest_file}')

# Load results
with open(latest_file, 'r') as f:
    results = json.load(f)

# Convert to DataFrame for analysis
data = []
for model_name, metrics in results.items():
    if 'model_name' in metrics:
        data.append({
            'Model': model_name,
            'Base Model': metrics.get('base_model', 'unknown'),
            'Variant': metrics.get('variant', 'unknown'),
            'Size (MB)': metrics.get('size_mb', 0),
            'Avg Inference (ms)': metrics.get('avg_inference_time_ms', 0),
            'Min Inference (ms)': metrics.get('min_inference_time_ms', 0),
            'Max Inference (ms)': metrics.get('max_inference_time_ms', 0),
            'Std Inference (ms)': metrics.get('std_inference_time_ms', 0),
            'Throughput (FPS)': metrics.get('throughput_fps', 0),
            'Memory (MB)': metrics.get('avg_memory_usage_mb', 0),
            'Input Size': str(metrics.get('input_size', [])),
            'Description': metrics.get('description', '')
        })

if not data:
    print('No valid data found in results!')
    exit(1)

df = pd.DataFrame(data)

# Sort by throughput (descending)
df_sorted = df.sort_values('Throughput (FPS)', ascending=False)

print('\\n' + '='*100)
print('MODEL OPTIMIZATION RESULTS SUMMARY')
print('='*100)
print(df_sorted[['Model', 'Size (MB)', 'Avg Inference (ms)', 'Throughput (FPS)', 'Input Size']].to_string(index=False, float_format='%.2f'))

# Detailed performance analysis
print('\\n' + '='*100)
print('DETAILED PERFORMANCE ANALYSIS')
print('='*100)

# Group by base model
base_models = df['Base Model'].unique()
for base_model in base_models:
    base_df = df[df['Base Model'] == base_model].sort_values('Throughput (FPS)', ascending=False)
    if len(base_df) > 0:
        print(f'\\n{base_model.upper()}:')
        for _, row in base_df.iterrows():
            print(f'  {row[\"Variant\"]}: {row[\"Size (MB)\"]:.1f}MB, {row[\"Throughput (FPS)\"]:.1f}FPS, {row[\"Avg Inference (ms)\"]:.1f}ms')

# Find best performers
print('\\n' + '='*100)
print('BEST PERFORMERS BY CATEGORY')
print('='*100)

# Only consider models with successful runs (FPS > 0)
valid_df = df[df['Throughput (FPS)'] > 0]

if len(valid_df) > 0:
    best_throughput = valid_df.loc[valid_df['Throughput (FPS)'].idxmax()]
    best_size = valid_df.loc[valid_df['Size (MB)'].idxmin()]
    best_inference = valid_df.loc[valid_df['Avg Inference (ms)'].idxmin()]

    print(f'\\nBest Throughput: {best_throughput[\"Model\"]}')
    print(f'  - {best_throughput[\"Throughput (FPS)\"]:.2f} FPS')
    print(f'  - {best_throughput[\"Size (MB)\"]:.2f} MB')
    print(f'  - {best_throughput[\"Avg Inference (ms)\"]:.2f} ms avg inference')

    print(f'\\nSmallest Size: {best_size[\"Model\"]}')
    print(f'  - {best_size[\"Size (MB)\"]:.2f} MB')
    print(f'  - {best_size[\"Throughput (FPS)\"]:.2f} FPS')

    print(f'\\nFastest Inference: {best_inference[\"Model\"]}')
    print(f'  - {best_inference[\"Avg Inference (ms)\"]:.2f} ms')
    print(f'  - {best_inference[\"Throughput (FPS)\"]:.2f} FPS')

# Raspberry Pi Zero recommendations
print('\\n' + '='*100)
print('RASPBERRY PI ZERO RECOMMENDATIONS')
print('='*100)

# Filter for models suitable for Pi Zero (< 50MB, > 5 FPS, original variants preferred)
pi_suitable = valid_df[
    (valid_df['Size (MB)'] < 50) & 
    (valid_df['Throughput (FPS)'] > 5)
].sort_values('Throughput (FPS)', ascending=False)

if not pi_suitable.empty:
    print('\\nModels suitable for Raspberry Pi Zero (< 50MB, > 5 FPS):')
    print(pi_suitable[['Model', 'Size (MB)', 'Throughput (FPS)', 'Avg Inference (ms)']].to_string(index=False, float_format='%.2f'))
    
    # Separate analysis for original vs optimized
    original_models = pi_suitable[pi_suitable['Variant'] == 'original']
    optimized_models = pi_suitable[pi_suitable['Variant'].isin(['int8', 'mobile'])]
    
    if not original_models.empty:
        print('\\n  ORIGINAL MODELS (Recommended):')
        for _, row in original_models.iterrows():
            print(f'    {row[\"Base Model\"]}: {row[\"Size (MB)\"]:.1f}MB, {row[\"Throughput (FPS)\"]:.1f}FPS')
    
    if not optimized_models.empty:
        print('\\n  OPTIMIZED MODELS:')
        for _, row in optimized_models.iterrows():
            print(f'    {row[\"Model\"]}: {row[\"Size (MB)\"]:.1f}MB, {row[\"Throughput (FPS)\"]:.1f}FPS')
else:
    print('\\nNo models meet strict Pi Zero criteria (< 50MB, > 5 FPS)')
    print('Consider these lightweight options:')
    lightweight = valid_df[valid_df['Size (MB)'] < 100].sort_values('Throughput (FPS)', ascending=False)
    if not lightweight.empty:
        print(lightweight.head(5)[['Model', 'Size (MB)', 'Throughput (FPS)']].to_string(index=False, float_format='%.2f'))

# Model-specific recommendations
print('\\n' + '='*100)
print('MODEL-SPECIFIC RECOMMENDATIONS')
print('='*100)

model_recommendations = {
    'superpoint': 'Keypoint detection and matching',
    'mobilenetv2': 'General classification, best size/performance ratio',
    'mobilenetv3': 'Latest mobile architecture with optimizations',
    'efficientnet': 'Efficient classification with compound scaling',
    'resnet50': 'Traditional CNN, good accuracy but large',
    'dino': 'Self-supervised vision transformer',
    'dinov2': 'Latest self-supervised ViT with improved performance'
}

for base_model in base_models:
    if base_model in model_recommendations:
        base_df = valid_df[valid_df['Base Model'] == base_model]
        if not base_df.empty:
            best_variant = base_df.loc[base_df['Throughput (FPS)'].idxmax()]
            print(f'\\n{base_model.upper()}: {model_recommendations[base_model]}')
            print(f'  Best variant: {best_variant[\"Variant\"]} ({best_variant[\"Throughput (FPS)\"]:.1f} FPS, {best_variant[\"Size (MB)\"]:.1f} MB)')

# Save detailed CSV reports
csv_file = 'optimization_results/detailed_benchmark_results.csv'
df_sorted.to_csv(csv_file, index=False)

# Save summary CSV for easy analysis
summary_df = df_sorted[['Model', 'Base Model', 'Variant', 'Size (MB)', 'Throughput (FPS)', 'Avg Inference (ms)']].copy()
summary_csv = 'optimization_results/benchmark_summary.csv'
summary_df.to_csv(summary_csv, index=False)

print(f'\\n' + '='*100)
print('FILES GENERATED')
print('='*100)
print(f'Detailed results: {latest_file}')
print(f'Detailed CSV: {csv_file}')
print(f'Summary CSV: {summary_csv}')
print(f'Logs: logs/optimization_detailed.log')

print(f'\\n' + '='*100)
print('BENCHMARK COMPLETE!')
print('='*100)
print(f'Total models tested: {len(data)}')
print(f'Successful benchmarks: {len(valid_df)}')
print(f'Pi Zero suitable models: {len(pi_suitable)}')
" 2>&1 | tee logs/analysis_report.log

# Step 5: Generate final status
log "Benchmark pipeline completed!"

# Display final summary
echo ""
echo "============================================="
echo "BENCHMARK PIPELINE COMPLETE!"
echo "============================================="
echo ""
echo "📁 Generated Files:"
echo "   📊 optimization_results/detailed_benchmark_results.csv"
echo "   📋 optimization_results/benchmark_summary.csv"  
echo "   📄 optimization_results/optimization_results_*.json"
echo "   📝 logs/optimization_detailed.log"
echo "   📝 logs/analysis_report.log"

log "All files saved to optimization_results/ directory"
log "Pipeline execution complete!" 