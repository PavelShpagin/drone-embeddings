#!/bin/bash

# Model Optimization and Benchmarking Script
# This script runs comprehensive model optimization for Raspberry Pi Zero deployment

set -e  # Exit on any error

echo "============================================="
echo "Model Optimization for Raspberry Pi Zero"
echo "============================================="

# Create results directory
mkdir -p optimization_results
mkdir -p logs

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a logs/optimization.log
}

log "Starting model optimization pipeline..."

# Step 1: Install dependencies
log "Installing required dependencies..."
pip install --quiet torch torchvision timm tqdm numpy matplotlib seaborn pandas psutil thop || {
    log "Failed to install dependencies"
    exit 1
}

# Step 2: Run the optimization script
log "Running model optimization..."
python3 optimize_models.py 2>&1 | tee logs/optimization_output.log

# Step 3: Generate summary report
log "Generating summary report..."
python3 -c "
import json
import glob
from pathlib import Path
import pandas as pd

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
            'Size (MB)': metrics.get('size_mb', 0),
            'Avg Inference (ms)': metrics.get('avg_inference_time_ms', 0),
            'Throughput (FPS)': metrics.get('throughput_fps', 0),
            'Memory (MB)': metrics.get('avg_memory_usage_mb', 0)
        })

if not data:
    print('No valid data found in results!')
    exit(1)

df = pd.DataFrame(data)

# Sort by throughput (descending)
df_sorted = df.sort_values('Throughput (FPS)', ascending=False)

print('\\n' + '='*80)
print('MODEL OPTIMIZATION RESULTS SUMMARY')
print('='*80)
print(df_sorted.to_string(index=False, float_format='%.2f'))

# Find best performers
print('\\n' + '='*80)
print('BEST PERFORMERS BY CATEGORY')
print('='*80)

best_throughput = df_sorted.iloc[0]
best_size = df.loc[df['Size (MB)'].idxmin()]
best_inference = df.loc[df['Avg Inference (ms)'].idxmin()]

print(f'\\nBest Throughput: {best_throughput[\"Model\"]}')
print(f'  - {best_throughput[\"Throughput (FPS)\"]:.2f} FPS')
print(f'  - {best_throughput[\"Size (MB)\"]:.2f} MB')

print(f'\\nSmallest Size: {best_size[\"Model\"]}')
print(f'  - {best_size[\"Size (MB)\"]:.2f} MB')
print(f'  - {best_size[\"Throughput (FPS)\"]:.2f} FPS')

print(f'\\nFastest Inference: {best_inference[\"Model\"]}')
print(f'  - {best_inference[\"Avg Inference (ms)\"]:.2f} ms')
print(f'  - {best_inference[\"Throughput (FPS)\"]:.2f} FPS')

# Raspberry Pi Zero recommendations
print('\\n' + '='*80)
print('RASPBERRY PI ZERO RECOMMENDATIONS')
print('='*80)

# Filter for models suitable for Pi Zero (< 50MB, > 10 FPS)
pi_suitable = df[(df['Size (MB)'] < 50) & (df['Throughput (FPS)'] > 10)]

if not pi_suitable.empty:
    pi_suitable_sorted = pi_suitable.sort_values('Throughput (FPS)', ascending=False)
    print('\\nModels suitable for Raspberry Pi Zero:')
    print(pi_suitable_sorted.to_string(index=False, float_format='%.2f'))
else:
    print('\\nNo models meet Pi Zero criteria (< 50MB, > 10 FPS)')
    print('Consider these lightweight options:')
    lightweight = df[df['Size (MB)'] < 100].sort_values('Throughput (FPS)', ascending=False)
    print(lightweight.head(3).to_string(index=False, float_format='%.2f'))

# Save CSV report
csv_file = 'optimization_results/summary_report.csv'
df_sorted.to_csv(csv_file, index=False)
print(f'\\nDetailed results saved to: {csv_file}')
print(f'Full results saved to: {latest_file}')
"

log "Optimization complete!"
log "Check optimization_results/ directory for detailed results"
log "Check logs/ directory for execution logs"

echo ""
echo "============================================="
echo "Optimization Pipeline Complete!"
echo "=============================================" 