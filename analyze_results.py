#!/usr/bin/env python3
"""
DINO Benchmark Results Analyzer
==============================

Analyze benchmark results and provide deployment recommendations for 
Raspberry Pi Zero based on specific use case requirements.
"""

import json
import sys
from typing import Dict, List

def load_results(filename: str = 'dino_benchmark_results.json') -> Dict:
    """Load benchmark results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file '{filename}' not found. Run the benchmark first.")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in '{filename}'")
        return None

def filter_compatible_models(results: List[Dict]) -> List[Dict]:
    """Filter models that are compatible with Pi Zero"""
    return [r for r in results if r['pi_zero_compatible']]

def find_best_model(results: List[Dict], criteria: str) -> Dict:
    """Find the best model based on specific criteria"""
    if not results:
        return None
    
    if criteria == 'accuracy':
        return max(results, key=lambda x: x['accuracy'])
    elif criteria == 'speed':
        return max(results, key=lambda x: x['throughput_fps'])
    elif criteria == 'size':
        return min(results, key=lambda x: x['model_size_mb'])
    elif criteria == 'memory':
        return min(results, key=lambda x: x['memory_peak_mb'])
    elif criteria == 'efficiency':
        return max(results, key=lambda x: x['accuracy'] / x['inference_time_ms'])
    else:
        return results[0]

def recommend_for_use_case(results: List[Dict], use_case: str) -> Dict:
    """Recommend best model configuration for specific use case"""
    compatible = filter_compatible_models(results)
    
    if not compatible:
        return None
    
    recommendations = {
        'real_time': {
            'priority': 'speed',
            'min_fps': 10,
            'description': 'Real-time processing (security, robotics)'
        },
        'batch_processing': {
            'priority': 'accuracy',
            'min_accuracy': 0.7,
            'description': 'Batch image analysis (offline processing)'
        },
        'iot_sensor': {
            'priority': 'size',
            'max_size_mb': 5,
            'description': 'IoT sensor with storage constraints'
        },
        'edge_inference': {
            'priority': 'efficiency',
            'min_efficiency': 5,
            'description': 'Edge inference with balanced requirements'
        },
        'prototype': {
            'priority': 'memory',
            'max_memory_mb': 50,
            'description': 'Prototype development with minimal resources'
        }
    }
    
    if use_case not in recommendations:
        return find_best_model(compatible, 'efficiency')
    
    req = recommendations[use_case]
    priority = req['priority']
    
    # Apply constraints
    filtered = compatible.copy()
    
    if 'min_fps' in req:
        filtered = [r for r in filtered if r['throughput_fps'] >= req['min_fps']]
    if 'min_accuracy' in req:
        filtered = [r for r in filtered if r['accuracy'] >= req['min_accuracy']]
    if 'max_size_mb' in req:
        filtered = [r for r in filtered if r['model_size_mb'] <= req['max_size_mb']]
    if 'max_memory_mb' in req:
        filtered = [r for r in filtered if r['memory_peak_mb'] <= req['max_memory_mb']]
    if 'min_efficiency' in req:
        filtered = [r for r in filtered if (r['accuracy'] / r['inference_time_ms'] * 1000) >= req['min_efficiency']]
    
    if not filtered:
        print(f"⚠️ No models meet all constraints for {use_case}. Showing best available...")
        filtered = compatible
    
    return find_best_model(filtered, priority)

def print_model_recommendation(model: Dict, use_case: str = None):
    """Print detailed model recommendation"""
    if not model:
        print("❌ No suitable model found")
        return
    
    print(f"🎯 {'Recommended Model' if use_case else 'Best Model'}")
    if use_case:
        print(f"   Use Case: {use_case}")
    print(f"   Model: {model['model_name']}")
    print(f"   Quantization: {model['quantization']}")
    print(f"   Accuracy: {model['accuracy']:.3f}")
    print(f"   Speed: {model['throughput_fps']:.1f} FPS")
    print(f"   Model Size: {model['model_size_mb']:.1f} MB")
    print(f"   Peak Memory: {model['memory_peak_mb']:.1f} MB")
    print(f"   Inference Time: {model['inference_time_ms']:.1f} ms")
    print(f"   CPU Usage: {model['cpu_usage_percent']:.1f}%")
    print(f"   Pi Zero Compatible: {'✅' if model['pi_zero_compatible'] else '❌'}")

def analyze_quantization_trade_offs(results: List[Dict]):
    """Analyze trade-offs between different quantization methods"""
    print("\n📊 QUANTIZATION TRADE-OFF ANALYSIS")
    print("=" * 50)
    
    # Group by quantization method
    by_quant = {}
    for result in results:
        quant = result['quantization']
        if quant not in by_quant:
            by_quant[quant] = []
        by_quant[quant].append(result)
    
    print(f"\n{'Method':<10} {'Avg Acc':<10} {'Avg Speed':<12} {'Avg Size':<12} {'Size Reduction'}")
    print("-" * 65)
    
    baseline_size = None
    for quant in ['none', '8bit', '4bit', 'binary']:
        if quant in by_quant:
            models = by_quant[quant]
            avg_acc = sum(m['accuracy'] for m in models) / len(models)
            avg_speed = sum(m['throughput_fps'] for m in models) / len(models)
            avg_size = sum(m['model_size_mb'] for m in models) / len(models)
            
            if quant == 'none':
                baseline_size = avg_size
                reduction = 0
            else:
                reduction = (1 - avg_size / baseline_size) * 100 if baseline_size else 0
            
            print(f"{quant:<10} {avg_acc:<10.3f} {avg_speed:<12.1f} {avg_size:<12.1f} {reduction:>6.1f}%")

def deployment_guide(results: List[Dict]):
    """Generate practical deployment guide"""
    print("\n🚀 DEPLOYMENT GUIDE")
    print("=" * 50)
    
    compatible = filter_compatible_models(results)
    
    if not compatible:
        print("❌ No models are compatible with Raspberry Pi Zero!")
        return
    
    print(f"\n✅ {len(compatible)} configurations are Pi Zero compatible\n")
    
    # Use case recommendations
    use_cases = {
        'real_time': 'Real-time Processing',
        'batch_processing': 'Batch Processing', 
        'iot_sensor': 'IoT Sensor',
        'edge_inference': 'Edge Inference',
        'prototype': 'Prototype Development'
    }
    
    for use_case, description in use_cases.items():
        print(f"📋 {description}:")
        best_model = recommend_for_use_case(results, use_case)
        if best_model:
            print(f"   → {best_model['model_name']} ({best_model['quantization']})")
            print(f"     {best_model['accuracy']:.3f} acc, {best_model['throughput_fps']:.1f} FPS, {best_model['model_size_mb']:.1f} MB")
        else:
            print("   → No suitable model found")
        print()

def memory_optimization_tips(results: List[Dict]):
    """Provide memory optimization recommendations"""
    print("\n💾 MEMORY OPTIMIZATION TIPS")
    print("=" * 50)
    
    compatible = filter_compatible_models(results)
    min_memory = min(r['memory_peak_mb'] for r in compatible)
    max_memory = max(r['memory_peak_mb'] for r in compatible)
    
    print(f"\nMemory usage range: {min_memory:.1f} - {max_memory:.1f} MB")
    print(f"Pi Zero total memory: 512 MB")
    print(f"Available for models: ~400 MB (after OS)")
    
    print(f"\n🔧 Optimization Strategies:")
    print("1. Use smaller models (DINO-Micro/Tiny)")
    print("2. Apply quantization (8-bit recommended)")
    print("3. Process single images (batch_size=1)")
    print("4. Implement model pruning")
    print("5. Use memory-mapped model loading")
    print("6. Monitor memory usage during inference")
    
    # Show memory-efficient configurations
    memory_efficient = sorted(compatible, key=lambda x: x['memory_peak_mb'])[:3]
    print(f"\n🏆 Most Memory-Efficient Configurations:")
    for i, model in enumerate(memory_efficient, 1):
        print(f"{i}. {model['model_name']} ({model['quantization']}) - {model['memory_peak_mb']:.1f} MB")

def main():
    """Main analysis function"""
    print("🔍 DINO Benchmark Results Analysis")
    print("=" * 50)
    
    # Load results
    data = load_results()
    if not data:
        sys.exit(1)
    
    results = data['results']
    benchmark_info = data['benchmark_info']
    
    print(f"\nBenchmark Info:")
    print(f"Platform: {benchmark_info['platform']}")
    print(f"Timestamp: {benchmark_info['timestamp']}")
    print(f"Total Configurations: {benchmark_info['total_configurations']}")
    
    # Overall best performers
    print(f"\n🏆 OVERALL BEST PERFORMERS")
    print("=" * 50)
    
    compatible = filter_compatible_models(results)
    
    criteria_map = {
        'accuracy': 'Highest Accuracy',
        'speed': 'Fastest Inference',
        'size': 'Smallest Size',
        'memory': 'Lowest Memory',
        'efficiency': 'Best Efficiency'
    }
    
    for criteria, description in criteria_map.items():
        best = find_best_model(compatible, criteria)
        print(f"\n{description}:")
        if best:
            if criteria == 'efficiency':
                eff_score = best['accuracy'] / best['inference_time_ms'] * 1000
                print(f"   {best['model_name']} ({best['quantization']}) - {eff_score:.3f} acc/sec")
            else:
                key = {
                    'accuracy': 'accuracy',
                    'speed': 'throughput_fps', 
                    'size': 'model_size_mb',
                    'memory': 'memory_peak_mb'
                }.get(criteria, 'accuracy')
                print(f"   {best['model_name']} ({best['quantization']}) - {best[key]:.3f}")
    
    # Detailed analysis
    analyze_quantization_trade_offs(results)
    deployment_guide(results)
    memory_optimization_tips(results)
    
    # Interactive recommendation
    print(f"\n🎯 CUSTOM RECOMMENDATION")
    print("=" * 50)
    print("Enter your use case for a custom recommendation:")
    print("Options: real_time, batch_processing, iot_sensor, edge_inference, prototype")
    
    try:
        use_case = input("\nUse case (or 'exit'): ").strip().lower()
        if use_case == 'exit':
            print("Goodbye!")
            return
        elif use_case in ['real_time', 'batch_processing', 'iot_sensor', 'edge_inference', 'prototype']:
            print()
            best_model = recommend_for_use_case(results, use_case)
            print_model_recommendation(best_model, use_case)
        else:
            print("Invalid use case. Showing most efficient model...")
            best_model = find_best_model(compatible, 'efficiency')
            print_model_recommendation(best_model)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")

if __name__ == "__main__":
    main()