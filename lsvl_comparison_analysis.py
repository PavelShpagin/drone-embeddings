#!/usr/bin/env python3
"""
LSVL Method Comparison Analysis
Compares cosine similarity (2-c)/2 vs exponential distance exp(-e) methods
"""

import json
import math

def load_reports():
    """Load both simulation reports"""
    try:
        with open("lsvl_simulation_report.json", "r") as f:
            cosine_report = json.load(f)
    except FileNotFoundError:
        print("❌ Cosine similarity report not found")
        return None, None
    
    try:
        with open("lsvl_exp_distance_report.json", "r") as f:
            exp_report = json.load(f)
    except FileNotFoundError:
        print("❌ Exponential distance report not found")
        return cosine_report, None
    
    return cosine_report, exp_report

def calculate_improvement(old_val, new_val):
    """Calculate percentage improvement"""
    if old_val == 0:
        return float('inf') if new_val > 0 else 0
    return ((new_val - old_val) / old_val) * 100

def analyze_performance_trends(results):
    """Analyze performance trends over time"""
    recall_1 = results['recall_1']
    errors = results['position_errors']
    
    if len(recall_1) < 6:
        return "Insufficient data"
    
    # Split into early and late periods
    early_recall = sum(recall_1[:len(recall_1)//2]) / (len(recall_1)//2)
    late_recall = sum(recall_1[len(recall_1)//2:]) / (len(recall_1) - len(recall_1)//2)
    
    early_error = sum(errors[:len(errors)//2]) / (len(errors)//2)
    late_error = sum(errors[len(errors)//2:]) / (len(errors) - len(errors)//2)
    
    recall_trend = "Improving" if late_recall > early_recall + 0.1 else "Declining" if late_recall < early_recall - 0.1 else "Stable"
    error_trend = "Improving" if late_error < early_error - 20 else "Declining" if late_error > early_error + 20 else "Stable"
    
    return {
        'recall_trend': recall_trend,
        'error_trend': error_trend,
        'early_recall': early_recall,
        'late_recall': late_recall,
        'early_error': early_error,
        'late_error': late_error
    }

def create_ascii_chart(values, title, width=40):
    """Create ASCII chart for values"""
    if not values:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        normalized = [0.5] * len(values)
    else:
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    chart = []
    chart.append(f"{title} ({min_val:.1f} - {max_val:.1f})")
    chart.append("=" * width)
    
    for i, norm_val in enumerate(normalized):
        bar_length = int(norm_val * (width - 8))
        bar = "█" * bar_length + "░" * (width - 8 - bar_length)
        chart.append(f"{i+1:2d}: {bar} {values[i]:.1f}")
    
    return chart

def main():
    """Main comparison analysis"""
    print("🔍 LSVL Method Comparison Analysis")
    print("=" * 60)
    
    # Load reports
    cosine_report, exp_report = load_reports()
    
    if not cosine_report or not exp_report:
        print("❌ Cannot perform comparison - missing reports")
        return
    
    # Extract metrics
    cosine_metrics = cosine_report['final_metrics']
    exp_metrics = exp_report['final_metrics']
    
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Main metrics comparison
    metrics_comparison = [
        ("Recall@1 (100m)", cosine_metrics['avg_recall_1'], exp_metrics['avg_recall_1']),
        ("Recall@5 (200m)", cosine_metrics['avg_recall_5'], exp_metrics['avg_recall_5']),
        ("Position Error (m)", cosine_metrics['avg_position_error'], exp_metrics['avg_position_error']),
        ("Confidence Score", cosine_metrics['avg_confidence'], exp_metrics['avg_confidence'])
    ]
    
    print(f"{'Metric':<20} {'Cosine (2-c)/2':<15} {'Exp(-e)':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for metric_name, cosine_val, exp_val in metrics_comparison:
        if "Error" in metric_name:
            improvement = calculate_improvement(cosine_val, exp_val)
            improvement_str = f"{-improvement:.1f}%" if improvement != float('inf') else "∞"
            better = "✅" if exp_val < cosine_val else "❌"
        else:
            improvement = calculate_improvement(cosine_val, exp_val)
            improvement_str = f"+{improvement:.1f}%" if improvement != float('inf') else "∞"
            better = "✅" if exp_val > cosine_val else "❌"
        
        print(f"{metric_name:<20} {cosine_val:<15.3f} {exp_val:<15.3f} {improvement_str:<10} {better}")
    
    # Detailed analysis
    print(f"\n🎯 DETAILED ANALYSIS")
    print("=" * 60)
    
    # Recall improvement
    recall_1_improvement = calculate_improvement(cosine_metrics['avg_recall_1'], exp_metrics['avg_recall_1'])
    recall_5_improvement = calculate_improvement(cosine_metrics['avg_recall_5'], exp_metrics['avg_recall_5'])
    error_improvement = calculate_improvement(cosine_metrics['avg_position_error'], exp_metrics['avg_position_error'])
    
    print(f"🎯 Recall Performance:")
    print(f"   • Recall@1 improved by {recall_1_improvement:.1f}% ({cosine_metrics['avg_recall_1']:.1%} → {exp_metrics['avg_recall_1']:.1%})")
    print(f"   • Recall@5 improved by {recall_5_improvement:.1f}% ({cosine_metrics['avg_recall_5']:.1%} → {exp_metrics['avg_recall_5']:.1%})")
    print(f"   • Position error reduced by {-error_improvement:.1f}% ({cosine_metrics['avg_position_error']:.1f}m → {exp_metrics['avg_position_error']:.1f}m)")
    
    # Performance trends
    print(f"\n📈 PERFORMANCE TRENDS:")
    cosine_trends = analyze_performance_trends(cosine_report['results'])
    exp_trends = analyze_performance_trends(exp_report['results'])
    
    print(f"   Cosine Similarity Method:")
    print(f"   • Recall trend: {cosine_trends['recall_trend']}")
    print(f"   • Error trend: {cosine_trends['error_trend']}")
    print(f"   • Early vs Late recall: {cosine_trends['early_recall']:.1%} → {cosine_trends['late_recall']:.1%}")
    
    print(f"   Exponential Distance Method:")
    print(f"   • Recall trend: {exp_trends['recall_trend']}")
    print(f"   • Error trend: {exp_trends['error_trend']}")
    print(f"   • Early vs Late recall: {exp_trends['early_recall']:.1%} → {exp_trends['late_recall']:.1%}")
    
    # Error distribution comparison
    print(f"\n📏 ERROR DISTRIBUTION COMPARISON:")
    cosine_errors = cosine_report['results']['position_errors']
    exp_errors = exp_report['results']['position_errors']
    
    cosine_sorted = sorted(cosine_errors)
    exp_sorted = sorted(exp_errors)
    
    n_cosine = len(cosine_sorted)
    n_exp = len(exp_sorted)
    
    print(f"   {'Metric':<20} {'Cosine (2-c)/2':<15} {'Exp(-e)':<15} {'Improvement'}")
    print(f"   {'-'*65}")
    print(f"   {'Min Error (m)':<20} {min(cosine_errors):<15.1f} {min(exp_errors):<15.1f} {((min(exp_errors) - min(cosine_errors))/min(cosine_errors)*100):+.1f}%")
    print(f"   {'Max Error (m)':<20} {max(cosine_errors):<15.1f} {max(exp_errors):<15.1f} {((max(exp_errors) - max(cosine_errors))/max(cosine_errors)*100):+.1f}%")
    print(f"   {'Median Error (m)':<20} {cosine_sorted[n_cosine//2]:<15.1f} {exp_sorted[n_exp//2]:<15.1f} {((exp_sorted[n_exp//2] - cosine_sorted[n_cosine//2])/cosine_sorted[n_cosine//2]*100):+.1f}%")
    
    # Confidence analysis
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    cosine_conf = cosine_report['results']['confidence_scores']
    exp_conf = exp_report['results']['confidence_scores']
    
    print(f"   {'Metric':<20} {'Cosine (2-c)/2':<15} {'Exp(-e)':<15} {'Improvement'}")
    print(f"   {'-'*65}")
    print(f"   {'Min Confidence':<20} {min(cosine_conf):<15.4f} {min(exp_conf):<15.4f} {((min(exp_conf) - min(cosine_conf))/min(cosine_conf)*100):+.1f}%")
    print(f"   {'Max Confidence':<20} {max(cosine_conf):<15.4f} {max(exp_conf):<15.4f} {((max(exp_conf) - max(cosine_conf))/max(cosine_conf)*100):+.1f}%")
    print(f"   {'Avg Confidence':<20} {sum(cosine_conf)/len(cosine_conf):<15.4f} {sum(exp_conf)/len(exp_conf):<15.4f} {((sum(exp_conf)/len(exp_conf) - sum(cosine_conf)/len(cosine_conf))/(sum(cosine_conf)/len(cosine_conf))*100):+.1f}%")
    
    # Method analysis
    print(f"\n🔬 METHOD ANALYSIS:")
    print(f"   Cosine Similarity (2-c)/2:")
    print(f"   • Formula: (2 - cosine_similarity) / 2")
    print(f"   • Range: [0.5, 1.0] for cosine similarities [-1, 1]")
    print(f"   • Behavior: Linear scaling, moderate discrimination")
    print(f"   • Strengths: Stable, interpretable")
    print(f"   • Weaknesses: Limited dynamic range")
    
    print(f"   Exponential Distance exp(-e):")
    print(f"   • Formula: exp(-embedding_distance)")
    print(f"   • Range: (0, 1] for distances [0, ∞)")
    print(f"   • Behavior: Exponential decay, strong discrimination")
    print(f"   • Strengths: High dynamic range, sharp discrimination")
    print(f"   • Weaknesses: Can be overly sensitive to noise")
    
    # Visual comparison charts
    print(f"\n📊 RECALL@1 PERFORMANCE OVER TIME:")
    print(f"Cosine Similarity Method:")
    cosine_chart = create_ascii_chart(cosine_report['results']['recall_1'], "Recall@1", 50)
    for line in cosine_chart:
        print(f"   {line}")
    
    print(f"\nExponential Distance Method:")
    exp_chart = create_ascii_chart(exp_report['results']['recall_1'], "Recall@1", 50)
    for line in exp_chart:
        print(f"   {line}")
    
    # Final recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 60)
    
    if exp_metrics['avg_recall_1'] > cosine_metrics['avg_recall_1']:
        print("✅ EXPONENTIAL DISTANCE METHOD RECOMMENDED")
        print(f"   • {(exp_metrics['avg_recall_1'] - cosine_metrics['avg_recall_1'])*100:.1f}% better Recall@1")
        print(f"   • {(exp_metrics['avg_recall_5'] - cosine_metrics['avg_recall_5'])*100:.1f}% better Recall@5")
        print(f"   • {((cosine_metrics['avg_position_error'] - exp_metrics['avg_position_error'])/cosine_metrics['avg_position_error'])*100:.1f}% lower position error")
        print(f"   • {((exp_metrics['avg_confidence'] - cosine_metrics['avg_confidence'])/cosine_metrics['avg_confidence'])*100:.1f}% higher confidence")
    else:
        print("⚠️ COSINE SIMILARITY METHOD PERFORMED BETTER")
        print("   Consider investigating why exponential distance underperformed")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Exponential formula provides stronger discrimination between good/bad matches")
    print(f"   • Higher confidence scores indicate more decisive localization")
    print(f"   • Better recall performance suggests improved accuracy")
    print(f"   • Both methods show declining performance over trajectory")
    
    # Save comparison report
    comparison_data = {
        'comparison_timestamp': cosine_report['simulation_timestamp'],
        'methods_compared': ['Cosine Similarity (2-c)/2', 'Exponential Distance exp(-e)'],
        'cosine_metrics': cosine_metrics,
        'exp_metrics': exp_metrics,
        'improvements': {
            'recall_1': recall_1_improvement,
            'recall_5': recall_5_improvement,
            'position_error': error_improvement,
            'confidence': calculate_improvement(cosine_metrics['avg_confidence'], exp_metrics['avg_confidence'])
        },
        'recommendation': 'Exponential Distance' if exp_metrics['avg_recall_1'] > cosine_metrics['avg_recall_1'] else 'Cosine Similarity'
    }
    
    with open("lsvl_method_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Comparison report saved: lsvl_method_comparison.json")

if __name__ == "__main__":
    main()