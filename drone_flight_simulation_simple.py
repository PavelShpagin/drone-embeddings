#!/usr/bin/env python3
"""
Simplified Drone Flight Simulation: DINO Model Comparison
Text-based visualization of trajectories and recall metrics
"""

import json
import time
import random
import math

class SimpleDroneSimulator:
    """Simplified drone flight simulator for DINO model comparison"""
    
    def __init__(self):
        self.map_size = (100, 50)  # Text map size
        self.models = {
            "original": {
                "name": "Original ViT-B/14",
                "size_mb": 344,
                "query_time_ms": 20000,
                "recall_1_base": 0.75,
                "recall_5_base": 0.90,
                "symbol": "R",  # Red (Original)
                "compatible": False
            },
            "quantized": {
                "name": "ViT-S/14 INT8",
                "size_mb": 21,
                "query_time_ms": 3333,
                "recall_1_base": 0.71,
                "recall_5_base": 0.87,
                "symbol": "G",  # Green (Quantized)
                "compatible": True
            }
        }
        
        # Simulation results
        self.results = {}
        for model in self.models:
            self.results[model] = {
                "recall_1": [],
                "recall_5": [],
                "query_times": [],
                "positions": []
            }
        
        # Create reference database
        self.create_database()
    
    def create_database(self):
        """Create reference database points"""
        self.db_points = []
        # Grid of reference points every 200 meters
        for x in range(-1000, 1001, 200):
            for y in range(-1000, 1001, 200):
                self.db_points.append((x, y))
        print(f"📍 Database created: {len(self.db_points)} reference points")
    
    def generate_trajectory(self, num_steps=100):
        """Generate realistic drone trajectory"""
        trajectory = []
        
        # Start at origin
        current_pos = [0.0, 0.0]
        trajectory.append(tuple(current_pos))
        
        # Waypoints for realistic flight pattern
        waypoints = [
            (300, 200), (600, -100), (200, 400), (-300, 300),
            (-500, -200), (100, -400), (500, 100), (-100, -300)
        ]
        
        current_waypoint = 0
        
        for step in range(num_steps - 1):
            # Move towards current waypoint
            if current_waypoint < len(waypoints):
                target = waypoints[current_waypoint]
                
                # Calculate direction to target
                dx = target[0] - current_pos[0]
                dy = target[1] - current_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 50:  # Close to waypoint
                    current_waypoint += 1
                
                # Move towards target with some randomness
                if distance > 0:
                    step_size = 30
                    current_pos[0] += (dx / distance) * step_size + random.uniform(-10, 10)
                    current_pos[1] += (dy / distance) * step_size + random.uniform(-10, 10)
                else:
                    current_pos[0] += random.uniform(-20, 20)
                    current_pos[1] += random.uniform(-20, 20)
            else:
                # Random movement after waypoints
                current_pos[0] += random.uniform(-30, 30)
                current_pos[1] += random.uniform(-30, 30)
            
            # Keep within reasonable bounds
            current_pos[0] = max(-800, min(800, current_pos[0]))
            current_pos[1] = max(-800, min(800, current_pos[1]))
            
            trajectory.append(tuple(current_pos))
        
        return trajectory
    
    def simulate_localization(self, position, model_name):
        """Simulate localization for a given position and model"""
        model = self.models[model_name]
        
        # Find distances to all database points
        distances = []
        for db_x, db_y in self.db_points:
            dist = math.sqrt((position[0] - db_x)**2 + (position[1] - db_y)**2)
            distances.append(dist)
        
        # Add model-specific noise
        if model_name == "original":
            noise_factor = 0.9  # Better accuracy
        else:
            noise_factor = 1.1  # Slightly worse
        
        # Apply noise
        noisy_distances = []
        for dist in distances:
            noise = random.uniform(0.8, 1.2) * noise_factor
            noisy_distances.append(dist * noise)
        
        # Find closest matches
        sorted_pairs = sorted(enumerate(noisy_distances), key=lambda x: x[1])
        closest_indices = [pair[0] for pair in sorted_pairs[:5]]
        
        # Calculate actual distances for recall
        actual_distances = [distances[i] for i in closest_indices]
        
        # Recall metrics (within 150m tolerance)
        tolerance = 150
        recall_1 = 1.0 if actual_distances[0] <= tolerance else 0.0
        recall_5 = 1.0 if any(d <= tolerance for d in actual_distances) else 0.0
        
        # Apply model baseline performance
        recall_1 *= random.uniform(0.8, 1.0) * (model["recall_1_base"] / 0.75)
        recall_5 *= random.uniform(0.9, 1.0) * (model["recall_5_base"] / 0.90)
        
        # Clamp to [0, 1]
        recall_1 = max(0, min(1, recall_1))
        recall_5 = max(0, min(1, recall_5))
        
        # Simulate query time
        query_time = model["query_time_ms"] + random.uniform(-500, 500)
        
        return {
            "recall_1": recall_1,
            "recall_5": recall_5,
            "query_time_ms": max(100, query_time)
        }
    
    def create_text_map(self, trajectory, current_step):
        """Create ASCII map visualization"""
        width, height = self.map_size
        
        # Initialize map
        text_map = [['.' for _ in range(width)] for _ in range(height)]
        
        # Add reference points
        for db_x, db_y in self.db_points:
            # Convert world coordinates to map coordinates
            map_x = int((db_x + 1000) / 2000 * (width - 1))
            map_y = int((db_y + 1000) / 2000 * (height - 1))
            
            if 0 <= map_x < width and 0 <= map_y < height:
                text_map[map_y][map_x] = '+'
        
        # Add trajectory
        for i, pos in enumerate(trajectory[:current_step + 1]):
            map_x = int((pos[0] + 1000) / 2000 * (width - 1))
            map_y = int((pos[1] + 1000) / 2000 * (height - 1))
            
            if 0 <= map_x < width and 0 <= map_y < height:
                if i == current_step:
                    text_map[map_y][map_x] = 'D'  # Drone current position
                else:
                    text_map[map_y][map_x] = '*'  # Trajectory path
        
        return text_map
    
    def print_frame(self, trajectory, step, show_map=True):
        """Print current simulation frame"""
        print(f"\n{'='*80}")
        print(f"🚁 DRONE FLIGHT SIMULATION - Step {step}")
        print(f"{'='*80}")
        
        if step < len(trajectory):
            pos = trajectory[step]
            print(f"📍 Current Position: ({pos[0]:.1f}, {pos[1]:.1f})m")
        
        # Show performance metrics
        print(f"\n📊 CURRENT PERFORMANCE:")
        print("-" * 60)
        
        for model_name, model in self.models.items():
            if self.results[model_name]["recall_1"]:
                current_r1 = self.results[model_name]["recall_1"][-1]
                current_r5 = self.results[model_name]["recall_5"][-1]
                avg_r1 = sum(self.results[model_name]["recall_1"]) / len(self.results[model_name]["recall_1"])
                avg_r5 = sum(self.results[model_name]["recall_5"]) / len(self.results[model_name]["recall_5"])
                avg_time = sum(self.results[model_name]["query_times"]) / len(self.results[model_name]["query_times"])
                
                compat = "✅ Compatible" if model["compatible"] else "❌ Too Large"
                
                print(f"\n{model['name']}:")
                print(f"  Current R@1: {current_r1:.3f} | R@5: {current_r5:.3f}")
                print(f"  Average R@1: {avg_r1:.3f} | R@5: {avg_r5:.3f}")
                print(f"  Query Time: {avg_time:.0f}ms | Size: {model['size_mb']}MB")
                print(f"  Pi Zero: {compat}")
        
        # Show map every 10 steps
        if show_map and step % 10 == 0:
            print(f"\n🗺️ FLIGHT MAP (Step {step}):")
            print("Legend: D=Drone, *=Path, +=Database, .=Empty")
            print("-" * self.map_size[0])
            
            text_map = self.create_text_map(trajectory, step)
            for row in text_map:
                print(''.join(row))
            print("-" * self.map_size[0])
    
    def run_simulation(self, num_steps=80):
        """Run the complete simulation"""
        print("🚁 Starting Drone Flight Simulation")
        print("🎯 Comparing Original DINO vs Quantized ViT-S/14")
        print("=" * 60)
        
        # Generate trajectory
        trajectory = self.generate_trajectory(num_steps)
        print(f"✅ Generated trajectory with {len(trajectory)} waypoints")
        
        # Run simulation
        for step in range(0, num_steps, 5):  # Test every 5 steps
            if step < len(trajectory):
                position = trajectory[step]
                
                # Test both models
                for model_name in self.models.keys():
                    result = self.simulate_localization(position, model_name)
                    
                    self.results[model_name]["recall_1"].append(result["recall_1"])
                    self.results[model_name]["recall_5"].append(result["recall_5"])
                    self.results[model_name]["query_times"].append(result["query_time_ms"])
                    self.results[model_name]["positions"].append(position)
                
                # Print progress
                self.print_frame(trajectory, step, show_map=(step % 20 == 0))
                
                # Small delay for readability
                time.sleep(0.1)
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n{'='*80}")
        print("📊 FINAL SIMULATION RESULTS")
        print(f"{'='*80}")
        
        report_data = {
            "simulation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {}
        }
        
        # Calculate final metrics
        for model_name, model in self.models.items():
            if self.results[model_name]["recall_1"]:
                recalls_1 = self.results[model_name]["recall_1"]
                recalls_5 = self.results[model_name]["recall_5"]
                query_times = self.results[model_name]["query_times"]
                
                avg_r1 = sum(recalls_1) / len(recalls_1)
                avg_r5 = sum(recalls_5) / len(recalls_5)
                avg_time = sum(query_times) / len(query_times)
                
                print(f"\n🔍 {model['name']}:")
                print(f"   • Final Recall@1: {avg_r1:.3f}")
                print(f"   • Final Recall@5: {avg_r5:.3f}")
                print(f"   • Average Query Time: {avg_time:.0f}ms")
                print(f"   • Model Size: {model['size_mb']}MB")
                print(f"   • Pi Zero Ready: {'✅ YES' if model['compatible'] else '❌ NO'}")
                
                # Performance trend
                if len(recalls_1) >= 2:
                    trend_start = sum(recalls_1[:3]) / min(3, len(recalls_1))
                    trend_end = sum(recalls_1[-3:]) / min(3, len(recalls_1))
                    trend = "📈 Improving" if trend_end > trend_start else "📉 Declining" if trend_end < trend_start else "➡️ Stable"
                    print(f"   • Performance Trend: {trend}")
                
                report_data["models"][model_name] = {
                    "name": model["name"],
                    "avg_recall_1": avg_r1,
                    "avg_recall_5": avg_r5,
                    "avg_query_time_ms": avg_time,
                    "size_mb": model["size_mb"],
                    "pi_zero_compatible": model["compatible"],
                    "recall_history": recalls_1
                }
        
        # Comparison analysis
        if "original" in report_data["models"] and "quantized" in report_data["models"]:
            orig = report_data["models"]["original"]
            quant = report_data["models"]["quantized"]
            
            size_reduction = ((orig["size_mb"] - quant["size_mb"]) / orig["size_mb"]) * 100
            speed_improvement = orig["avg_query_time_ms"] / quant["avg_query_time_ms"]
            accuracy_retention = quant["avg_recall_1"] / orig["avg_recall_1"] if orig["avg_recall_1"] > 0 else 1.0
            
            print(f"\n🚀 QUANTIZATION IMPACT ANALYSIS:")
            print(f"   • Memory Reduction: {size_reduction:.1f}% ({orig['size_mb']}MB → {quant['size_mb']}MB)")
            print(f"   • Speed Improvement: {speed_improvement:.1f}x faster")
            print(f"   • Accuracy Retention: {accuracy_retention:.1%}")
            print(f"   • Deployment: Impossible → {'Feasible' if quant['pi_zero_compatible'] else 'Still Impossible'}")
        
        # Performance visualization
        print(f"\n📈 RECALL@1 PERFORMANCE VISUALIZATION:")
        self.create_performance_chart()
        
        # Recommendations
        print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS:")
        print("-" * 50)
        
        best_model = None
        best_score = 0
        
        for model_name, model_data in report_data["models"].items():
            model = self.models[model_name]
            
            # Calculate deployment score
            score = 0
            if model["compatible"]:
                score += 0.4  # Pi Zero compatibility
            if model_data["avg_query_time_ms"] < 5000:
                score += 0.3  # Reasonable speed
            score += model_data["avg_recall_1"] * 0.3  # Accuracy
            
            if score > best_score:
                best_score = score
                best_model = model_name
            
            recommendation = "🥇 RECOMMENDED" if score > 0.8 else "🥈 ACCEPTABLE" if score > 0.6 else "❌ NOT SUITABLE"
            print(f"   {model['name']}: {recommendation} (Score: {score:.2f})")
        
        if best_model:
            best = self.models[best_model]
            print(f"\n🏆 WINNER: {best['name']}")
            print(f"   Perfect choice for Pi Zero drone deployment!")
        
        # Save report
        with open("drone_flight_simulation_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved: drone_flight_simulation_report.json")
    
    def create_performance_chart(self):
        """Create ASCII performance chart"""
        chart_width = 60
        chart_height = 10
        
        for model_name, model in self.models.items():
            if self.results[model_name]["recall_1"]:
                recalls = self.results[model_name]["recall_1"]
                print(f"\n{model['name']} Recall@1:")
                
                # Create chart
                chart = []
                for i in range(chart_height):
                    row = []
                    threshold = 1.0 - (i / chart_height)
                    
                    for j in range(chart_width):
                        if j < len(recalls):
                            recall_idx = int((j / chart_width) * len(recalls))
                            if recalls[recall_idx] >= threshold:
                                row.append('█')
                            else:
                                row.append('░')
                        else:
                            row.append(' ')
                    chart.append(''.join(row))
                
                # Print chart
                for i, row in enumerate(chart):
                    label = f"{1.0 - (i / chart_height):.1f}"
                    print(f"{label:4} │{row}│")
                
                print(f"{'':4} └{'─' * chart_width}┘")
                print(f"{'':6}0{'':int(chart_width/2)-1}Time{'':int(chart_width/2)-4}End")

def main():
    """Main simulation function"""
    print("🚁 Simplified Drone Flight Simulation")
    print("🎯 DINO Model Comparison: Original vs Quantized")
    print("=" * 60)
    
    # Create and run simulator
    simulator = SimpleDroneSimulator()
    simulator.run_simulation(num_steps=60)
    
    print(f"\n✅ Simulation Complete!")
    print("📁 Check drone_flight_simulation_report.json for detailed results")

if __name__ == "__main__":
    main()