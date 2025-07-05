#!/usr/bin/env python3
"""
Drone Flight Simulation: Original DINO vs Quantized ViT-S/14 Comparison
Visualizes trajectories, recall metrics, and performance differences
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

class DroneFlightSimulator:
    """Simulate drone flight with DINO model comparison"""
    
    def __init__(self, map_size=(2048, 2048)):
        self.map_size = map_size
        self.m_per_pixel = 2.0  # 2 meters per pixel
        self.crop_size = 224
        self.update_interval_steps = 10  # Update every 10 steps
        
        # Model specifications
        self.models = {
            "original": {
                "name": "Original ViT-B/14",
                "size_mb": 344,
                "query_time_ms": 20000,
                "recall_1_base": 0.75,
                "recall_5_base": 0.90,
                "color": (0, 0, 255),  # Red
                "compatible": False
            },
            "quantized": {
                "name": "ViT-S/14 INT8",
                "size_mb": 21,
                "query_time_ms": 3333,
                "recall_1_base": 0.71,
                "recall_5_base": 0.87,
                "color": (0, 255, 0),  # Green
                "compatible": True
            }
        }
        
        # Simulation state
        self.step = 0
        self.results = {model: {"recall_1": [], "recall_5": [], "query_times": [], "positions": []} 
                       for model in self.models.keys()}
        
        # Create synthetic map
        self.create_synthetic_map()
        self.create_database()
        
    def create_synthetic_map(self):
        """Create a synthetic satellite map for simulation"""
        print("🗺️ Creating synthetic map...")
        
        # Create base terrain
        map_img = np.random.randint(80, 120, (*self.map_size, 3), dtype=np.uint8)
        
        # Add roads
        for _ in range(20):
            start = (np.random.randint(0, self.map_size[0]), np.random.randint(0, self.map_size[1]))
            end = (np.random.randint(0, self.map_size[0]), np.random.randint(0, self.map_size[1]))
            cv2.line(map_img, start, end, (60, 60, 60), 8)
        
        # Add buildings
        for _ in range(50):
            x = np.random.randint(50, self.map_size[0] - 50)
            y = np.random.randint(50, self.map_size[1] - 50)
            w = np.random.randint(20, 80)
            h = np.random.randint(20, 80)
            color = (np.random.randint(40, 80), np.random.randint(40, 80), np.random.randint(40, 80))
            cv2.rectangle(map_img, (x, y), (x + w, y + h), color, -1)
        
        # Add vegetation
        for _ in range(100):
            center = (np.random.randint(0, self.map_size[0]), np.random.randint(0, self.map_size[1]))
            radius = np.random.randint(10, 30)
            color = (np.random.randint(40, 80), np.random.randint(100, 140), np.random.randint(40, 80))
            cv2.circle(map_img, center, radius, color, -1)
        
        self.map_image = map_img
        print(f"✅ Map created: {self.map_size[0]}x{self.map_size[1]} pixels")
    
    def create_database(self):
        """Create database of reference locations"""
        print("💾 Creating reference database...")
        
        self.db_locations = []
        grid_spacing = 200  # pixels
        
        for y in range(grid_spacing, self.map_size[1] - grid_spacing, grid_spacing):
            for x in range(grid_spacing, self.map_size[0] - grid_spacing, grid_spacing):
                # Convert to world coordinates
                world_x = (x - self.map_size[0] // 2) * self.m_per_pixel
                world_y = (y - self.map_size[1] // 2) * self.m_per_pixel
                self.db_locations.append((world_x, world_y, x, y))
        
        print(f"✅ Database created: {len(self.db_locations)} reference points")
    
    def world_to_pixel(self, world_pos):
        """Convert world coordinates to pixel coordinates"""
        x_px = int(world_pos[0] / self.m_per_pixel + self.map_size[0] // 2)
        y_px = int(world_pos[1] / self.m_per_pixel + self.map_size[1] // 2)
        return (x_px, y_px)
    
    def pixel_to_world(self, pixel_pos):
        """Convert pixel coordinates to world coordinates"""
        world_x = (pixel_pos[0] - self.map_size[0] // 2) * self.m_per_pixel
        world_y = (pixel_pos[1] - self.map_size[1] // 2) * self.m_per_pixel
        return (world_x, world_y)
    
    def generate_trajectory(self, start_pos, num_steps, step_size=20, noise_std=5):
        """Generate drone trajectory with realistic flight patterns"""
        trajectory = [np.array(start_pos)]
        
        # Add waypoints for more realistic flight
        waypoints = [
            (200, 200), (600, 300), (400, 600), (800, 500), 
            (300, 800), (700, 700), (100, 400), (500, 100)
        ]
        
        current_waypoint = 0
        
        for step in range(num_steps):
            current_pos = trajectory[-1]
            
            # Navigate towards current waypoint
            if current_waypoint < len(waypoints):
                target = np.array(waypoints[current_waypoint])
                direction = target - current_pos
                distance = np.linalg.norm(direction)
                
                if distance < 50:  # Close to waypoint, move to next
                    current_waypoint += 1
                
                if distance > 0:
                    direction = direction / distance * step_size
                else:
                    direction = np.array([step_size, 0])
            else:
                # Random movement after waypoints
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)]) * step_size
            
            # Add noise
            noise = np.random.normal(0, noise_std, 2)
            next_pos = current_pos + direction + noise
            
            # Keep within bounds
            next_pos[0] = np.clip(next_pos[0], 100, self.map_size[0] - 100)
            next_pos[1] = np.clip(next_pos[1], 100, self.map_size[1] - 100)
            
            trajectory.append(next_pos)
        
        return np.array(trajectory)
    
    def simulate_localization(self, query_pos, model_name):
        """Simulate localization performance for a given model"""
        model = self.models[model_name]
        
        # Simulate query processing time
        start_time = time.time()
        
        # Calculate distances to all database points
        distances = []
        for db_x, db_y, _, _ in self.db_locations:
            dist = np.sqrt((query_pos[0] - db_x)**2 + (query_pos[1] - db_y)**2)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Add model-specific noise to simulate accuracy differences
        if model_name == "original":
            noise_factor = 0.8  # Better accuracy
        else:
            noise_factor = 1.1  # Slightly worse but still good
        
        noisy_distances = distances * (1 + np.random.normal(0, 0.1) * noise_factor)
        
        # Find closest matches
        sorted_indices = np.argsort(noisy_distances)
        
        # Calculate recall metrics
        spatial_tolerance = 100  # meters
        
        # Recall@1
        closest_dist = distances[sorted_indices[0]]
        recall_1 = 1.0 if closest_dist <= spatial_tolerance else 0.0
        
        # Recall@5
        top5_dists = distances[sorted_indices[:5]]
        recall_5 = 1.0 if np.any(top5_dists <= spatial_tolerance) else 0.0
        
        # Add some randomness based on model characteristics
        recall_1 *= np.random.uniform(0.8, 1.0) * (model["recall_1_base"] / 0.75)
        recall_5 *= np.random.uniform(0.9, 1.0) * (model["recall_5_base"] / 0.90)
        
        # Simulate processing time
        processing_time = model["query_time_ms"] + np.random.normal(0, model["query_time_ms"] * 0.1)
        
        return {
            "recall_1": min(recall_1, 1.0),
            "recall_5": min(recall_5, 1.0),
            "query_time_ms": max(processing_time, 100),
            "closest_match": self.db_locations[sorted_indices[0]]
        }
    
    def create_visualization_frame(self, trajectory, current_step):
        """Create visualization frame showing both models' performance"""
        # Create larger frame for side-by-side comparison
        frame_width = 1920
        frame_height = 1080
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Map area (left side)
        map_area_width = 800
        map_area_height = 800
        map_x_offset = 50
        map_y_offset = 50
        
        # Resize map to fit
        map_resized = cv2.resize(self.map_image, (map_area_width, map_area_height))
        frame[map_y_offset:map_y_offset + map_area_height, 
              map_x_offset:map_x_offset + map_area_width] = map_resized
        
        # Scale trajectory to map area
        scale_x = map_area_width / self.map_size[0]
        scale_y = map_area_height / self.map_size[1]
        
        # Draw trajectory
        if current_step > 1:
            for i in range(1, min(current_step + 1, len(trajectory))):
                pt1 = (int(trajectory[i-1][0] * scale_x) + map_x_offset,
                       int(trajectory[i-1][1] * scale_y) + map_y_offset)
                pt2 = (int(trajectory[i][0] * scale_x) + map_x_offset,
                       int(trajectory[i][1] * scale_y) + map_y_offset)
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
        
        # Draw current position
        if current_step < len(trajectory):
            current_pos = (int(trajectory[current_step][0] * scale_x) + map_x_offset,
                          int(trajectory[current_step][1] * scale_y) + map_y_offset)
            cv2.circle(frame, current_pos, 8, (0, 255, 255), -1)  # Yellow drone
        
        # Draw database points
        for db_x, db_y, px, py in self.db_locations:
            db_pos = (int(px * scale_x) + map_x_offset,
                     int(py * scale_y) + map_y_offset)
            cv2.circle(frame, db_pos, 3, (100, 100, 100), -1)
        
        # Right side: Performance metrics
        metrics_x = 900
        
        # Title
        cv2.putText(frame, "DINO Model Comparison", (metrics_x, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Model comparison table
        y_pos = 120
        for i, (model_name, model) in enumerate(self.models.items()):
            color = model["color"]
            
            # Model name
            cv2.putText(frame, f"{model['name']}", (metrics_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_pos += 30
            
            # Model specs
            cv2.putText(frame, f"  Size: {model['size_mb']}MB", (metrics_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 25
            
            cv2.putText(frame, f"  Query: {model['query_time_ms']/1000:.1f}s", (metrics_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 25
            
            # Current metrics
            if self.results[model_name]["recall_1"]:
                current_r1 = self.results[model_name]["recall_1"][-1]
                current_r5 = self.results[model_name]["recall_5"][-1]
                avg_r1 = np.mean(self.results[model_name]["recall_1"])
                avg_r5 = np.mean(self.results[model_name]["recall_5"])
                
                cv2.putText(frame, f"  R@1: {current_r1:.3f} (avg: {avg_r1:.3f})", 
                           (metrics_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_pos += 25
                
                cv2.putText(frame, f"  R@5: {current_r5:.3f} (avg: {avg_r5:.3f})", 
                           (metrics_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_pos += 25
            
            # Pi Zero compatibility
            compat_text = "Pi Zero: Compatible" if model["compatible"] else "Pi Zero: Too Large"
            compat_color = (0, 255, 0) if model["compatible"] else (0, 0, 255)
            cv2.putText(frame, f"  {compat_text}", (metrics_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, compat_color, 1)
            y_pos += 50
        
        # Performance graphs
        if self.step > 10:
            self.draw_performance_graphs(frame, metrics_x, y_pos)
        
        # Flight info
        flight_info_y = frame_height - 150
        cv2.putText(frame, f"Flight Step: {current_step}", (50, flight_info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if current_step < len(trajectory):
            world_pos = self.pixel_to_world(trajectory[current_step])
            cv2.putText(frame, f"Position: ({world_pos[0]:.1f}, {world_pos[1]:.1f})m", 
                       (50, flight_info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def draw_performance_graphs(self, frame, x_offset, y_offset):
        """Draw performance graphs on the frame"""
        graph_width = 300
        graph_height = 150
        
        # Recall@1 graph
        cv2.putText(frame, "Recall@1 Over Time", (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        graph_y = y_offset + 30
        
        # Draw graph background
        cv2.rectangle(frame, (x_offset, graph_y), 
                     (x_offset + graph_width, graph_y + graph_height), (50, 50, 50), -1)
        
        # Draw grid
        for i in range(0, graph_width, 50):
            cv2.line(frame, (x_offset + i, graph_y), 
                    (x_offset + i, graph_y + graph_height), (70, 70, 70), 1)
        for i in range(0, graph_height, 30):
            cv2.line(frame, (x_offset, graph_y + i), 
                    (x_offset + graph_width, graph_y + i), (70, 70, 70), 1)
        
        # Plot recall curves
        for model_name, model in self.models.items():
            if len(self.results[model_name]["recall_1"]) > 1:
                points = []
                recalls = self.results[model_name]["recall_1"]
                
                for i, recall in enumerate(recalls):
                    x = int((i / max(len(recalls) - 1, 1)) * graph_width) + x_offset
                    y = int((1 - recall) * graph_height) + graph_y
                    points.append((x, y))
                
                # Draw line
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], model["color"], 2)
        
        # Legend
        legend_y = graph_y + graph_height + 20
        for i, (model_name, model) in enumerate(self.models.items()):
            legend_x = x_offset + i * 150
            cv2.line(frame, (legend_x, legend_y), (legend_x + 20, legend_y), model["color"], 3)
            cv2.putText(frame, model["name"], (legend_x + 25, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run_simulation(self, num_steps=200, save_video=True):
        """Run the complete simulation"""
        print("🚁 Starting drone flight simulation...")
        
        # Generate trajectory
        start_pos = (self.map_size[0] // 2, self.map_size[1] // 2)
        trajectory = self.generate_trajectory(start_pos, num_steps)
        
        # Video setup
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_out = cv2.VideoWriter('drone_flight_comparison.avi', fourcc, 10, (1920, 1080))
        
        # Run simulation
        for step in range(num_steps):
            self.step = step
            
            # Perform localization every few steps
            if step % self.update_interval_steps == 0 and step > 0:
                current_world_pos = self.pixel_to_world(trajectory[step])
                
                # Test both models
                for model_name in self.models.keys():
                    result = self.simulate_localization(current_world_pos, model_name)
                    
                    self.results[model_name]["recall_1"].append(result["recall_1"])
                    self.results[model_name]["recall_5"].append(result["recall_5"])
                    self.results[model_name]["query_times"].append(result["query_time_ms"])
                    self.results[model_name]["positions"].append(current_world_pos)
            
            # Create visualization frame
            frame = self.create_visualization_frame(trajectory, step)
            
            # Save frame
            if save_video:
                video_out.write(frame)
            
            # Display progress
            if step % 20 == 0:
                print(f"  Step {step}/{num_steps} ({step/num_steps*100:.1f}%)")
        
        if save_video:
            video_out.release()
            print(f"✅ Video saved: drone_flight_comparison.avi")
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate final comparison report"""
        print("\n📊 SIMULATION RESULTS")
        print("=" * 60)
        
        report_data = {
            "simulation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_steps": self.step,
            "models": {}
        }
        
        for model_name, model in self.models.items():
            if self.results[model_name]["recall_1"]:
                avg_r1 = np.mean(self.results[model_name]["recall_1"])
                avg_r5 = np.mean(self.results[model_name]["recall_5"])
                avg_time = np.mean(self.results[model_name]["query_times"])
                
                print(f"\n🔍 {model['name']}:")
                print(f"   • Average Recall@1: {avg_r1:.3f}")
                print(f"   • Average Recall@5: {avg_r5:.3f}")
                print(f"   • Average Query Time: {avg_time:.0f}ms")
                print(f"   • Model Size: {model['size_mb']}MB")
                print(f"   • Pi Zero Compatible: {'✅ YES' if model['compatible'] else '❌ NO'}")
                
                report_data["models"][model_name] = {
                    "name": model["name"],
                    "avg_recall_1": avg_r1,
                    "avg_recall_5": avg_r5,
                    "avg_query_time_ms": avg_time,
                    "size_mb": model["size_mb"],
                    "pi_zero_compatible": model["compatible"],
                    "recall_1_history": self.results[model_name]["recall_1"],
                    "recall_5_history": self.results[model_name]["recall_5"]
                }
        
        # Comparison
        if len(report_data["models"]) == 2:
            orig = report_data["models"]["original"]
            quant = report_data["models"]["quantized"]
            
            size_reduction = ((orig["size_mb"] - quant["size_mb"]) / orig["size_mb"]) * 100
            speed_improvement = orig["avg_query_time_ms"] / quant["avg_query_time_ms"]
            accuracy_retention = quant["avg_recall_1"] / orig["avg_recall_1"]
            
            print(f"\n🚀 QUANTIZATION IMPACT:")
            print(f"   • Size reduction: {size_reduction:.1f}%")
            print(f"   • Speed improvement: {speed_improvement:.1f}x")
            print(f"   • Accuracy retention: {accuracy_retention:.1%}")
            print(f"   • Deployment feasibility: Impossible → Possible")
        
        # Save report
        with open("drone_simulation_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Report saved: drone_simulation_report.json")
        print(f"🎬 Video saved: drone_flight_comparison.avi")

def main():
    """Main simulation function"""
    print("🚁 Drone Flight Simulation: DINO Model Comparison")
    print("=" * 60)
    
    # Create simulator
    simulator = DroneFlightSimulator()
    
    # Run simulation
    simulator.run_simulation(num_steps=150, save_video=True)
    
    print("\n✅ Simulation complete!")
    print("📁 Generated files:")
    print("   • drone_flight_comparison.avi - Flight visualization")
    print("   • drone_simulation_report.json - Performance metrics")

if __name__ == "__main__":
    main()