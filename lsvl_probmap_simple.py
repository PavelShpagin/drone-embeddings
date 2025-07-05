#!/usr/bin/env python3
"""
LSVL Probability Map Implementation (Simplified)
Implements probability map updates using (2-c)/2 where c is cosine similarity
Visualizes drone flight and logs recall metrics
"""

import json
import time
import math
import random

class LSVLProbabilityMap:
    """LSVL-style probability map for GPS-denied localization"""
    
    def __init__(self, map_bounds=(-1000, 1000, -1000, 1000), grid_resolution=50):
        """Initialize probability map"""
        self.min_x, self.max_x, self.min_y, self.max_y = map_bounds
        self.resolution = grid_resolution
        
        # Calculate grid dimensions
        self.grid_width = int((self.max_x - self.min_x) / self.resolution)
        self.grid_height = int((self.max_y - self.min_y) / self.resolution)
        
        # Initialize uniform probability map
        total_cells = self.grid_width * self.grid_height
        uniform_prob = 1.0 / total_cells
        self.prob_map = [[uniform_prob for _ in range(self.grid_width)] 
                        for _ in range(self.grid_height)]
        
        # Track updates
        self.update_history = []
        
        print(f"📊 LSVL Probability Map initialized:")
        print(f"   • Grid size: {self.grid_width}x{self.grid_height}")
        print(f"   • Resolution: {self.resolution}m per cell")
        print(f"   • Total cells: {total_cells}")
        
    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid indices"""
        grid_x = int((world_x - self.min_x) / self.resolution)
        grid_y = int((world_y - self.min_y) / self.resolution)
        
        # Clamp to valid range
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates (cell center)"""
        world_x = self.min_x + (grid_x + 0.5) * self.resolution
        world_y = self.min_y + (grid_y + 0.5) * self.resolution
        return world_x, world_y
    
    def update_probabilities(self, observations):
        """
        Update probability map using LSVL approach with cosine similarity
        
        Args:
            observations: List of (world_x, world_y, cosine_similarity) tuples
        """
        if not observations:
            return
            
        # Create update map
        update_map = [[0.0 for _ in range(self.grid_width)] 
                     for _ in range(self.grid_height)]
        
        for world_x, world_y, cosine_sim in observations:
            grid_x, grid_y = self.world_to_grid(world_x, world_y)
            
            # LSVL probability update: (2-c)/2 where c is cosine similarity
            # Higher cosine similarity (closer to 1) -> lower update value -> higher confidence
            prob_update = (2 - cosine_sim) / 2
            
            # Add to update map
            update_map[grid_y][grid_x] += prob_update
        
        # Normalize update map to create probability distribution
        update_sum = sum(sum(row) for row in update_map)
        if update_sum > 0:
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    update_map[i][j] /= update_sum
            
            # Multiply current probabilities by update probabilities (Bayesian update)
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    self.prob_map[i][j] *= update_map[i][j]
            
            # Renormalize to maintain probability distribution
            prob_sum = sum(sum(row) for row in self.prob_map)
            if prob_sum > 0:
                for i in range(self.grid_height):
                    for j in range(self.grid_width):
                        self.prob_map[i][j] /= prob_sum
            else:
                # Fallback to uniform if all probabilities become zero
                uniform_prob = 1.0 / (self.grid_width * self.grid_height)
                for i in range(self.grid_height):
                    for j in range(self.grid_width):
                        self.prob_map[i][j] = uniform_prob
        
        # Track update for history
        max_prob = max(max(row) for row in self.prob_map)
        
        # Calculate entropy
        entropy = 0.0
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                p = self.prob_map[i][j]
                if p > 0:
                    entropy -= p * math.log(p)
        
        self.update_history.append({
            'max_probability': max_prob,
            'entropy': entropy,
            'num_observations': len(observations)
        })
    
    def get_most_likely_position(self):
        """Get the most likely position from probability map"""
        max_prob = 0
        max_i, max_j = 0, 0
        
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.prob_map[i][j] > max_prob:
                    max_prob = self.prob_map[i][j]
                    max_i, max_j = i, j
        
        world_x, world_y = self.grid_to_world(max_j, max_i)
        return world_x, world_y, max_prob
    
    def visualize_ascii(self, drone_pos=None, true_pos=None):
        """Create ASCII visualization of probability map"""
        # Downsample for ASCII display
        display_height = 20
        display_width = 40
        
        # Resize probability map for display
        y_step = max(1, self.grid_height // display_height)
        x_step = max(1, self.grid_width // display_width)
        
        display_map = [[0.0 for _ in range(display_width)] 
                      for _ in range(display_height)]
        
        for i in range(display_height):
            for j in range(display_width):
                y_start = i * y_step
                y_end = min((i + 1) * y_step, self.grid_height)
                x_start = j * x_step
                x_end = min((j + 1) * x_step, self.grid_width)
                
                # Average probability in this region
                total_prob = 0
                count = 0
                for y in range(y_start, y_end):
                    for x in range(x_start, x_end):
                        total_prob += self.prob_map[y][x]
                        count += 1
                
                if count > 0:
                    display_map[i][j] = total_prob / count
        
        # Normalize for display
        max_display = max(max(row) for row in display_map)
        if max_display > 0:
            for i in range(display_height):
                for j in range(display_width):
                    display_map[i][j] /= max_display
        
        # Create ASCII representation
        chars = " .:-=+*#%@"
        ascii_map = []
        
        for i in range(display_height):
            row = ""
            for j in range(display_width):
                # Convert probability to character
                char_idx = int(display_map[i][j] * (len(chars) - 1))
                char = chars[char_idx]
                
                # Add drone position if specified
                if drone_pos:
                    drone_grid_x, drone_grid_y = self.world_to_grid(drone_pos[0], drone_pos[1])
                    display_drone_x = drone_grid_x // x_step
                    display_drone_y = drone_grid_y // y_step
                    
                    if i == display_drone_y and j == display_drone_x:
                        char = 'D'
                
                # Add true position if specified
                if true_pos:
                    true_grid_x, true_grid_y = self.world_to_grid(true_pos[0], true_pos[1])
                    display_true_x = true_grid_x // x_step
                    display_true_y = true_grid_y // y_step
                    
                    if i == display_true_y and j == display_true_x:
                        char = 'T'
                
                row += char
            ascii_map.append(row)
        
        return ascii_map

class LSVLDroneSimulator:
    """Drone simulator using LSVL probability maps"""
    
    def __init__(self):
        self.prob_map = LSVLProbabilityMap()
        
        # Create reference database
        self.create_reference_database()
        
        # Simulation tracking
        self.results = {
            'recall_1': [],
            'recall_5': [],
            'position_errors': [],
            'confidence_scores': [],
            'entropy_history': []
        }
        
    def create_reference_database(self):
        """Create reference database with feature vectors"""
        print("💾 Creating reference database...")
        
        self.db_points = []
        self.db_features = []
        
        # Grid of reference points
        for x in range(-800, 801, 200):  # Every 200m
            for y in range(-800, 801, 200):
                self.db_points.append((x, y))
                
                # Simulate normalized feature vector
                feature = [random.gauss(0, 1) for _ in range(256)]
                norm = math.sqrt(sum(f*f for f in feature))
                if norm > 0:
                    feature = [f/norm for f in feature]
                self.db_features.append(feature)
        
        print(f"✅ Database created: {len(self.db_points)} reference points")
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a*b for a, b in zip(vec1, vec2))
        return dot_product  # Vectors are already normalized
    
    def simulate_observations(self, drone_pos, num_observations=5):
        """Simulate visual observations with cosine similarities"""
        observations = []
        
        # Find closest database points
        distances = []
        for db_x, db_y in self.db_points:
            dist = math.sqrt((drone_pos[0] - db_x)**2 + (drone_pos[1] - db_y)**2)
            distances.append(dist)
        
        # Get top observations (closest points)
        sorted_pairs = sorted(enumerate(distances), key=lambda x: x[1])
        closest_indices = [pair[0] for pair in sorted_pairs[:num_observations]]
        
        for idx in closest_indices:
            db_x, db_y = self.db_points[idx]
            db_feature = self.db_features[idx]
            
            # Simulate query feature (similar to database feature + noise)
            query_feature = []
            for f in db_feature:
                noisy_f = f + random.gauss(0, 0.1)
                query_feature.append(noisy_f)
            
            # Normalize query feature
            norm = math.sqrt(sum(f*f for f in query_feature))
            if norm > 0:
                query_feature = [f/norm for f in query_feature]
            
            # Calculate cosine similarity
            cosine_sim = self.cosine_similarity(query_feature, db_feature)
            
            # Add some noise based on distance
            distance = distances[idx]
            noise_factor = min(0.3, distance / 1000)  # More noise for distant points
            cosine_sim += random.gauss(0, noise_factor)
            cosine_sim = max(-1, min(1, cosine_sim))  # Clamp to valid range
            
            observations.append((db_x, db_y, cosine_sim))
        
        return observations
    
    def generate_trajectory(self, num_steps=50):
        """Generate realistic drone trajectory"""
        trajectory = []
        
        # Start at random position
        current_pos = [random.uniform(-400, 400), random.uniform(-400, 400)]
        trajectory.append(tuple(current_pos))
        
        # Waypoints for navigation
        waypoints = [
            (300, 200), (-200, 300), (400, -100), (-300, -200),
            (100, 400), (-400, 100), (200, -300), (0, 0)
        ]
        
        current_waypoint = 0
        
        for step in range(num_steps - 1):
            # Move towards current waypoint
            if current_waypoint < len(waypoints):
                target = waypoints[current_waypoint]
                
                dx = target[0] - current_pos[0]
                dy = target[1] - current_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 100:  # Close to waypoint
                    current_waypoint += 1
                
                if distance > 0:
                    step_size = 40
                    current_pos[0] += (dx / distance) * step_size + random.uniform(-20, 20)
                    current_pos[1] += (dy / distance) * step_size + random.uniform(-20, 20)
                else:
                    current_pos[0] += random.uniform(-30, 30)
                    current_pos[1] += random.uniform(-30, 30)
            else:
                # Random movement
                current_pos[0] += random.uniform(-40, 40)
                current_pos[1] += random.uniform(-40, 40)
            
            # Keep within bounds
            current_pos[0] = max(-900, min(900, current_pos[0]))
            current_pos[1] = max(-900, min(900, current_pos[1]))
            
            trajectory.append(tuple(current_pos))
        
        return trajectory
    
    def calculate_recall_metrics(self, estimated_pos, true_pos, tolerance=100):
        """Calculate recall metrics"""
        error = math.sqrt((estimated_pos[0] - true_pos[0])**2 + (estimated_pos[1] - true_pos[1])**2)
        
        recall_1 = 1.0 if error <= tolerance else 0.0
        recall_5 = 1.0 if error <= tolerance * 2 else 0.0  # More lenient for recall@5
        
        return recall_1, recall_5, error
    
    def run_simulation(self, num_steps=25):
        """Run LSVL probability map simulation"""
        print("🚁 Starting LSVL Probability Map Simulation")
        print("=" * 60)
        
        # Generate trajectory
        trajectory = self.generate_trajectory(num_steps)
        print(f"✅ Generated trajectory with {len(trajectory)} waypoints")
        
        # Run simulation
        for step in range(len(trajectory)):
            true_pos = trajectory[step]
            
            print(f"\n{'='*60}")
            print(f"🚁 LSVL SIMULATION - Step {step + 1}/{len(trajectory)}")
            print(f"{'='*60}")
            print(f"📍 True Position: ({true_pos[0]:.1f}, {true_pos[1]:.1f})m")
            
            # Simulate observations
            observations = self.simulate_observations(true_pos)
            print(f"👁️ Generated {len(observations)} visual observations")
            
            # Log observation details
            for i, (obs_x, obs_y, cos_sim) in enumerate(observations):
                prob_update = (2 - cos_sim) / 2
                print(f"   Obs {i+1}: ({obs_x}, {obs_y}) cos_sim={cos_sim:.3f} prob_update={prob_update:.3f}")
            
            # Update probability map
            self.prob_map.update_probabilities(observations)
            
            # Get most likely position
            est_x, est_y, max_prob = self.prob_map.get_most_likely_position()
            print(f"📊 Estimated Position: ({est_x:.1f}, {est_y:.1f})m (prob={max_prob:.4f})")
            
            # Calculate metrics
            recall_1, recall_5, error = self.calculate_recall_metrics((est_x, est_y), true_pos)
            
            print(f"📈 Metrics:")
            print(f"   • Position Error: {error:.1f}m")
            print(f"   • Recall@1 (100m): {'✅' if recall_1 else '❌'} {recall_1:.0f}")
            print(f"   • Recall@5 (200m): {'✅' if recall_5 else '❌'} {recall_5:.0f}")
            
            # Store results
            self.results['recall_1'].append(recall_1)
            self.results['recall_5'].append(recall_5)
            self.results['position_errors'].append(error)
            self.results['confidence_scores'].append(max_prob)
            
            if self.prob_map.update_history:
                self.results['entropy_history'].append(self.prob_map.update_history[-1]['entropy'])
            
            # Visualize probability map every few steps
            if step % 5 == 0 or step == len(trajectory) - 1:
                print(f"\n🗺️ LSVL Probability Map (Step {step + 1}):")
                print("Legend: D=Drone, T=True, @=High Prob, .=Low Prob")
                print("-" * 42)
                
                ascii_map = self.prob_map.visualize_ascii(drone_pos=(est_x, est_y), true_pos=true_pos)
                for row in ascii_map:
                    print(row)
                print("-" * 42)
                
                print(f"📊 Confidence Analysis:")
                print(f"   • Max Probability: {max_prob:.4f}")
                if self.results['entropy_history']:
                    print(f"   • Map entropy: {self.results['entropy_history'][-1]:.2f}")
            
            # Small delay for readability
            time.sleep(0.1)
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n{'='*60}")
        print("📊 FINAL LSVL SIMULATION RESULTS")
        print(f"{'='*60}")
        
        # Calculate final metrics
        if self.results['recall_1']:
            avg_recall_1 = sum(self.results['recall_1']) / len(self.results['recall_1'])
            avg_recall_5 = sum(self.results['recall_5']) / len(self.results['recall_5'])
            avg_error = sum(self.results['position_errors']) / len(self.results['position_errors'])
            avg_confidence = sum(self.results['confidence_scores']) / len(self.results['confidence_scores'])
            final_entropy = self.results['entropy_history'][-1] if self.results['entropy_history'] else 0
            
            print(f"\n🎯 LSVL Performance Metrics:")
            print(f"   • Average Recall@1 (100m): {avg_recall_1:.3f}")
            print(f"   • Average Recall@5 (200m): {avg_recall_5:.3f}")
            print(f"   • Average Position Error: {avg_error:.1f}m")
            print(f"   • Average Confidence: {avg_confidence:.4f}")
            print(f"   • Final Map Entropy: {final_entropy:.2f}")
            
            # Performance trend analysis
            if len(self.results['recall_1']) >= 6:
                early_recall = sum(self.results['recall_1'][:3]) / 3
                late_recall = sum(self.results['recall_1'][-3:]) / 3
                
                if late_recall > early_recall + 0.1:
                    trend = "📈 Improving"
                elif late_recall < early_recall - 0.1:
                    trend = "📉 Declining"
                else:
                    trend = "➡️ Stable"
                
                print(f"   • Performance Trend: {trend}")
            
            # Error distribution
            errors = self.results['position_errors']
            errors_sorted = sorted(errors)
            n = len(errors_sorted)
            median_error = errors_sorted[n//2] if n > 0 else 0
            p90_error = errors_sorted[int(n*0.9)] if n > 0 else 0
            
            print(f"\n📏 Error Distribution:")
            print(f"   • Min Error: {min(errors):.1f}m")
            print(f"   • Max Error: {max(errors):.1f}m")
            print(f"   • Median Error: {median_error:.1f}m")
            print(f"   • 90th Percentile: {p90_error:.1f}m")
            
            # Confidence analysis
            confidences = self.results['confidence_scores']
            print(f"\n🎯 Confidence Analysis:")
            print(f"   • Min Confidence: {min(confidences):.4f}")
            print(f"   • Max Confidence: {max(confidences):.4f}")
            print(f"   • Avg Confidence: {avg_confidence:.4f}")
        
        # LSVL Method Analysis
        print(f"\n🔬 LSVL Method Analysis:")
        print(f"   • Probability Update Formula: (2-c)/2 where c=cosine_similarity")
        print(f"   • High cosine similarity → Low update value → High confidence")
        print(f"   • Low cosine similarity → High update value → Low confidence")
        print(f"   • Bayesian updates maintain probability distribution")
        
        # Save detailed report
        report_data = {
            'simulation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'LSVL Probability Map',
            'update_formula': '(2-c)/2 where c=cosine_similarity',
            'results': self.results,
            'final_metrics': {
                'avg_recall_1': avg_recall_1 if self.results['recall_1'] else 0,
                'avg_recall_5': avg_recall_5 if self.results['recall_1'] else 0,
                'avg_position_error': avg_error if self.results['recall_1'] else 0,
                'avg_confidence': avg_confidence if self.results['recall_1'] else 0
            }
        }
        
        with open("lsvl_simulation_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved: lsvl_simulation_report.json")

def main():
    """Main simulation function"""
    print("🚁 LSVL Probability Map Simulation")
    print("🎯 GPS-Denied Localization with Cosine Similarity Updates")
    print("=" * 60)
    
    # Create and run simulator
    simulator = LSVLDroneSimulator()
    simulator.run_simulation(num_steps=20)
    
    print(f"\n✅ LSVL Simulation Complete!")
    print("📁 Check lsvl_simulation_report.json for detailed results")

if __name__ == "__main__":
    main()