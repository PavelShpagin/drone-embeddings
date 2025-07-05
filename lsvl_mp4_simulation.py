#!/usr/bin/env python3
"""
Corrected LSVL Probability Map Simulation with MP4 Output
- Proper normalization after every update
- Update only when drone flies 50m from last update point
- MP4 video output with visualization
- Comprehensive error checking
"""

import json
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from pathlib import Path

class SimulatedSuperPoint:
    """Simulated SuperPoint for demonstration"""
    
    def __init__(self, weights_path=None):
        self.weights_loaded = weights_path is not None and Path(weights_path).exists()
        print(f"🔧 SuperPoint initialized (weights: {'✅' if self.weights_loaded else '❌'})")
    
    def detect(self, image, conf_thresh=0.01, nms_dist=4):
        """Simulate SuperPoint keypoint detection and descriptor extraction"""
        num_keypoints = random.randint(20, 80)
        
        keypoints = []
        scores = []
        descriptors = []
        
        for _ in range(num_keypoints):
            x = random.uniform(10, 246)
            y = random.uniform(10, 246)
            keypoints.append([x, y])
            
            score = random.uniform(conf_thresh, 1.0)
            scores.append(score)
            
            descriptor = [random.gauss(0, 1) for _ in range(256)]
            norm = math.sqrt(sum(d*d for d in descriptor))
            if norm > 0:
                descriptor = [d/norm for d in descriptor]
            descriptors.append(descriptor)
        
        return keypoints, scores, descriptors

class CorrectedLSVLMap:
    """Corrected LSVL probability map with proper normalization and update logic"""
    
    def __init__(self, map_bounds=(-1000, 1000, -1000, 1000), grid_resolution=50):
        """Initialize corrected probability map"""
        self.min_x, self.max_x, self.min_y, self.max_y = map_bounds
        self.resolution = grid_resolution
        
        # Calculate grid dimensions
        self.grid_width = int((self.max_x - self.min_x) / self.resolution)
        self.grid_height = int((self.max_y - self.min_y) / self.resolution)
        
        # Initialize uniform probability map
        total_cells = self.grid_width * self.grid_height
        uniform_prob = 1.0 / total_cells
        self.prob_map = np.full((self.grid_height, self.grid_width), uniform_prob, dtype=np.float64)
        
        # Verify initial normalization
        initial_sum = np.sum(self.prob_map)
        assert abs(initial_sum - 1.0) < 1e-10, f"Initial probability sum: {initial_sum}"
        
        # Track updates and last update position
        self.update_history = []
        self.last_update_pos = None
        self.update_threshold = 50.0  # 50m threshold
        
        # Initialize SuperPoint
        self.init_superpoint()
        
        print(f"📊 Corrected LSVL Map initialized:")
        print(f"   • Grid size: {self.grid_width}x{self.grid_height}")
        print(f"   • Resolution: {self.resolution}m per cell")
        print(f"   • Total cells: {total_cells}")
        print(f"   • Initial prob sum: {initial_sum:.10f}")
        print(f"   • Update threshold: {self.update_threshold}m")
        
    def init_superpoint(self):
        """Initialize SuperPoint with trained weights"""
        weight_paths = [
            "superpoint_uav_trained/superpoint_uav_epoch_20.pth",
            "superpoint_uav_trained/superpoint_uav_final.pth",
            "superpoint_uav_trained/superpoint_uav_epoch_15.pth",
            "superpoint_v1.pth"
        ]
        
        for weights_path in weight_paths:
            if Path(weights_path).exists():
                self.superpoint = SimulatedSuperPoint(weights_path)
                print(f"✅ SuperPoint loaded from: {weights_path}")
                return
        
        print("⚠️ No SuperPoint weights found, using simulated SuperPoint")
        self.superpoint = SimulatedSuperPoint(None)
    
    def should_update(self, current_pos):
        """Check if we should update based on 50m threshold"""
        if self.last_update_pos is None:
            return True
        
        distance = math.sqrt(
            (current_pos[0] - self.last_update_pos[0])**2 + 
            (current_pos[1] - self.last_update_pos[1])**2
        )
        
        return distance >= self.update_threshold
    
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
    
    def verify_normalization(self, stage=""):
        """Verify probability map normalization"""
        prob_sum = np.sum(self.prob_map)
        if abs(prob_sum - 1.0) > 1e-8:
            print(f"⚠️ Normalization warning {stage}: sum = {prob_sum:.10f}")
            return False
        return True
    
    def normalize_probabilities(self):
        """Ensure probability map sums to 1.0"""
        prob_sum = np.sum(self.prob_map)
        if prob_sum > 0:
            self.prob_map /= prob_sum
        else:
            # Reset to uniform if all probabilities are zero
            uniform_prob = 1.0 / (self.grid_width * self.grid_height)
            self.prob_map.fill(uniform_prob)
        
        # Verify normalization
        final_sum = np.sum(self.prob_map)
        assert abs(final_sum - 1.0) < 1e-10, f"Failed normalization: {final_sum}"
    
    def update_probabilities_progressive(self, observations, query_image=None, drone_pos=None):
        """
        Update probabilities with progressive SuperPoint weighting and proper normalization
        """
        if not observations:
            return False
        
        # Check if we should update based on distance threshold
        if not self.should_update(drone_pos):
            return False
        
        print(f"🔄 Updating probabilities (moved {self.get_distance_from_last_update(drone_pos):.1f}m)")
        
        # Verify normalization before update
        self.verify_normalization("before update")
        
        # Step 1: Get top 5 candidates
        sorted_obs = sorted(observations, key=lambda x: x[2])  # Sort by embedding distance
        top_5_obs = sorted_obs[:5]
        other_obs = sorted_obs[5:]
        
        # Step 2: SuperPoint descriptor matching on top 5
        descriptor_distances = {}
        if self.superpoint and query_image is not None:
            descriptor_distances = self.compute_superpoint_matches(query_image, top_5_obs)
        
        # Step 3: Create update map with progressive weighting
        update_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float64)
        
        # Process top 5 candidates with progressive SuperPoint weighting
        for rank, (world_x, world_y, embedding_dist) in enumerate(top_5_obs, 1):
            grid_x, grid_y = self.world_to_grid(world_x, world_y)
            
            # Base exponential update
            base_update = math.exp(-embedding_dist)
            
            # Progressive SuperPoint weighting: exp(-rank × desc_dist)
            if (world_x, world_y) in descriptor_distances:
                desc_dist = descriptor_distances[(world_x, world_y)]
                superpoint_weight = math.exp(-rank * desc_dist)
                print(f"   Top{rank} ({world_x}, {world_y}): exp(-{rank}×{desc_dist:.3f})={superpoint_weight:.4f}")
            else:
                superpoint_weight = 1.0
            
            # Combined update
            prob_update = base_update * superpoint_weight
            update_map[grid_y, grid_x] += prob_update
        
        # Process other candidates with constant penalty exp(-10)
        constant_penalty = math.exp(-10)  # ≈ 0.000045
        for world_x, world_y, embedding_dist in other_obs:
            grid_x, grid_y = self.world_to_grid(world_x, world_y)
            base_update = math.exp(-embedding_dist)
            prob_update = base_update * constant_penalty
            update_map[grid_y, grid_x] += prob_update
        
        # Step 4: Normalize update map
        update_sum = np.sum(update_map)
        if update_sum > 0:
            update_map /= update_sum
        else:
            print("⚠️ Update map sum is zero, skipping update")
            return False
        
        # Verify update map normalization
        update_map_sum = np.sum(update_map)
        assert abs(update_map_sum - 1.0) < 1e-10, f"Update map not normalized: {update_map_sum}"
        
        # Step 5: Apply Bayesian update (element-wise multiplication)
        self.prob_map *= update_map
        
        # Step 6: CRITICAL - Normalize after update
        self.normalize_probabilities()
        
        # Verify final normalization
        self.verify_normalization("after update")
        
        # Update tracking
        self.last_update_pos = drone_pos
        
        # Track update for history
        max_prob = np.max(self.prob_map)
        entropy = -np.sum(self.prob_map * np.log(self.prob_map + 1e-15))
        
        self.update_history.append({
            'position': drone_pos,
            'max_probability': max_prob,
            'entropy': entropy,
            'num_observations': len(observations),
            'distance_moved': self.get_distance_from_last_update(drone_pos) if len(self.update_history) > 0 else 0
        })
        
        print(f"✅ Update complete: max_prob={max_prob:.4f}, entropy={entropy:.2f}")
        return True
    
    def get_distance_from_last_update(self, current_pos):
        """Get distance from last update position"""
        if self.last_update_pos is None:
            return 0.0
        return math.sqrt(
            (current_pos[0] - self.last_update_pos[0])**2 + 
            (current_pos[1] - self.last_update_pos[1])**2
        )
    
    def compute_superpoint_matches(self, query_image, top_5_obs):
        """Compute SuperPoint descriptor distances for top 5 candidates"""
        descriptor_distances = {}
        
        try:
            query_kpts, query_scores, query_descs = self.superpoint.detect(query_image)
            
            if len(query_descs) == 0:
                return descriptor_distances
            
            for world_x, world_y, _ in top_5_obs:
                db_descs = self.simulate_database_descriptors(world_x, world_y, len(query_descs))
                
                if len(db_descs) > 0:
                    distances = []
                    for q_desc in query_descs:
                        desc_dists = [self.euclidean_distance(q_desc, db_desc) for db_desc in db_descs]
                        distances.append(min(desc_dists))
                    
                    mean_dist = sum(distances) / len(distances)
                    descriptor_distances[(world_x, world_y)] = mean_dist
            
        except Exception as e:
            print(f"❌ SuperPoint matching failed: {e}")
        
        return descriptor_distances
    
    def euclidean_distance(self, vec1, vec2):
        """Calculate Euclidean distance between two vectors"""
        return math.sqrt(sum((a - b)**2 for a, b in zip(vec1, vec2)))
    
    def simulate_database_descriptors(self, world_x, world_y, num_descs):
        """Simulate database descriptors for a given location"""
        random.seed(int((world_x + 1000) * 1000 + (world_y + 1000)))
        
        descriptors = []
        for _ in range(min(num_descs, 10)):
            desc = [random.gauss(0, 1) for _ in range(256)]
            norm = math.sqrt(sum(d*d for d in desc))
            if norm > 0:
                desc = [d/norm for d in desc]
            descriptors.append(desc)
        
        random.seed()
        return descriptors
    
    def get_most_likely_position(self):
        """Get the most likely position from probability map"""
        max_idx = np.unravel_index(np.argmax(self.prob_map), self.prob_map.shape)
        max_prob = self.prob_map[max_idx]
        world_x, world_y = self.grid_to_world(max_idx[1], max_idx[0])
        return world_x, world_y, max_prob

class CorrectedLSVLSimulator:
    """Corrected drone simulator with MP4 output"""
    
    def __init__(self):
        self.prob_map = CorrectedLSVLMap()
        self.create_reference_database()
        
        # Simulation tracking
        self.results = {
            'trajectory': [],
            'estimated_positions': [],
            'recall_1': [],
            'recall_5': [],
            'position_errors': [],
            'confidence_scores': [],
            'update_points': []
        }
        
        # MP4 output setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('LSVL GPS-Denied Localization Simulation', fontsize=16)
        
    def create_reference_database(self):
        """Create reference database with feature vectors"""
        print("💾 Creating reference database...")
        
        self.db_points = []
        self.db_features = []
        
        # Grid of reference points every 200m
        for x in range(-800, 801, 200):
            for y in range(-800, 801, 200):
                self.db_points.append((x, y))
                
                # Simulate normalized feature vector
                feature = [random.gauss(0, 1) for _ in range(256)]
                norm = math.sqrt(sum(f*f for f in feature))
                if norm > 0:
                    feature = [f/norm for f in feature]
                self.db_features.append(feature)
        
        print(f"✅ Database created: {len(self.db_points)} reference points")
    
    def euclidean_distance(self, vec1, vec2):
        """Calculate Euclidean distance between two vectors"""
        return math.sqrt(sum((a - b)**2 for a, b in zip(vec1, vec2)))
    
    def simulate_observations(self, drone_pos, num_observations=8):
        """Simulate visual observations with embedding distances"""
        observations = []
        
        distances = []
        for db_x, db_y in self.db_points:
            dist = math.sqrt((drone_pos[0] - db_x)**2 + (drone_pos[1] - db_y)**2)
            distances.append(dist)
        
        sorted_pairs = sorted(enumerate(distances), key=lambda x: x[1])
        closest_indices = [pair[0] for pair in sorted_pairs[:num_observations]]
        
        for idx in closest_indices:
            db_x, db_y = self.db_points[idx]
            db_feature = self.db_features[idx]
            
            # Simulate query feature with noise
            query_feature = []
            for f in db_feature:
                noisy_f = f + random.gauss(0, 0.1)
                query_feature.append(noisy_f)
            
            # Normalize query feature
            norm = math.sqrt(sum(f*f for f in query_feature))
            if norm > 0:
                query_feature = [f/norm for f in query_feature]
            
            # Calculate embedding distance
            embedding_dist = self.euclidean_distance(query_feature, db_feature)
            
            # Add noise based on physical distance
            physical_distance = distances[idx]
            noise_factor = min(0.3, physical_distance / 1000)
            embedding_dist += random.gauss(0, noise_factor)
            embedding_dist = max(0, embedding_dist)
            
            observations.append((db_x, db_y, embedding_dist))
        
        return observations
    
    def generate_trajectory(self, num_steps=60):
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
                
                if distance < 100:
                    current_waypoint += 1
                
                if distance > 0:
                    step_size = 30  # Smaller steps for more realistic movement
                    current_pos[0] += (dx / distance) * step_size + random.uniform(-15, 15)
                    current_pos[1] += (dy / distance) * step_size + random.uniform(-15, 15)
                else:
                    current_pos[0] += random.uniform(-20, 20)
                    current_pos[1] += random.uniform(-20, 20)
            else:
                current_pos[0] += random.uniform(-30, 30)
                current_pos[1] += random.uniform(-30, 30)
            
            # Keep within bounds
            current_pos[0] = max(-900, min(900, current_pos[0]))
            current_pos[1] = max(-900, min(900, current_pos[1]))
            
            trajectory.append(tuple(current_pos))
        
        return trajectory
    
    def calculate_recall_metrics(self, estimated_pos, true_pos, tolerance=100):
        """Calculate recall metrics"""
        error = math.sqrt((estimated_pos[0] - true_pos[0])**2 + (estimated_pos[1] - true_pos[1])**2)
        
        recall_1 = 1.0 if error <= tolerance else 0.0
        recall_5 = 1.0 if error <= tolerance * 2 else 0.0
        
        return recall_1, recall_5, error
    
    def update_visualization(self, step, trajectory):
        """Update visualization for current step"""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        true_pos = trajectory[step]
        
        # Get current estimated position
        est_x, est_y, max_prob = self.prob_map.get_most_likely_position()
        
        # Calculate error
        error = math.sqrt((est_x - true_pos[0])**2 + (est_y - true_pos[1])**2)
        
        # Store results
        self.results['trajectory'].append(true_pos)
        self.results['estimated_positions'].append((est_x, est_y))
        
        recall_1, recall_5, _ = self.calculate_recall_metrics((est_x, est_y), true_pos)
        self.results['recall_1'].append(recall_1)
        self.results['recall_5'].append(recall_5)
        self.results['position_errors'].append(error)
        self.results['confidence_scores'].append(max_prob)
        
        # Plot 1: Probability Map
        ax1 = self.axes[0, 0]
        im1 = ax1.imshow(self.prob_map.prob_map, extent=[self.prob_map.min_x, self.prob_map.max_x, 
                                                        self.prob_map.min_y, self.prob_map.max_y], 
                        origin='lower', cmap='hot', interpolation='bilinear')
        ax1.plot(true_pos[0], true_pos[1], 'bo', markersize=8, label='True Position')
        ax1.plot(est_x, est_y, 'r*', markersize=12, label='Estimated Position')
        
        # Add update points
        for update_info in self.prob_map.update_history:
            pos = update_info['position']
            ax1.plot(pos[0], pos[1], 'go', markersize=4, alpha=0.7)
        
        ax1.set_title(f'Probability Map (Step {step+1})')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trajectory
        ax2 = self.axes[0, 1]
        if len(self.results['trajectory']) > 1:
            traj_x = [pos[0] for pos in self.results['trajectory']]
            traj_y = [pos[1] for pos in self.results['trajectory']]
            est_x_list = [pos[0] for pos in self.results['estimated_positions']]
            est_y_list = [pos[1] for pos in self.results['estimated_positions']]
            
            ax2.plot(traj_x, traj_y, 'b-', linewidth=2, label='True Trajectory')
            ax2.plot(est_x_list, est_y_list, 'r--', linewidth=2, label='Estimated Trajectory')
            ax2.plot(traj_x[-1], traj_y[-1], 'bo', markersize=8)
            ax2.plot(est_x_list[-1], est_y_list[-1], 'r*', markersize=12)
        
        # Add database points
        db_x = [pos[0] for pos in self.db_points]
        db_y = [pos[1] for pos in self.db_points]
        ax2.plot(db_x, db_y, 'k.', markersize=3, alpha=0.5, label='Database Points')
        
        ax2.set_title('Trajectory Comparison')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Plot 3: Performance Metrics
        ax3 = self.axes[1, 0]
        if len(self.results['position_errors']) > 0:
            steps = range(1, len(self.results['position_errors']) + 1)
            ax3.plot(steps, self.results['position_errors'], 'g-', linewidth=2, label='Position Error')
            ax3.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='100m Threshold')
            ax3.axhline(y=200, color='orange', linestyle='--', alpha=0.7, label='200m Threshold')
        
        ax3.set_title('Position Error Over Time')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Error (m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Confidence and Recall
        ax4 = self.axes[1, 1]
        if len(self.results['confidence_scores']) > 0:
            steps = range(1, len(self.results['confidence_scores']) + 1)
            ax4.plot(steps, self.results['confidence_scores'], 'purple', linewidth=2, label='Confidence')
            
            # Add recall indicators
            recall_1_y = [r * 0.8 for r in self.results['recall_1']]  # Scale for visibility
            recall_5_y = [r * 0.6 for r in self.results['recall_5']]
            ax4.plot(steps, recall_1_y, 'ro', markersize=4, label='Recall@1')
            ax4.plot(steps, recall_5_y, 'go', markersize=4, label='Recall@5')
        
        ax4.set_title('Confidence and Recall')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add overall statistics
        if len(self.results['position_errors']) > 0:
            avg_error = sum(self.results['position_errors']) / len(self.results['position_errors'])
            avg_recall_1 = sum(self.results['recall_1']) / len(self.results['recall_1'])
            avg_recall_5 = sum(self.results['recall_5']) / len(self.results['recall_5'])
            
            stats_text = f'Avg Error: {avg_error:.1f}m\nRecall@1: {avg_recall_1:.1%}\nRecall@5: {avg_recall_5:.1%}'
            self.fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
    
    def run_simulation(self, num_steps=60):
        """Run corrected LSVL simulation with MP4 output"""
        print("🚁 Starting Corrected LSVL Simulation with MP4 Output")
        print("=" * 60)
        
        # Generate trajectory
        trajectory = self.generate_trajectory(num_steps)
        print(f"✅ Generated trajectory with {len(trajectory)} waypoints")
        
        # Prepare for animation
        frames_data = []
        
        # Run simulation
        for step in range(len(trajectory)):
            true_pos = trajectory[step]
            
            print(f"\n🚁 Step {step + 1}/{len(trajectory)} - Position: ({true_pos[0]:.1f}, {true_pos[1]:.1f})m")
            
            # Simulate observations
            observations = self.simulate_observations(true_pos)
            
            # Generate query image
            query_image = f"simulated_image_{step}"
            
            # Update probability map (only if moved 50m)
            updated = self.prob_map.update_probabilities_progressive(observations, query_image, true_pos)
            
            if updated:
                self.results['update_points'].append(step)
                print(f"✅ Map updated at step {step + 1}")
            else:
                print(f"⏭️ Skipped update (distance < 50m)")
            
            # Update visualization
            self.update_visualization(step, trajectory)
            
            # Store frame data
            frames_data.append(step)
        
        # Create MP4 animation
        print("\n🎬 Creating MP4 animation...")
        
        def animate(frame):
            self.update_visualization(frame, trajectory)
            return []
        
        # Create animation
        anim = animation.FuncAnimation(self.fig, animate, frames=len(trajectory), 
                                     interval=500, blit=False, repeat=True)
        
        # Save as MP4
        mp4_path = "lsvl_corrected_simulation.mp4"
        try:
            anim.save(mp4_path, writer='ffmpeg', fps=2, bitrate=1800)
            print(f"✅ MP4 saved: {mp4_path}")
        except Exception as e:
            print(f"❌ MP4 save failed: {e}")
            # Try alternative writer
            try:
                anim.save(mp4_path, writer='pillow', fps=2)
                print(f"✅ MP4 saved with pillow: {mp4_path}")
            except Exception as e2:
                print(f"❌ Alternative MP4 save failed: {e2}")
        
        # Generate final report
        self.generate_final_report()
        
        return mp4_path
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n{'='*60}")
        print("📊 CORRECTED LSVL SIMULATION RESULTS")
        print(f"{'='*60}")
        
        if self.results['recall_1']:
            avg_recall_1 = sum(self.results['recall_1']) / len(self.results['recall_1'])
            avg_recall_5 = sum(self.results['recall_5']) / len(self.results['recall_5'])
            avg_error = sum(self.results['position_errors']) / len(self.results['position_errors'])
            avg_confidence = sum(self.results['confidence_scores']) / len(self.results['confidence_scores'])
            
            print(f"\n🎯 Performance Metrics:")
            print(f"   • Average Recall@1 (100m): {avg_recall_1:.1%}")
            print(f"   • Average Recall@5 (200m): {avg_recall_5:.1%}")
            print(f"   • Average Position Error: {avg_error:.1f}m")
            print(f"   • Average Confidence: {avg_confidence:.4f}")
            
            print(f"\n📍 Update Statistics:")
            print(f"   • Total Steps: {len(self.results['trajectory'])}")
            print(f"   • Updates Performed: {len(self.results['update_points'])}")
            print(f"   • Update Rate: {len(self.results['update_points'])/len(self.results['trajectory']):.1%}")
            
            # Final normalization check
            final_sum = np.sum(self.prob_map.prob_map)
            print(f"\n🔍 Final Validation:")
            print(f"   • Final probability sum: {final_sum:.10f}")
            print(f"   • Normalization: {'✅ PASSED' if abs(final_sum - 1.0) < 1e-8 else '❌ FAILED'}")
        
        # Save detailed report
        report_data = {
            'simulation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'Corrected LSVL + Progressive SuperPoint',
            'corrections': [
                'Proper normalization after every update',
                'Update only when drone moves 50m',
                'MP4 video output',
                'Comprehensive error checking'
            ],
            'results': self.results,
            'final_metrics': {
                'avg_recall_1': avg_recall_1 if self.results['recall_1'] else 0,
                'avg_recall_5': avg_recall_5 if self.results['recall_1'] else 0,
                'avg_position_error': avg_error if self.results['recall_1'] else 0,
                'avg_confidence': avg_confidence if self.results['recall_1'] else 0,
                'total_steps': len(self.results['trajectory']),
                'updates_performed': len(self.results['update_points']),
                'final_normalization': float(np.sum(self.prob_map.prob_map))
            }
        }
        
        with open("lsvl_corrected_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved: lsvl_corrected_report.json")

def main():
    """Main simulation function"""
    print("🚁 Corrected LSVL GPS-Denied Localization Simulation")
    print("🔧 Fixed: Normalization + 50m Update Threshold + MP4 Output")
    print("=" * 60)
    
    # Create and run corrected simulator
    simulator = CorrectedLSVLSimulator()
    mp4_path = simulator.run_simulation(num_steps=60)
    
    print(f"\n✅ Corrected LSVL Simulation Complete!")
    print(f"📁 MP4 Output: {mp4_path}")
    print(f"📁 Report: lsvl_corrected_report.json")
    
    # Get absolute path
    abs_path = Path(mp4_path).resolve()
    print(f"📍 Absolute MP4 Path: {abs_path}")

if __name__ == "__main__":
    main()