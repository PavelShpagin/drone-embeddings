#!/usr/bin/env python3
"""
Global GPS-Denied Geolocalization Simulation

This script implements the probabilistic localization algorithm based on the paper reference,
using:
- 100m x 100m satellite image patches with pre-computed embeddings
- Confidence circle with probability distribution
- VIO measurement prediction with convolution
- Visual-inertial corrections when confidence is high

The algorithm maintains a confidence circle around the estimated drone position,
updates probabilities using VIO measurements and camera observations, and
applies corrections when localization confidence is high and radius is large.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from geolocalization.database import EmbeddingDatabase
from geolocalization.drone import Drone
from geolocalization.state import LocalizationState
from geolocalization.visualizer import SimulationVisualizer
from geolocalization import config

class GeolocalizationSimulation:
    """
    Main simulation class that orchestrates the probabilistic localization algorithm.
    """
    def __init__(self):
        self.step_count = 0
        self.correction_count = 0
        self.total_distance_traveled = 0.0
        
        print("=== Initializing Global GPS-Denied Geolocalization Simulation ===")
        
        # Initialize components
        self.database = self._initialize_database()
        self.drone = self._initialize_drone()
        self.localization_state = self._initialize_localization_state()
        self.visualizer = self._initialize_visualizer()
        
        print(f"Simulation initialized successfully!")
        print(f"Map: {config.MAP_IMAGE_PATH}")
        print(f"Patches: {self.database.grid_w}x{self.database.grid_h} ({self.database.grid_w * self.database.grid_h} total)")
        print(f"Initial drone position: {self.drone.get_true_position_world()}")
        print(f"Initial confidence radius: {self.localization_state.radius_m}m")

    def _initialize_database(self) -> EmbeddingDatabase:
        """Initialize the embedding database with pre-computed patch embeddings."""
        print("Building embedding database...")
        start_time = time.time()
        
        database = EmbeddingDatabase()
        database.build_database()
        
        elapsed_time = time.time() - start_time
        print(f"Database built in {elapsed_time:.1f} seconds")
        
        return database

    def _initialize_drone(self) -> Drone:
        """Initialize the drone with dynamic trajectory generation."""
        print("Initializing drone...")
        
        # Start the drone at a random position within the map
        drone = Drone(self.database)
        
        print(f"Drone initialized at world position: {drone.get_true_position_world()}")
        return drone

    def _initialize_localization_state(self) -> LocalizationState:
        """Initialize the probabilistic localization state."""
        print("Initializing localization state...")
        
        # Initialize with the drone's true starting position (in practice, this would be GPS or user input)
        initial_pos = self.drone.get_true_position_world()
        state = LocalizationState(self.database, initial_pos)
        
        return state

    def _initialize_visualizer(self) -> SimulationVisualizer:
        """Initialize the visualization system."""
        print("Initializing visualizer...")
        
        output_dir = Path("simulation_results")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "geolocalization_simulation.mp4"
        visualizer = SimulationVisualizer(self.database, str(output_path))
        
        return visualizer

    def run_simulation(self, max_steps: int = None):
        """
        Run the main simulation loop implementing the probabilistic localization algorithm.
        """
        max_steps = max_steps or config.NUM_STEPS
        print(f"\n=== Starting Simulation ({max_steps} steps) ===")
        
        try:
            for step in range(max_steps):
                self.step_count = step
                
                # Execute one simulation step
                correction_triggered = self._simulation_step()
                
                # Visualization
                self.visualizer.draw_frame(
                    self.localization_state, 
                    self.drone, 
                    step + 1, 
                    correction_triggered
                )
                
                # Progress reporting
                if (step + 1) % 100 == 0:
                    self._print_progress_report(step + 1)
                
                # Check termination conditions
                if not self.drone.is_within_bounds():
                    print(f"Drone reached map boundary at step {step + 1}")
                    break
                    
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nSimulation error: {e}")
            raise
        finally:
            # Cleanup and final reports
            self._finalize_simulation()

    def _simulation_step(self) -> bool:
        """
        Execute one step of the simulation algorithm.
        Returns True if a correction was triggered.
        """
        # 1. Drone movement with VIO measurements
        vio_delta_m, epsilon_m, should_update = self.drone.move_step()
        
        if vio_delta_m is None:
            return False  # End of trajectory
        
        self.total_distance_traveled += np.linalg.norm(vio_delta_m)
        
        # 2. VIO prediction step (from paper equations 9a-9c)
        self.localization_state.update_motion_prediction(vio_delta_m, epsilon_m)
        
        correction_triggered = False
        
        # 3. Image measurement update (every UPDATE_INTERVAL_M meters)
        if should_update:
            # Get current camera view
            camera_view = self.drone.get_camera_view(view_size_m=config.GRID_PATCH_SIZE_M)
            
            # Update probabilities based on visual measurement
            self.localization_state.update_measurement(camera_view)
            
            # 4. Check for correction trigger conditions
            correction_triggered = self._check_and_apply_correction()
        
        return correction_triggered

    def _check_and_apply_correction(self) -> bool:
        """
        Check if conditions are met for position correction and apply if needed.
        Returns True if correction was applied.
        """
        # Check if radius is large enough to warrant correction
        if self.localization_state.radius_m < config.CORRECTION_THRESHOLD_M:
            return False
        
        # Check if we have high confidence localization
        metrics = self.localization_state.get_confidence_metrics()
        confidence_threshold = 5.0  # Peak-to-average ratio threshold
        
        if metrics['peak_to_avg'] < confidence_threshold:
            return False
        
        # Get the most confident position estimate
        confident_position = self.localization_state.get_most_confident_position()
        
        # Apply correction to both localization state and drone
        self.localization_state.apply_correction(confident_position)
        self.drone.apply_correction(confident_position)
        
        self.correction_count += 1
        
        print(f"CORRECTION #{self.correction_count} applied at step {self.step_count + 1}")
        print(f"  Position estimate: {confident_position}")
        print(f"  Confidence (P/A ratio): {metrics['peak_to_avg']:.2f}")
        print(f"  Radius reduced from {self.localization_state.radius_m * 2:.1f}m to {self.localization_state.radius_m:.1f}m")
        
        return True

    def _print_progress_report(self, step: int):
        """Print a progress report with key metrics."""
        metrics = self.localization_state.get_confidence_metrics()
        position_error = self.drone.get_position_error_m()
        
        print(f"\n--- Step {step} Progress Report ---")
        print(f"Distance traveled: {self.total_distance_traveled:.1f}m")
        print(f"Position error: {position_error:.1f}m")
        print(f"Confidence radius: {metrics['radius_m']:.1f}m")
        print(f"Active patches: {metrics['num_patches']}")
        print(f"Peak/Avg ratio: {metrics['peak_to_avg']:.2f}")
        print(f"Corrections applied: {self.correction_count}")

    def _finalize_simulation(self):
        """Finalize the simulation and generate reports."""
        print(f"\n=== Simulation Complete ===")
        print(f"Total steps: {self.step_count + 1}")
        print(f"Total distance: {self.total_distance_traveled:.1f}m")
        print(f"Total corrections: {self.correction_count}")
        
        final_error = self.drone.get_position_error_m()
        final_metrics = self.localization_state.get_confidence_metrics()
        
        print(f"Final position error: {final_error:.1f}m")
        print(f"Final confidence radius: {final_metrics['radius_m']:.1f}m")
        
        # Calculate average correction interval
        if self.correction_count > 0:
            avg_correction_interval = self.total_distance_traveled / self.correction_count
            print(f"Average correction interval: {avg_correction_interval:.1f}m")
        
        # Close visualizer (this will generate summary plots)
        self.visualizer.close()
        
        print("\nResults saved to simulation_results/")

def main():
    print("=== GPS-Denied Geolocalization Simulation ===")
    print("Based on paper: Probabilistic localization using 2D satellite image patches")
    print(f"Target duration: 2 minutes with {config.NUM_STEPS} steps")
    
    # Initialize components
    print("\n1. Building database...")
    database = Database()
    
    print("\n2. Initializing drone...")
    drone = Drone(database)
    
    print("\n3. Initializing localization state...")
    localization_state = LocalizationState(database)
    
    print("\n4. Setting up visualization...")
    visualizer = SimulationVisualizer(database, "simulation_output.mp4")
    
    print(f"\n5. Starting simulation ({config.NUM_STEPS} steps)...")
    print("=" * 60)
    
    # Simulation loop
    update_count = 0
    correction_count = 0
    
    for step in range(config.NUM_STEPS):
        # Show progress every 100 steps
        if step % 100 == 0:
            progress_pct = (step / config.NUM_STEPS) * 100
            distance_traveled = drone.total_distance_traveled
            position_error = drone.get_position_error_m()
            print(f"Step {step:4d}/{config.NUM_STEPS} ({progress_pct:5.1f}%) | "
                  f"Distance: {distance_traveled:6.0f}m | Error: {position_error:5.1f}m")
        
        # Execute drone movement
        vio_delta_m, epsilon_m, should_update = drone.move_step()
        
        # Update localization state with VIO measurement
        if should_update:
            update_count += 1
            
            # Get camera measurement
            camera_view = drone.get_camera_view()
            
            # Update localization with new measurement
            localization_state.update_with_measurement(vio_delta_m, camera_view, epsilon_m)
            
            # Check if correction is needed
            metrics = localization_state.get_confidence_metrics()
            correction_triggered = False
            
            if (metrics['radius_m'] > config.CORRECTION_THRESHOLD_M and 
                metrics['peak_to_avg'] > 2.0):  # High confidence threshold
                
                # Apply correction
                confident_pos = localization_state.get_most_confident_position()
                drone.apply_correction(confident_pos)
                correction_count += 1
                correction_triggered = True
                
                print(f"  CORRECTION #{correction_count} at step {step} | "
                      f"Radius: {metrics['radius_m']:.0f}m | "
                      f"Confidence: {metrics['peak_to_avg']:.2f}")
            
            # Draw visualization frame
            visualizer.draw_frame(localization_state, drone, step, correction_triggered)
        
        # Check bounds and safety
        if not drone.is_within_bounds():
            print(f"Warning: Drone approaching map boundaries at step {step}")
    
    print("=" * 60)
    print(f"Simulation completed!")
    print(f"Final Statistics:")
    print(f"  Total updates: {update_count}")
    print(f"  Total corrections: {correction_count}")
    print(f"  Distance traveled: {drone.total_distance_traveled:.0f}m")
    print(f"  Final position error: {drone.get_position_error_m():.1f}m")
    
    # Finalize visualization
    print(f"\n6. Generating video output...")
    visualizer.close()
    
    # Save video with proper 2-minute duration
    visualizer.save_video("long_trajectory_simulation.mp4")
    
    print("\nSimulation completed successfully!")
    print("Output files:")
    print("  - long_trajectory_simulation.mp4 (2-minute trajectory video)")
    print("  - simulation_summary.png (trajectory analysis)")

if __name__ == "__main__":
    main() 