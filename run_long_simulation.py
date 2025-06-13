#!/usr/bin/env python3
"""
GPS-Denied Geolocalization Simulation - Long Trajectory Version
Based on probabilistic localization using 2D satellite image patches
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from geolocalization.database import EmbeddingDatabase
from geolocalization.drone import Drone
from geolocalization.state import LocalizationState
from geolocalization.visualizer import SimulationVisualizer
from geolocalization import config

def main():
    print("=== GPS-Denied Geolocalization Simulation - Long Trajectory ===")
    print("Based on paper: Probabilistic localization using 2D satellite image patches")
    print(f"Target: 2-minute video with {config.NUM_STEPS} steps")
    print(f"Trajectory segments: {config.SEGMENT_LENGTH_MIN_M}m - {config.SEGMENT_LENGTH_MAX_M}m")
    
    # Initialize components
    print("\n1. Building database...")
    database = EmbeddingDatabase()
    
    print("\n2. Initializing drone...")
    drone = Drone(database)
    
    print("\n3. Initializing localization state...")
    # Get the drone's starting position for localization state
    initial_pos = drone.get_true_position_world()
    localization_state = LocalizationState(database, initial_pos)
    
    print("\n4. Setting up visualization...")
    visualizer = SimulationVisualizer(database, output_resolution=(1920, 1080), max_fps=20)
    
    print(f"\n5. Starting simulation ({config.NUM_STEPS} steps)...")
    print("=" * 60)
    
    # Simulation loop
    update_count = 0
    correction_count = 0
    
    try:
        for step in range(config.NUM_STEPS):
            # Show progress every 100 steps
            if step % 100 == 0 or step < 10:
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
                
                # First update with VIO motion prediction
                localization_state.update_motion_prediction(vio_delta_m, epsilon_m)
                
                # Get camera measurement
                camera_view = drone.get_camera_view()
                
                # Update localization with visual measurement
                localization_state.update_measurement(camera_view)
                
                # Check if correction is needed
                metrics = localization_state.get_confidence_metrics()
                correction_triggered = False
                
                # Check if we have radius information and if correction is needed
                current_radius = metrics.get('radius_m', localization_state.radius_m)
                if (current_radius > config.CORRECTION_THRESHOLD_M and 
                    metrics.get('peak_to_avg', 0) > 2.0):  # High confidence threshold
                    
                    # Apply correction
                    confident_pos = localization_state.get_most_confident_position()
                    drone.apply_correction(confident_pos)
                    correction_count += 1
                    correction_triggered = True
                    
                    print(f"  CORRECTION #{correction_count} at step {step} | "
                          f"Radius: {current_radius:.0f}m | "
                          f"Confidence: {metrics.get('peak_to_avg', 0):.2f}")
                
                # Draw visualization frame
                visualizer.draw_frame(localization_state, drone, step, correction_triggered)
            
            # Check bounds and safety
            if not drone.is_within_bounds():
                print(f"Warning: Drone approaching map boundaries at step {step}")
    
    except KeyboardInterrupt:
        print(f"\nSimulation interrupted at step {step}")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        raise
    
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