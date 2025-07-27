#!/usr/bin/env python3
"""
Test script for the improved simulation with:
1. Zig-zag rectangular trajectories
2. Dynamic viewport scaling
3. Fixed embedding likelihood calculations
4. True position sampling for images
"""

import numpy as np
from geolocalization import config
from geolocalization.database import EmbeddingDatabase
from geolocalization.drone import Drone
from geolocalization.state import LocalizationState
from geolocalization.visualizer import SimulationVisualizer

def main():
    print("=== Testing Improved Simulation ===")
    
    # Set test parameters
    test_steps = 200  # Much shorter for quick testing
    original_steps = config.NUM_STEPS
    config.NUM_STEPS = test_steps
    
    try:
        # 1. Load database
        print("1. Loading database...")
        database = EmbeddingDatabase()
        database.build_database()
        
        # 2. Initialize drone with zig-zag trajectory
        print("2. Initializing drone...")
        drone = Drone(database)
        
        # 3. Initialize localization state
        print("3. Initializing localization...")
        start_pos_world = drone.get_true_position_world()
        localization_state = LocalizationState(database, start_pos_world)
        
        # 4. Initialize visualizer
        print("4. Initializing visualizer...")
        visualizer = SimulationVisualizer(database)
        
        # 5. Run short simulation
        print("5. Running test simulation...")
        for step in range(test_steps):
            if step % 20 == 0:
                print(f"Test step {step}/{test_steps}")
            
            # Execute drone movement
            vio_delta_m, epsilon_m, should_update = drone.move_step()
            
            # Update localization if needed
            if should_update:
                # Motion prediction
                localization_state.update_motion_prediction(vio_delta_m, epsilon_m)
                
                # Camera measurement (from TRUE position)
                camera_view = drone.get_camera_view()
                
                # Visual measurement update
                localization_state.update_measurement(camera_view)
                
                # Check metrics
                metrics = localization_state.get_confidence_metrics()
                print(f"  Step {step}: patches={metrics['num_patches']}, "
                      f"ratio={metrics['peak_to_avg']:.2f}, radius={metrics['radius_m']:.1f}m")
            
            # Draw frame
            visualizer.draw_frame(localization_state, drone, step)
        
        # 6. Generate test video
        print("6. Saving test video...")
        visualizer.save_video("test_improved_simulation.mp4")
        
        print("=== Test completed successfully! ===")
        print("Check test_improved_simulation.mp4 for results")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original config
        config.NUM_STEPS = original_steps

if __name__ == "__main__":
    main() 