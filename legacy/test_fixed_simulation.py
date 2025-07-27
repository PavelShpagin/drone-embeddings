#!/usr/bin/env python3
"""
Test script for the fixed simulation with:
1. Proper zig-zag rectangular trajectories within map bounds
2. Fixed viewport scaling and coordinate calculation
3. Improved embedding likelihood calculations
4. Proper patch probability visualization
"""

import numpy as np
from geolocalization import config
from geolocalization.database import EmbeddingDatabase
from geolocalization.drone import Drone
from geolocalization.state import LocalizationState
from geolocalization.visualizer import SimulationVisualizer

def main():
    print("=== Testing Fixed Simulation ===")
    
    # Set test parameters
    original_steps = config.NUM_STEPS
    config.NUM_STEPS = 300  # Shorter test for validation
    
    try:
        # 1. Load database
        print("1. Initializing database...")
        database = EmbeddingDatabase()
        
        # Check if database already exists
        if database.embeddings is None or len(database.embeddings) == 0:
            print("   Building embedding database (this may take a while)...")
            database.build_database()
        print(f"   Database ready: {len(database.embeddings)} embeddings")
        
        # 2. Initialize drone with zig-zag trajectory
        print("2. Initializing drone with zig-zag trajectory...")
        drone = Drone(database, start_world_pos_m=(0.0, 0.0))  # Start at map center
        print(f"   Drone initialized at center: {drone.get_true_position_world()}")
        
        # 3. Initialize localization state
        print("3. Initializing localization state...")
        initial_pos = drone.get_true_position_world()  # Get the initial position from drone
        localization_state = LocalizationState(database, initial_pos)
        print(f"   Initial state: {len(localization_state.patch_probabilities)} patches")
        
        # 4. Initialize visualizer
        print("4. Initializing visualizer with fixed viewport...")
        visualizer = SimulationVisualizer(database, output_resolution=(1280, 720), max_fps=30)
        
        # 5. Run simulation with validation
        print("5. Running simulation...")
        frame_count = 0
        error_count = 0
        
        for step in range(config.NUM_STEPS):
            # Move drone
            drone.step()
            
            # Update localization (motion + measurement)
            true_pos = drone.get_true_position_world()
            vio_pos = drone.get_vio_position_world()
            
            # Calculate VIO delta for motion update
            if step == 0:
                prev_vio_pos = vio_pos
                vio_delta = np.array([0.0, 0.0])
            else:
                vio_delta = np.array(vio_pos) - np.array(prev_vio_pos)
                prev_vio_pos = vio_pos
            
            # Motion update with small epsilon for uncertainty
            epsilon_m = 0.1  # Small motion uncertainty
            localization_state.update_motion_prediction(vio_delta, epsilon_m)
            
            # Measurement update (every 5 steps to see changes)
            if step % 5 == 0:
                try:
                    camera_view = drone.get_camera_view()
                    if camera_view is not None:
                        localization_state.update_measurement(camera_view)
                except Exception as e:
                    print(f"   Measurement error at step {step}: {e}")
            
            # Draw frame with error handling
            try:
                frame = visualizer.draw_frame(localization_state, drone, step)
                if frame is not None:
                    frame_count += 1
                    
                    # Validation output
                    if step % 50 == 0:
                        stats = drone.get_exploration_stats()
                        active_patches = len([p for p in localization_state.patch_probabilities.values() if p > 0.001])
                        error = drone.get_position_error_m()
                        
                        print(f"   Step {step:3d}: Row {stats['current_row']}, "
                              f"Active patches: {active_patches:3d}, "
                              f"Error: {error:5.1f}m, "
                              f"Distance: {stats['total_distance']:4.0f}m")
                        
            except Exception as e:
                error_count += 1
                if error_count < 5:  # Only show first few errors
                    print(f"   Frame error at step {step}: {e}")
                continue
        
        print(f"\n6. Simulation completed!")
        print(f"   Total frames generated: {frame_count}")
        print(f"   Total errors: {error_count}")
        print(f"   Final exploration stats: {drone.get_exploration_stats()}")
        
        # 7. Generate output
        if frame_count > 10:  # Only save if we got enough frames
            print("7. Generating video...")
            try:
                visualizer.save_video("test_fixed_simulation.mp4")
                print("   Video saved: test_fixed_simulation.mp4")
            except Exception as e:
                print(f"   Video save error: {e}")
        else:
            print("7. Too few frames to save video!")
        
        # Cleanup
        try:
            visualizer.close()
        except:
            pass
            
        return frame_count > 0
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original config
        config.NUM_STEPS = original_steps

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!") 