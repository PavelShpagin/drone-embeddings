#!/usr/bin/env python3
"""
New Global GPS-Denied Geolocalization Simulation

Implements the updated algorithm with:
- Dynamic patch-based database (only for confidence circle)
- Circle-based probability tracking with VIO prediction
- Correction triggers when radius exceeds threshold
- Two trajectories: desired (ground truth) and current (noisy VIO)
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from geolocalization.new_localization import NewGlobalLocalizer
from geolocalization.new_drone import NewDrone
from geolocalization.new_visualizer import NewSimulationVisualizer

class Config:
    """Configuration for the new global localization algorithm."""
    
    # Map and Model
    MAP_IMAGE_PATH = "inference/46.6234, 32.7851.jpg"
    MAP_CENTER_LAT = 46.6234
    MAP_CENTER_LNG = 32.7851
    M_PER_PIXEL = 0.487
    MODEL_WEIGHTS_PATH = "training_results/efficientnet_b0/checkpoints/checkpoint_epoch_11.pth"
    BACKBONE_NAME = 'efficientnet_b0'
    
    # Patch and Grid
    GRID_PATCH_SIZE_M = 100.0
    CROP_SIZE_PX = 224
    
    # Confidence Circle
    INITIAL_RADIUS_M = 150.0
    MAX_RADIUS_M = 400.0
    CORRECTION_THRESHOLD_M = 300.0
    
    # VIO and Motion
    VIO_ERROR_STD_M = 1.5
    VIO_X_VARIANCE = 1.0
    VIO_Y_VARIANCE = 1.0
    UPDATE_INTERVAL_M = 20.0
    STEP_SIZE_M = 1.0
    
    # Simulation
    NUM_STEPS = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_new_global_simulation():
    """Main function to run the new global localization simulation."""
    
    print("=== New Global GPS-Denied Geolocalization Simulation ===\n")
    
    # Check if required files exist
    config = Config()
    
    if not Path(config.MAP_IMAGE_PATH).exists():
        print(f"ERROR: Map image not found at {config.MAP_IMAGE_PATH}")
        print("Please ensure the satellite image is available.")
        return
    
    if not Path(config.MODEL_WEIGHTS_PATH).exists():
        print(f"ERROR: Model weights not found at {config.MODEL_WEIGHTS_PATH}")
        print("Please run train_encoder.py first to train the model.")
        return
    
    # Create output directory
    output_dir = Path("new_simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    print("--- Initializing Components ---")
    
    # 1. Initialize the Global Localizer
    print("Loading global localizer...")
    localizer = NewGlobalLocalizer(config)
    
    # 2. Initialize the Drone
    print("Initializing drone with dynamic trajectory...")
    # Start at map center for this demo
    start_pos = (0.0, 0.0)  # World coordinates relative to map center
    drone = NewDrone(config, localizer, start_pos)
    
    # 3. Initialize the localizer's confidence circle
    print("Setting up initial confidence circle...")
    localizer.initialize_circle(start_pos)
    
    # 4. Initialize the Visualizer
    video_path = output_dir / "new_global_simulation.avi"
    visualizer = NewSimulationVisualizer(config, localizer, str(video_path))
    
    print(f"--- Starting Simulation ({config.NUM_STEPS} steps) ---\n")
    
    # Simulation loop
    for step in range(1, config.NUM_STEPS + 1):
        print(f"Step {step:4d}/{config.NUM_STEPS}: ", end="")
        
        # 1. Move drone and get VIO measurement
        vio_delta_m, epsilon_m, should_update = drone.move_step()
        
        print(f"VIO: [{vio_delta_m[0]:+6.2f}, {vio_delta_m[1]:+6.2f}]m, ε: {epsilon_m:.2f}m", end="")
        
        # 1.5. Always update circle center to ground truth position
        localizer.update_circle_center(drone.get_true_position_world())
        
        # 2. Update localization if needed
        if should_update:
            # a. Motion prediction step
            localizer.update_motion_prediction(vio_delta_m, epsilon_m)
            
            # b. Get camera image and measurement update
            camera_image = drone.get_camera_view(view_size_m=50.0)
            localizer.update_measurement(camera_image)
            
            # c. Check if correction is needed
            should_correct, correction_target = localizer.check_correction_trigger()
            
            if should_correct:
                print(f" -> CORRECTION to {correction_target}", end="")
                
                # Apply correction to drone
                drone.apply_correction(correction_target)
                
                # Apply correction to localizer
                correction_delta = np.array(correction_target) - localizer.center_world_m
                localizer.apply_correction(correction_delta)
            
            print(f" [R: {localizer.radius_m:.1f}m, P: {len(localizer.patch_probabilities)}]")
        else:
            print("")
        
        # 3. Draw visualization frame
        visualizer.draw_frame(drone, step)
        
        # 4. Check bounds
        if not drone.is_within_bounds():
            print(f"Drone left safe bounds at step {step}. Ending simulation.")
            break
    
    # Cleanup
    visualizer.close()
    
    # Print final statistics
    print("\n--- Simulation Completed ---")
    print(f"Final position error: {drone.get_position_error_m():.2f}m")
    print(f"Total distance traveled: {drone.total_distance_traveled:.1f}m")
    print(f"Final confidence radius: {localizer.radius_m:.1f}m")
    print(f"Active patches in final circle: {len(localizer.patch_probabilities)}")
    print(f"Video saved to: {video_path}")
    
    # Save final state
    state_file = output_dir / "final_state.txt"
    with open(state_file, 'w') as f:
        f.write(f"Final Statistics:\n")
        f.write(f"Position Error: {drone.get_position_error_m():.2f}m\n")
        f.write(f"Distance Traveled: {drone.total_distance_traveled:.1f}m\n")
        f.write(f"Final Radius: {localizer.radius_m:.1f}m\n")
        f.write(f"Active Patches: {len(localizer.patch_probabilities)}\n")
        f.write(f"True Position: {drone.get_true_position_world()}\n")
        f.write(f"VIO Position: {drone.get_vio_position_world()}\n")
        f.write(f"Circle Center: {localizer.center_world_m}\n")
    
    print(f"Final state saved to: {state_file}")

if __name__ == "__main__":
    run_new_global_simulation() 