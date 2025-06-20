import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import time

# Add geolocalization directory to path for imports
sys.path.append(str(Path(__file__).parent / "geolocalization"))

from new_config import Config

# Import classes after config is defined, as they rely on it
from new_localization import NewGlobalLocalizer
from new_drone import NewDrone
from new_visualizer import NewSimulationVisualizer

def main():
    parser = argparse.ArgumentParser(description="Run new GPS-denied geolocalization simulation.")
    parser.add_argument('--num_steps', type=int, default=Config.NUM_STEPS, help='Number of simulation steps.')
    parser.add_argument('--output_video', type=str, default="new_simulation_output.avi", help='Output video filename.')
    args = parser.parse_args()

    config = Config()

    # Initialize localizer
    localizer = NewGlobalLocalizer(config)

    # Initialize drone at a random position within map bounds
    # The drone's init handles random start
    drone = NewDrone(config, localizer)
    start_world_pos = drone.get_true_position_world()
    localizer.initialize_circle(start_world_pos)

    # Initialize visualizer
    visualizer = NewSimulationVisualizer(config, localizer, args.output_video)

    print("Starting simulation...")
    # --- Recall statistics ---
    recall1_count = 0
    recall5_count = 0
    recall_total = 0
    
    try:
        num_frames = args.num_steps
        for step in range(num_frames):
            print(f"\n--- Simulation Step {step} ---")

            # Drone takes a step (generates VIO delta and epsilon, checks for update interval)
            should_update_localization = drone.move_step()

            if should_update_localization:
                print(f"Performing measurement update at step {step}")
                
                # Get accumulated VIO data for this update interval
                accumulated_vio_delta, accumulated_epsilon = drone.get_accumulated_vio_data()

                # Update localization based on VIO (prediction step) using accumulated data
                localizer.update_motion_prediction(accumulated_vio_delta, accumulated_epsilon)

                # Update circle center based on accumulated VIO data
                localizer.update_circle_center(accumulated_vio_delta)

                # Get camera view from true position
                camera_view_image = drone.get_camera_view(view_size_m=localizer.patch_size_m) # Camera view is 100x100m

                # Update probabilities with camera measurement and SuperPoint
                localizer.update_measurement(camera_view_image)

                # --- Recall@1 and Recall@5 calculation ---
                # Get the true position in grid coordinates
                true_pos = drone.get_true_position_world()
                true_grid = localizer._world_to_grid(true_pos[0], true_pos[1])

                # Get top-5 patch predictions by probability
                patch_probs = localizer.patch_probabilities
                if patch_probs:
                    top_patches = sorted(patch_probs.items(), key=lambda x: x[1], reverse=True)
                    top1 = top_patches[0][0]
                    top5 = [x[0] for x in top_patches[:5]]

                    recall_total += 1
                    if true_grid == top1:
                        recall1_count += 1
                    if true_grid in top5:
                        recall5_count += 1

                    print(f"Recall@1: {recall1_count}/{recall_total} ({recall1_count/recall_total:.3f}) | Recall@5: {recall5_count}/{recall_total} ({recall5_count/recall_total:.3f})")

                # Reset accumulated VIO data in drone after localization update
                drone.reset_accumulated_vio_data()

                # Check if correction is needed
                should_correct, correction_target = localizer.check_correction_trigger()

                if should_correct:
                    print(f"Localizer triggered correction to {correction_target}")
                    drone.apply_correction(correction_target) # Drone starts moving towards target
                    # Localizer also shrinks radius upon correction application in its own apply_correction
                    localizer.apply_correction(correction_target) # Shrink radius for localizer after correction
                else:
                    print("No correction triggered.")
            else:
                print("Skipping localization update (not enough distance traveled or correction active).")

            # Draw current frame for visualization
            visualizer.draw_frame(drone, step)

            # Optional: Add a small delay for real-time viewing
            # time.sleep(0.01)

            if not drone.is_within_bounds():
                print("Drone moved out of safe bounds. Ending simulation.")
                break

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        visualizer.close()
        print("Simulation finished.")
        # --- Print final recall statistics ---
        if recall_total > 0:
            print(f"\nFinal Recall@1: {recall1_count}/{recall_total} = {recall1_count/recall_total:.3f}")
            print(f"Final Recall@5: {recall5_count}/{recall_total} = {recall5_count/recall_total:.3f}")
        else:
            print("No recall statistics to report.")

if __name__ == "__main__":
    main() 