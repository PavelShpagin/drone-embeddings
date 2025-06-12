import os
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from train_encoder import SiameseNet
from geolocalization.database import EmbeddingDatabase
from geolocalization.drone import Drone
from geolocalization.state import LocalizationState
from geolocalization.visualizer import SimulationVisualizer
from geolocalization import config

def run_simulation():
    """
    Main function to set up and run the localization simulation.
    """
    print("--- Initializing Simulation ---")
    
    # Load the high-resolution map
    try:
        map_image = Image.open(config.MAP_IMAGE_PATH).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Map image not found at {config.MAP_IMAGE_PATH}")
        return

    map_w_px, map_h_px = map_image.size
    # This is a placeholder, as the actual m/pixel might be stored in metadata
    # For now, we assume a known value.
    m_per_pixel = 0.487 
    patch_size_px = int(config.GRID_PATCH_SIZE_M / m_per_pixel)

    # 1. Initialize the Drone
    drone = Drone(map_w_px, map_h_px, m_per_pixel)

    # 2. Initialize Encoder and Embedding Database
    print("Loading encoder model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "training_results/efficientnet_b0/stage_1/efficientnet_b0_stage1_best_recall.pth"
    if not os.path.exists(model_path):
        print(f"FATAL: Model weights not found at {model_path}")
        print("Please run train_encoder.py first.")
        return
        
    model = SiameseNet(backbone_name='efficientnet_b0')
    # The loaded state dict contains the full model, not just the encoder
    state_dict = torch.load(model_path, map_location=device)
    # Adjust key names if they were saved with a 'module.' prefix (from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    db = EmbeddingDatabase(model, device, map_image, patch_size_px, m_per_pixel)
    print("Building embedding database from map...")
    db.build_database()
    print(f"Database built with {len(db.embeddings)} embeddings.")

    # 3. Initialize Localization State
    initial_pos_px = drone.get_true_position_px()
    state = LocalizationState(map_w_px, map_h_px, initial_pos_px, m_per_pixel)
    
    # 4. Initialize Visualizer
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "simulation_run.avi")
    visualizer = SimulationVisualizer(map_image, m_per_pixel, patch_size_px, video_path, config)

    print("\n--- Starting Simulation Loop ---\n")

    # 5. Run Simulation Loop
    for step in range(1, config.NUM_STEPS + 1):
        print(f"--- Step {step}/{config.NUM_STEPS} ---")

        # a. Move drone and get VIO update
        vio_delta_m, epsilon_m = drone.move()
        if vio_delta_m is None:
            break
        print(f"Drone moved by [{vio_delta_m[0]:.8f} {vio_delta_m[1]:.8f}] (VIO). Error magnitude (epsilon): {epsilon_m:.2f}m")

        # b. Update state with motion model (prediction step)
        state.update_motion(vio_delta_m, epsilon_m)

        # c. Get current image from drone's true location
        true_pos_px = drone.get_true_position_px()
        
        # Ensure the crop box is within the image boundaries
        left = max(0, int(true_pos_px[0] - patch_size_px // 2))
        top = max(0, int(true_pos_px[1] - patch_size_px // 2))
        right = min(map_w_px, left + patch_size_px)
        bottom = min(map_h_px, top + patch_size_px)
        
        # The crop needs to be of the correct size for the model
        drone_image_crop = map_image.crop((left, top, right, bottom))
        if drone_image_crop.size != (patch_size_px, patch_size_px):
             drone_image_crop = drone_image_crop.resize((patch_size_px, patch_size_px), Image.Resampling.LANCZOS)


        # d. Update state with measurement model (correction step)
        state.update_measurement(drone_image_crop, db)

        # e. Draw visualization frame
        visualizer.draw_frame(state, drone, step)
    
    visualizer.close()
    print("\n--- Simulation Finished ---")
    print(f"Output video saved to: {video_path}")


if __name__ == "__main__":
    run_simulation() 