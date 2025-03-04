import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.simulation.drone import DroneFlight
from src.google_maps import get_static_map, calculate_google_zoom
from src.azure_maps import get_azure_maps_image, load_azure_maps_key, calculate_azure_zoom
from src.api.elevation import get_elevation
from src.image_processing import crop_and_resize_azure_image, crop_and_resize_google_image
import time

def simulate_drone_flight(
    start_lat: float,
    start_lng: float,
    start_altitude: float,
    velocity: np.ndarray,
    duration: int = 100,
    sample_interval: int = 10
):
    # Initialize drone flight
    drone = DroneFlight(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=start_altitude,
        velocity=velocity
    )

    output_path = Path("output")
    if output_path.exists():
        for file in output_path.glob("*"):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                for subfile in file.glob("*"):
                    subfile.unlink()
                file.rmdir()
        output_path.rmdir()
    
    # Create output directories
    Path("output/azure_maps_images").mkdir(parents=True, exist_ok=True)
    Path("output/google_maps_images").mkdir(parents=True, exist_ok=True)
    Path("flight_data").mkdir(parents=True, exist_ok=True)
    
    # Simulate flight
    positions = []
    
    for t in range(duration):
        state = drone.step()
        
        if t % sample_interval == 0:
            print(f"Time: {t}s, Position: {state.position}, Lat/Lng: {state.lat}, {state.lng}")
            
            # Get Azure Maps image with Azure-specific zoom
            azure_zoom = calculate_azure_zoom(state.altitude, state.lat)
            azure_image = get_azure_maps_image(
                latitude=state.lat,
                longitude=state.lng,
                zoom=azure_zoom,
                size=256,
                scale=2
            )
            
            if azure_image:
                azure_image_path = f"output/azure_maps_images/frame_{t:03d}.png"
                azure_image.save(azure_image_path)
            
            # Get Google Maps image with Google-specific zoom
            google_zoom = calculate_google_zoom(state.altitude, state.lat)
            google_image = get_static_map(
                state.lat,
                state.lng,
                zoom=google_zoom,
                scale=2,
                size="256x256"
            )
            
            if google_image:
                google_image_path = f"output/google_maps_images/frame_{t:03d}.png"
                google_image.save(google_image_path)
            
            positions.append(state.position)
            
            # Add a small delay between requests to respect rate limits
            time.sleep(1)
    
    # Plot flight path
    plot_flight_path(positions)

def plot_flight_path(positions):
    positions = np.array(positions)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot flight path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Flight Path')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    
    # Add labels and title
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Drone Flight Path')
    
    # Save the plot to a file
    Path("flight_data").mkdir(parents=True, exist_ok=True)
    plt.savefig('flight_data/flight_path.png')
    plt.close()

if __name__ == "__main__":
    # Example flight parameters - Using Odesa coordinates
    start_lat = 46.4825  # Odesa City
    start_lng = 30.7233
    start_altitude = 100  # meters
    velocity = np.array([0.5, 0.5, 0.1])  # meters/second
    
    simulate_drone_flight(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=start_altitude,
        velocity=velocity,
        duration=50,  # Reduced duration for testing
        sample_interval=5  # More frequent sampling
    ) 