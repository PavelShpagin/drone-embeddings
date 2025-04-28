import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyproj
from src.simulation.drone import DroneFlight
from src.google_maps import get_static_map, calculate_google_zoom
from src.azure_maps import get_azure_maps_image, calculate_azure_zoom
from src.elevation import get_google_elevation
from src.image_processing import crop_and_resize_azure_image, crop_and_resize_google_image
import time

def enu_to_ecef_velocity(lat_deg, lon_deg, velocity_enu) -> np.ndarray:
    """Transforms a velocity vector from local ENU to ECEF frame."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

    # Rotation matrix from ENU to ECEF
    R = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [       0,          cos_lat,          sin_lat]
    ])

    velocity_ecef = R @ np.asarray(velocity_enu)
    return velocity_ecef


def simulate_drone_flight(
    start_lat: float,
    start_lng: float,
    start_height_above_ground: float,
    velocity_enu: np.ndarray, # Renamed to clarify it's ENU
    duration: int = 100,
    sample_interval: int = 10
):
    # --- Get Ground Elevation ---
    print(f"Fetching ground elevation for start coordinates ({start_lat}, {start_lng})...")
    ground_elevation = get_google_elevation(start_lat, start_lng)

    if ground_elevation is None:
        print("Error: Could not fetch ground elevation. Aborting simulation.")
        # Optionally, you could fall back to a default altitude or raise an exception
        # For now, we'll just exit.
        return

    print(f"Ground elevation received: {ground_elevation:.2f} meters")
    # NOTE: DroneFlight now uses ellipsoidal height. get_google_elevation typically
    # returns orthometric height (MSL). For precise work, a geoid model would be needed
    # to convert MSL to ellipsoidal height. Here, we'll assume they are close enough
    # or treat the input as relative to the ellipsoid for simplicity.
    actual_start_altitude = ground_elevation + start_height_above_ground
    print(f"Calculated starting altitude (Ellipsoidal assumed ~MSL): {actual_start_altitude:.2f} meters")
    # ---------------------------

    # --- Convert Velocity ENU to ECEF ---
    print(f"Initial ENU Velocity: {velocity_enu} m/step")
    velocity_ecef = enu_to_ecef_velocity(start_lat, start_lng, velocity_enu)
    print(f"Converted ECEF Velocity: {velocity_ecef} m/step")
    # ------------------------------------

    # Initialize drone flight with the ECEF velocity
    drone = DroneFlight(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=actual_start_altitude, # Pass the calculated ellipsoidal height
        velocity_ecef=velocity_ecef, # Use the converted ECEF velocity
        noise_std=0.1
    )

    # --- Output Directory Setup --- (Moved cleanup logic here)
    output_path = Path("output")
    if output_path.exists():
        print("Cleaning up previous output directory...")
        # More robust cleanup
        import shutil
        try:
            shutil.rmtree(output_path)
            print("Previous output directory removed.")
        except OSError as e:
            print(f"Error removing directory {output_path}: {e}")
            return # Stop if cleanup fails

    # Create output directories
    azure_output_path = output_path / "azure_maps_images"
    google_output_path = output_path / "google_maps_images"
    flight_data_path = Path("flight_data") # Keep flight data separate? Or move under output? Let's keep it separate for now.

    try:
        azure_output_path.mkdir(parents=True, exist_ok=True)
        google_output_path.mkdir(parents=True, exist_ok=True)
        flight_data_path.mkdir(parents=True, exist_ok=True)
        print("Output directories created.")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        return # Stop if directory creation fails
    # ---------------------------

    # Simulate flight
    # Get initial state for plotting start point if needed
    lat_hist, lon_hist, alt_hist = [drone.lat], [drone.lng], [drone.altitude]
    ecef_pos_hist = [drone.position.copy()] # Store ECEF positions for plotting

    print("Starting simulation loop...")
    for t in range(duration):
        drone.step()
        current_lat, current_lng, current_alt, current_vel_ecef = drone.get_state()
        current_pos_ecef = drone.position # Access ECEF position directly

        # Store history
        lat_hist.append(current_lat)
        lon_hist.append(current_lng)
        alt_hist.append(current_alt)
        ecef_pos_hist.append(current_pos_ecef.copy())

        if t % sample_interval == 0:
            print(f"Time: {t:03d}s, Lat/Lng/Alt: {current_lat:.6f}, {current_lng:.6f}, {current_alt:.2f}m")

            # --- Image Fetching --- (Error handling added)
            try:
                # Get Azure Maps image
                azure_zoom = calculate_azure_zoom(current_alt, current_lat)
                azure_image = get_azure_maps_image(
                    latitude=current_lat,
                    longitude=current_lng,
                    zoom=azure_zoom,
                    size=256,
                    scale=2
                )
                if azure_image:
                    azure_image_path = azure_output_path / f"frame_{t:03d}.png"
                    azure_image.save(azure_image_path)

                # Get Google Maps image
                google_zoom = calculate_google_zoom(current_alt, current_lat)
                google_image = get_static_map(
                    current_lat,
                    current_lng,
                    zoom=google_zoom,
                    scale=2,
                    size="256x256"
                )
                if google_image:
                    google_image_path = google_output_path / f"frame_{t:03d}.png"
                    google_image.save(google_image_path)

            except Exception as e:
                print(f"Warning: Error fetching/saving map image at step {t}: {e}")
            # --------------------

            # Add a small delay between requests to respect rate limits
            time.sleep(0.5) # Reduced sleep slightly

    print("Simulation finished.")
    # Plot flight path using ECEF coordinates
    plot_flight_path_ecef(ecef_pos_hist, flight_data_path)

def plot_flight_path_ecef(ecef_positions, output_dir: Path):
    """Plots the flight path using ECEF coordinates."""
    if not ecef_positions:
        print("No positions recorded to plot.")
        return

    positions = np.array(ecef_positions) # Shape (N, 3)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot flight path - Use ECEF X, Y, Z
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Flight Path (ECEF)')
    # Plot start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='x', s=100, label='End')

    # Add labels and title
    ax.set_xlabel('ECEF X (meters)')
    ax.set_ylabel('ECEF Y (meters)')
    ax.set_zlabel('ECEF Z (meters)')
    ax.set_title('Drone Flight Path (ECEF Coordinates)')
    ax.legend()

    # Ensure equal aspect ratio for better visualization if needed, might distort significantly for ECEF
    # ax.set_aspect('equal', adjustable='box') # Uncomment carefully

    # Save the plot to a file
    plot_path = output_dir / 'flight_path_ecef.png'
    try:
        plt.savefig(plot_path)
        print(f"Flight path plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    # Example flight parameters
    start_lat = 40.7128  # New York City approx lat
    start_lng = -74.0060 # New York City approx lon
    start_height_above_ground = 100  # Start 100 meters above the ground level at lat/lon
    # Initial velocity in local ENU frame (East, North, Up) meters/step (or m/s if step is 1s)
    velocity_enu = np.array([5.0, 5.0, 0.5])  # e.g., Moving Northeast and slightly up

    simulate_drone_flight(
        start_lat=start_lat,
        start_lng=start_lng,
        start_height_above_ground=start_height_above_ground,
        velocity_enu=velocity_enu,
        duration=60,  # Shorter duration for testing
        sample_interval=5 # Sample every 5 steps
    ) 