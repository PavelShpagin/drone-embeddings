from src.coordinates.xyz import get_xyz_coordinates
from src.api.maps import get_static_map
from PIL import Image
import sys
import os
from pathlib import Path
import numpy as np
from src.api.elevation import get_elevation

def interpolate_image(image, scale, size=1024,debug=False):
    """
    Process the image: load, resize, and crop
    """
    # Resize
    img_resized = image.resize((int(scale*size), int(scale*size)), Image.Resampling.LANCZOS)
    if debug:
        print(f"Resized to: {img_resized.size}")
    
    # Crop center
    left = (img_resized.width - size) // 2
    top = (img_resized.height - size) // 2
    right = left + size
    bottom = top + size
    
    img_cropped = img_resized.crop((left, top, right, bottom))
    if debug:
        print(f"Cropped to: {img_cropped.size}")
    
    return img_cropped

def main(debug=False):
    # List of example coordinates
    locations = [
        {"name": "San Francisco", "lat": 37.7749, "lng": -122.4194},
        {"name": "New York", "lat": 40.7128, "lng": -74.0060},
        {"name": "London", "lat": 51.5074, "lng": -0.1278},
        {"name": "Tokyo", "lat": 35.6895, "lng": 139.6917},
        {"name": "Sydney", "lat": -33.8688, "lng": 151.2093}
    ]
    heights = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    for height in heights:
        for location in locations:
            lat = location["lat"]
            lng = location["lng"]
            name = location["name"]

            ground_elevation = get_elevation(lat, lng) + height
            zoom = -np.log2(ground_elevation * 2 / (1024 * 156543.03392 * np.cos(lat * np.pi / 180)))
            print(f"Zoom for {name}: {zoom}")

            image = get_static_map(lat, lng, zoom=int(zoom), scale=2)

            scale = 2**(zoom - int(zoom))
            image_file_interpolated = interpolate_image(image, scale, 1024, debug)

            image_file_interpolated.save(f"images/{name}_{height}m_interpolated.png", "PNG", quality=100)

        # coords = get_xyz_coordinates(lat, lng, altitude_above_ground=10000)

        # if coords:
        #     print(f"\nCoordinate details for {name}:")
        #     print(f"Ground elevation: {coords['ground_elevation']:.2f} meters")
        #     print(f"Total altitude: {coords['total_altitude']:.2f} meters")
        #     print("\nXYZ Coordinates (meters):")
        #     print(f"X: {coords['x']:.2f}")
        #     print(f"Y: {coords['y']:.2f}")
        #     print(f"Z: {coords['z']:.2f}")

        #     # Get snapshot
        #     image_file_zoomed = get_static_map(lat, lng, zoom=int(zoom)+1, filename=f"{name}_image_file_zoomed.png")
        #     image_file = get_static_map(lat, lng, zoom=int(zoom), filename=f"{name}_image_file.png", scale=2)

        #     if image_file:
        #         print(f"\nSnapshot saved as: {image_file}")

        #     # After getting the snapshot
        #     if image_file:
        #         processed_file = process_image(
        #             image_file,
        #             f"{name}_processed_snapshot.png",
        #             debug=debug
        #         )
        #         if processed_file and debug:
        #             print(f"Successfully processed image for {name}: {processed_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    main(debug=args.debug) 