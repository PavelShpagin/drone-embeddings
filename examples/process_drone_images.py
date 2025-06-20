import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict

from src.image_metadata import extract_metadata, ImageMetadata, calculate_speed_between_images

def get_image_files(folder_path: str) -> List[str]:
    """Get all image files from the specified folder."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    return sorted([
        str(p) for p in Path(folder_path).glob('*')
        if p.suffix.lower() in image_extensions
    ])

def process_images(folder_path: str, min_height: float = 90.0) -> Tuple[List[ImageMetadata], Dict]:
    """
    Process all images in the folder and calculate statistics.
    
    Args:
        folder_path: Path to folder containing images
        min_height: Minimum height threshold for filtering images
        
    Returns:
        Tuple of (list of metadata for images above height threshold, statistics dictionary)
    """
    image_files = get_image_files(folder_path)
    if not image_files:
        raise FileNotFoundError(f"No image files found in {folder_path}")
    
    print(f"Found {len(image_files)} images. Processing...")
    
    # Process all images
    all_metadata = []
    for img_path in image_files:
        try:
            metadata = extract_metadata(img_path)
            all_metadata.append(metadata)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Filter by height
    filtered_metadata = [
        m for m in all_metadata 
        if m.telemetry and m.telemetry.height is not None and m.telemetry.height >= min_height
    ]
    
    # Calculate statistics
    stats = {
        "total_images": len(all_metadata),
        "images_above_height_threshold": len(filtered_metadata),
        "height_threshold": min_height,
        "height_stats": {
            "min": float('inf'),
            "max": float('-inf'),
            "mean": 0.0,
            "std": 0.0
        },
        "speed_stats": {
            "min": float('inf'),
            "max": float('-inf'),
            "mean": 0.0,
            "std": 0.0
        },
        "flight_stats": {
            "total_frames": 0,
            "total_duration_seconds": 0.0,
            "average_fps": 0.0,
            "flight_modes": defaultdict(int)
        },
        "position_stats": {
            "min_x": float('inf'),
            "max_x": float('-inf'),
            "min_y": float('inf'),
            "max_y": float('-inf'),
            "total_distance": 0.0
        },
        "tilt_stats": {
            "min": float('inf'),
            "max": float('-inf'),
            "mean": 0.0,
            "std": 0.0
        },
        "acceleration_stats": {
            "min": float('inf'),
            "max": float('-inf'),
            "mean": 0.0,
            "std": 0.0
        }
    }
    
    # Calculate height statistics
    heights = [m.telemetry.height for m in filtered_metadata if m.telemetry and m.telemetry.height is not None]
    if heights:
        stats["height_stats"].update({
            "min": min(heights),
            "max": max(heights),
            "mean": np.mean(heights),
            "std": np.std(heights)
        })
    
    # Calculate speed statistics and track positions
    speeds = []
    total_distance = 0.0
    valid_pairs = 0
    
    for i in range(len(filtered_metadata) - 1):
        speed = calculate_speed_between_images(filtered_metadata[i], filtered_metadata[i + 1])
        if speed is not None:
            speeds.append(speed)
            valid_pairs += 1
            
            # Update position bounds and distance
            pos1 = filtered_metadata[i].telemetry.position_2d
            pos2 = filtered_metadata[i + 1].telemetry.position_2d
            if pos1 and pos2:
                stats["position_stats"]["min_x"] = min(stats["position_stats"]["min_x"], pos1[0], pos2[0])
                stats["position_stats"]["max_x"] = max(stats["position_stats"]["max_x"], pos1[0], pos2[0])
                stats["position_stats"]["min_y"] = min(stats["position_stats"]["min_y"], pos1[1], pos2[1])
                stats["position_stats"]["max_y"] = max(stats["position_stats"]["max_y"], pos1[1], pos2[1])
                
                # Calculate distance
                distance = np.linalg.norm(np.array(pos2) - np.array(pos1))
                total_distance += distance
    
    if speeds:
        stats["speed_stats"].update({
            "min": min(speeds),
            "max": max(speeds),
            "mean": np.mean(speeds),
            "std": np.std(speeds)
        })
        stats["position_stats"]["total_distance"] = total_distance
    
    # Calculate flight statistics
    if filtered_metadata:
        # Get FMS numbers for duration calculation
        fms_numbers = [m.telemetry.fms for m in filtered_metadata if m.telemetry and m.telemetry.fms is not None]
        if fms_numbers:
            stats["flight_stats"]["total_frames"] = max(fms_numbers) - min(fms_numbers) + 1
            
            # Calculate duration using FPS
            fps_values = [m.telemetry.fps[0] for m in filtered_metadata if m.telemetry and m.telemetry.fps]
            if fps_values:
                avg_fps = np.mean(fps_values)
                stats["flight_stats"]["average_fps"] = avg_fps
                stats["flight_stats"]["total_duration_seconds"] = stats["flight_stats"]["total_frames"] / avg_fps
        
        # Count flight modes
        for m in filtered_metadata:
            if m.telemetry and m.telemetry.flight_mode:
                stats["flight_stats"]["flight_modes"][m.telemetry.flight_mode] += 1
    
    # Calculate tilt statistics
    tilts = []
    for m in filtered_metadata:
        if m.telemetry and m.telemetry.tilt:
            tilts.extend(m.telemetry.tilt)
    
    if tilts:
        stats["tilt_stats"].update({
            "min": min(tilts),
            "max": max(tilts),
            "mean": np.mean(tilts),
            "std": np.std(tilts)
        })
    
    # Calculate acceleration statistics
    accelerations = []
    for m in filtered_metadata:
        if m.telemetry and m.telemetry.acceleration:
            # Calculate magnitude of acceleration
            accel_mag = np.linalg.norm(m.telemetry.acceleration)
            accelerations.append(accel_mag)
    
    if accelerations:
        stats["acceleration_stats"].update({
            "min": min(accelerations),
            "max": max(accelerations),
            "mean": np.mean(accelerations),
            "std": np.std(accelerations)
        })
    
    return filtered_metadata, stats

def main():
    # Process images from the stream folder
    folder_path = "real_data/stream"
    try:
        filtered_metadata, stats = process_images(folder_path, min_height=90.0)
        
        # Print statistics
        print("\nDrone Flight Statistics:")
        print("=" * 50)
        print(f"Total images processed: {stats['total_images']}")
        print(f"Images above {stats['height_threshold']}m: {stats['images_above_height_threshold']}")
        
        print("\nHeight Statistics (meters):")
        print(f"  Min: {stats['height_stats']['min']:.2f}")
        print(f"  Max: {stats['height_stats']['max']:.2f}")
        print(f"  Mean: {stats['height_stats']['mean']:.2f}")
        print(f"  Std: {stats['height_stats']['std']:.2f}")
        
        print("\nSpeed Statistics (m/s):")
        print(f"  Min: {stats['speed_stats']['min']:.2f}")
        print(f"  Max: {stats['speed_stats']['max']:.2f}")
        print(f"  Mean: {stats['speed_stats']['mean']:.2f}")
        print(f"  Std: {stats['speed_stats']['std']:.2f}")
        
        print("\nFlight Information:")
        print(f"  Total Frames: {stats['flight_stats']['total_frames']}")
        print(f"  Duration: {stats['flight_stats']['total_duration_seconds']:.1f} seconds")
        print(f"  Average FPS: {stats['flight_stats']['average_fps']:.1f}")
        print("\n  Flight Modes:")
        for mode, count in stats['flight_stats']['flight_modes'].items():
            print(f"    {mode}: {count} frames")
        
        print("\nPosition Statistics:")
        print(f"  Total Distance: {stats['position_stats']['total_distance']:.2f} meters")
        print(f"  X Range: [{stats['position_stats']['min_x']:.2f}, {stats['position_stats']['max_x']:.2f}]")
        print(f"  Y Range: [{stats['position_stats']['min_y']:.2f}, {stats['position_stats']['max_y']:.2f}]")
        
        print("\nTilt Statistics (degrees):")
        print(f"  Min: {stats['tilt_stats']['min']:.2f}")
        print(f"  Max: {stats['tilt_stats']['max']:.2f}")
        print(f"  Mean: {stats['tilt_stats']['mean']:.2f}")
        print(f"  Std: {stats['tilt_stats']['std']:.2f}")
        
        print("\nAcceleration Statistics (m/s²):")
        print(f"  Min: {stats['acceleration_stats']['min']:.2f}")
        print(f"  Max: {stats['acceleration_stats']['max']:.2f}")
        print(f"  Mean: {stats['acceleration_stats']['mean']:.2f}")
        print(f"  Std: {stats['acceleration_stats']['std']:.2f}")
        
        # Save detailed statistics to JSON
        output_file = "drone_flight_stats.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\nDetailed statistics saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing images: {e}")

if __name__ == "__main__":
    main() 