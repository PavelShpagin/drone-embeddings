import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
from src.image_metadata import extract_metadata, ImageMetadata
from datetime import datetime, timedelta

def analyze_frame_gaps(frame_numbers: List[int], fps: float) -> Dict:
    """
    Analyze gaps between frames and calculate expected time between frames.
    Returns statistics about frame gaps and expected frame times.
    """
    frame_numbers = np.array(frame_numbers)
    frame_diffs = np.diff(frame_numbers)
    
    # Calculate expected time between frames
    expected_time = 1.0 / fps  # seconds per frame
    
    # Find gaps (where frame difference > 1)
    gaps = frame_diffs > 1
    gap_sizes = frame_diffs[gaps]
    gap_positions = np.where(gaps)[0]
    
    # Calculate statistics
    stats = {
        'total_frames': len(frame_numbers),
        'frame_range': (min(frame_numbers), max(frame_numbers)),
        'expected_frames': max(frame_numbers) - min(frame_numbers) + 1,
        'missing_frames': max(frame_numbers) - min(frame_numbers) + 1 - len(frame_numbers),
        'expected_time_per_frame': expected_time,
        'gaps': {
            'count': len(gap_sizes),
            'sizes': gap_sizes.tolist(),
            'positions': gap_positions.tolist(),
            'total_missing_frames': np.sum(gap_sizes - 1),
            'max_gap': int(max(gap_sizes)) if len(gap_sizes) > 0 else 0,
            'min_gap': int(min(gap_sizes)) if len(gap_sizes) > 0 else 0,
            'mean_gap': float(np.mean(gap_sizes)) if len(gap_sizes) > 0 else 0
        }
    }
    
    return stats

def calculate_distances(positions: List[Tuple[float, float]], heights: List[float], 
                       frame_numbers: List[int], fps: float, min_height: float = 90.0) -> Dict:
    """
    Calculate distances between consecutive frames, both actual and expected.
    Returns distances only for frames where both points are above min_height.
    """
    distances_2d = []
    distances_3d = []
    expected_distances = []
    frame_gaps = []
    actual_times = []
    
    for i in range(1, len(positions)):
        if heights[i] >= min_height and heights[i-1] >= min_height:
            # Calculate actual distances
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dist_2d = np.sqrt(dx*dx + dy*dy)
            distances_2d.append(dist_2d)
            
            dz = heights[i] - heights[i-1]
            dist_3d = np.sqrt(dx*dx + dy*dy + dz*dz)
            distances_3d.append(dist_3d)
            
            # Calculate frame gap and expected distance
            frame_gap = frame_numbers[i] - frame_numbers[i-1]
            frame_gaps.append(frame_gap)
            
            # Calculate actual time between frames
            actual_time = frame_gap / fps
            actual_times.append(actual_time)
            
            # Calculate expected distance based on average speed
            # We'll calculate this after we have all the data
            expected_distances.append(None)  # Placeholder
    
    # Calculate average speed (excluding outliers)
    speeds = np.array(distances_2d) / np.array(actual_times)
    # Remove outliers (speeds more than 3 standard deviations from mean)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    valid_speeds = speeds[(speeds >= mean_speed - 3*std_speed) & 
                         (speeds <= mean_speed + 3*std_speed)]
    avg_speed = np.mean(valid_speeds)
    
    # Calculate expected distances
    expected_distances = [avg_speed * (1/fps) * gap for gap in frame_gaps]
    
    return {
        'distances_2d': distances_2d,
        'distances_3d': distances_3d,
        'expected_distances': expected_distances,
        'frame_gaps': frame_gaps,
        'actual_times': actual_times,
        'speeds': speeds.tolist(),
        'avg_speed': avg_speed,
        'speed_stats': {
            'mean': float(np.mean(speeds)),
            'median': float(np.median(speeds)),
            'std': float(np.std(speeds)),
            'min': float(np.min(speeds)),
            'max': float(np.max(speeds))
        }
    }

def calculate_consecutive_mtc_distances(mtc_list, pos2d_list, height_list, min_height=90.0):
    """
    Calculate distances between consecutive frames where mtc is strictly consecutive and height > min_height.
    Returns a list of distances and the average.
    """
    distances = []
    for i in range(1, len(mtc_list)):
        if (
            height_list[i] >= min_height and height_list[i-1] >= min_height and
            mtc_list[i] is not None and mtc_list[i-1] is not None and
            mtc_list[i] == mtc_list[i-1] + 1 and
            pos2d_list[i] is not None and pos2d_list[i-1] is not None
        ):
            dx = pos2d_list[i][0] - pos2d_list[i-1][0]
            dy = pos2d_list[i][1] - pos2d_list[i-1][1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances.append(dist)
    
    if distances:
        avg_distance = np.mean(distances)
    else:
        fallback_distances = []
        for i in range(1, len(mtc_list)):
            if (height_list[i] >= min_height and height_list[i-1] >= min_height and pos2d_list[i] is not None and pos2d_list[i-1] is not None):
                dx = pos2d_list[i][0] - pos2d_list[i-1][0]
                dy = pos2d_list[i][1] - pos2d_list[i-1][1]
                dist = np.sqrt(dx*dx + dy*dy)
                fallback_distances.append(dist)
        avg_distance = np.mean(fallback_distances) if fallback_distances else 0.0
    
    return distances, avg_distance

def get_image_files(folder_path: str) -> List[str]:
    """Get all image files from the specified folder."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    return sorted([
        str(p) for p in Path(folder_path).glob('*')
        if p.suffix.lower() in image_extensions
    ])

def extract_flight_data(folder_path: str) -> Dict[str, Dict]:
    """
    Extract all flight data from images, including frames below 90m.
    
    Returns:
        Dictionary containing lists of heights, positions, frame_numbers, and heights_above_90
    """
    image_files = get_image_files(folder_path)
    heights = []
    positions = []
    frame_numbers = []
    heights_above_90 = []
    positions_above_90 = []
    frame_numbers_above_90 = []
    fps_values = []
    mtc_list = []
    mtc_list_above_90 = []
    
    print("Extracting flight data...")
    for img_path in image_files:
        try:
            metadata = extract_metadata(img_path)
            if (metadata.telemetry and 
                metadata.telemetry.height is not None and
                metadata.telemetry.position_2d is not None and
                metadata.telemetry.fms is not None):
                
                height = metadata.telemetry.height
                position = metadata.telemetry.position_2d
                frame = metadata.telemetry.fms
                
                # Store all data
                heights.append(height)
                positions.append(position)
                frame_numbers.append(frame)
                
                # Store FPS if available
                if metadata.telemetry.fps:
                    fps_values.append(metadata.telemetry.fps[0])
                
                # Store data for frames above 90m separately
                if height >= 90.0:
                    heights_above_90.append(height)
                    positions_above_90.append(position)
                    frame_numbers_above_90.append(frame)
                    mtc_list_above_90.append(metadata.telemetry.mtc if hasattr(metadata.telemetry, 'mtc') else None)
                
                mtc = metadata.telemetry.mtc if hasattr(metadata.telemetry, 'mtc') else None
                mtc_list.append(mtc)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate average FPS
    avg_fps = np.mean(fps_values) if fps_values else None
    
    # Analyze frame gaps
    gap_stats = analyze_frame_gaps(frame_numbers, avg_fps) if avg_fps else None
    
    # Calculate distances for frames above 90m
    distance_stats = calculate_distances(positions_above_90, heights_above_90, 
                                       frame_numbers_above_90, avg_fps) if avg_fps else None
    
    # Calculate consecutive mtc distances for height > 90
    if not (len(mtc_list_above_90) == len(positions_above_90) == len(heights_above_90)):
        print(f"DEBUG: Length mismatch: mtc_list_above_90={len(mtc_list_above_90)}, positions_above_90={len(positions_above_90)}, heights_above_90={len(heights_above_90)}")
    mtc_distances, avg_mtc_distance = calculate_consecutive_mtc_distances(
        mtc_list_above_90, positions_above_90, heights_above_90, min_height=90.0)
    
    return {
        'all': {
            'heights': heights,
            'positions': positions,
            'frame_numbers': frame_numbers,
            'fps': avg_fps,
            'gap_stats': gap_stats,
            'mtc_list': mtc_list
        },
        'above_90': {
            'heights': heights_above_90,
            'positions': positions_above_90,
            'frame_numbers': frame_numbers_above_90,
            'distance_stats': distance_stats,
            'mtc_distances': mtc_distances,
            'avg_mtc_distance': avg_mtc_distance
        }
    }

def plot_flight_data(flight_data: Dict[str, Dict], output_dir: str = "flight_visualizations"):
    """Create and save visualizations of the flight data."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract data
    all_data = flight_data['all']
    above_90_data = flight_data['above_90']
    
    heights = all_data['heights']
    positions = np.array(all_data['positions'])
    frame_numbers = all_data['frame_numbers']
    gap_stats = all_data['gap_stats']
    distance_stats = above_90_data['distance_stats']
    
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot height over time for all frames
    ax1.plot(frame_numbers, heights, 'b-', linewidth=1, alpha=0.7, label='All frames')
    
    # Add 90m threshold line
    ax1.axhline(y=90.0, color='r', linestyle='--', alpha=0.5, 
                label='90m threshold')
    
    # Add mean height line for frames above 90m
    mean_height_above_90 = np.mean(above_90_data['heights'])
    ax1.axhline(y=mean_height_above_90, color='g', linestyle='--', alpha=0.5,
                label=f'Mean height above 90m: {mean_height_above_90:.1f}m')
    
    ax1.set_title('Drone Height Over Time (All Frames)')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Height (meters)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot trajectory
    scatter = ax2.scatter(x_coords, y_coords, c=heights, cmap='viridis', 
                         s=10, alpha=0.6)
    ax2.plot(x_coords, y_coords, 'k-', linewidth=0.5, alpha=0.3)
    
    # Add start and end points
    ax2.scatter(x_coords[0], y_coords[0], c='green', s=100, label='Start', zorder=3)
    ax2.scatter(x_coords[-1], y_coords[-1], c='red', s=100, label='End', zorder=3)
    
    # Add colorbar for height
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Height (meters)')
    
    ax2.set_title('Drone Trajectory (All Frames)')
    ax2.set_xlabel('X Position (meters)')
    ax2.set_ylabel('Y Position (meters)')
    ax2.grid(True)
    ax2.legend()
    
    # Equal aspect ratio for trajectory plot
    ax2.set_aspect('equal')
    
    # Add some padding to the trajectory plot
    x_pad = (max(x_coords) - min(x_coords)) * 0.1
    y_pad = (max(y_coords) - min(y_coords)) * 0.1
    ax2.set_xlim(min(x_coords) - x_pad, max(x_coords) + x_pad)
    ax2.set_ylim(min(y_coords) - y_pad, max(y_coords) + y_pad)
    
    if distance_stats:
        # Plot actual vs expected distances
        frame_indices = range(len(distance_stats['distances_2d']))
        ax3.plot(frame_indices, distance_stats['distances_2d'], 'b-', alpha=0.7, 
                label='Actual distance')
        ax3.plot(frame_indices, distance_stats['expected_distances'], 'r--', alpha=0.7,
                label='Expected distance (no gaps)')
        
        # Add mean lines
        mean_actual = np.mean(distance_stats['distances_2d'])
        mean_expected = np.mean(distance_stats['expected_distances'])
        ax3.axhline(y=mean_actual, color='blue', linestyle=':', alpha=0.5,
                    label=f'Mean actual: {mean_actual:.2f}m')
        ax3.axhline(y=mean_expected, color='red', linestyle=':', alpha=0.5,
                    label=f'Mean expected: {mean_expected:.2f}m')
        
        ax3.set_title('Actual vs Expected Distances Between Frames (Height > 90m)')
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('Distance (meters)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot speed distribution
        speeds = distance_stats['speeds']
        ax4.hist(speeds, bins=50, alpha=0.7, color='blue', label='Actual speeds')
        ax4.axvline(x=distance_stats['avg_speed'], color='r', linestyle='--', alpha=0.5,
                    label=f'Average speed: {distance_stats["avg_speed"]:.2f} m/s')
        
        ax4.set_title('Speed Distribution (Height > 90m)')
        ax4.set_xlabel('Speed (m/s)')
        ax4.set_ylabel('Number of Frames')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path / 'drone_flight_visualization_all_frames.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create additional visualization showing height distribution
    plt.figure(figsize=(10, 6))
    plt.hist(heights, bins=50, alpha=0.7, color='blue', label='All frames')
    plt.hist(above_90_data['heights'], bins=30, alpha=0.7, color='red', 
             label='Frames above 90m')
    plt.axvline(x=90.0, color='r', linestyle='--', alpha=0.5, 
                label='90m threshold')
    plt.axvline(x=mean_height_above_90, color='g', linestyle='--', alpha=0.5,
                label=f'Mean height above 90m: {mean_height_above_90:.1f}m')
    
    plt.title('Height Distribution')
    plt.xlabel('Height (meters)')
    plt.ylabel('Number of Frames')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'height_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    folder_path = "real_data/stream"
    try:
        # Extract flight data
        flight_data = extract_flight_data(folder_path)
        
        all_data = flight_data['all']
        above_90_data = flight_data['above_90']
        
        if not all_data['heights']:
            print("No valid flight data found!")
            return
        
        print(f"\nExtracted {len(all_data['heights'])} total data points")
        print(f"Frames above 90m: {len(above_90_data['heights'])}")
        print(f"Height range: {min(all_data['heights']):.1f}m to {max(all_data['heights']):.1f}m")
        print(f"Frame range: {min(all_data['frame_numbers'])} to {max(all_data['frame_numbers'])}")
        
        # Print frame gap statistics
        if all_data['gap_stats']:
            gap_stats = all_data['gap_stats']
            print("\nFrame Gap Statistics:")
            print(f"Total frames: {gap_stats['total_frames']}")
            print(f"Frame range: {gap_stats['frame_range'][0]} to {gap_stats['frame_range'][1]}")
            print(f"Expected frames: {gap_stats['expected_frames']}")
            print(f"Missing frames: {gap_stats['missing_frames']}")
            print(f"Expected time per frame: {gap_stats['expected_time_per_frame']*1000:.1f}ms")
            print("\nGaps:")
            print(f"  Number of gaps: {gap_stats['gaps']['count']}")
            print(f"  Total missing frames: {gap_stats['gaps']['total_missing_frames']}")
            print(f"  Max gap size: {gap_stats['gaps']['max_gap']} frames")
            print(f"  Min gap size: {gap_stats['gaps']['min_gap']} frames")
            print(f"  Mean gap size: {gap_stats['gaps']['mean_gap']:.1f} frames")
        
        # Print distance and speed statistics
        if above_90_data['distance_stats']:
            dist_stats = above_90_data['distance_stats']
            print("\nDistance and Speed Statistics (for frames above 90m):")
            print(f"Average speed: {dist_stats['avg_speed']:.2f} m/s")
            print("\nSpeed statistics:")
            print(f"  Mean: {dist_stats['speed_stats']['mean']:.2f} m/s")
            print(f"  Median: {dist_stats['speed_stats']['median']:.2f} m/s")
            print(f"  Std: {dist_stats['speed_stats']['std']:.2f} m/s")
            print(f"  Min: {dist_stats['speed_stats']['min']:.2f} m/s")
            print(f"  Max: {dist_stats['speed_stats']['max']:.2f} m/s")
        
        # Print average distance between consecutive mtc frames (height > 90)
        print("\nAverage distance between consecutive frames (using mtc, height > 90m):")
        print(f"  Number of consecutive mtc pairs: {len(above_90_data['mtc_distances'])}")
        print(f"  Mean distance: {above_90_data['avg_mtc_distance']:.3f} meters")
        if above_90_data['mtc_distances']:
            print(f"  Min: {np.min(above_90_data['mtc_distances']):.3f} m, Max: {np.max(above_90_data['mtc_distances']):.3f} m, Std: {np.std(above_90_data['mtc_distances']):.3f} m")
        
        # Create visualizations
        plot_flight_data(flight_data)
        print("\nVisualizations saved to 'flight_visualizations' directory")
        
    except Exception as e:
        print(f"Error processing flight data: {e}")

if __name__ == "__main__":
    main() 