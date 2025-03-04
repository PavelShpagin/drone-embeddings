import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from pathlib import Path

# Import the GeoLocalizer from drone_inference.py
from examples.drone_inference import GeoLocalizer, lat_lng_alt_to_xyz
from src.simulation.drone import DroneFlight
from src.google_maps import get_static_map, calculate_google_zoom
from src.azure_maps import get_azure_maps_image, calculate_azure_zoom
from scipy import stats
import yaml
import torch
from PIL import Image
from torchvision import transforms
from scipy.spatial import distance

def generate_method_visualizations(df, true_positions, drifted_positions, 
                                  method1_positions, method2_positions, 
                                  method3_positions, method4_positions):
    """
    Generate GIFs and error plots for each correction method
    
    Args:
        df: DataFrame with error metrics
        true_positions, drifted_positions: Arrays of true and drifted positions
        method1_positions, method2_positions, method3_positions, method4_positions: 
            Arrays of corrected positions for each method
    """
    os.makedirs('output/drone_inference/methods', exist_ok=True)
    
    # Generate trajectory GIFs for each method
    print("Generating trajectory GIFs for each method...")
    
    method_positions = [
        method1_positions, 
        method2_positions, 
        method3_positions, 
        method4_positions
    ]
    
    method_names = [
        "Top1_Constant", 
        "TopK_Mean", 
        "STD_Weighted", 
        "Gaussian_Product"
    ]
    
    method_titles = [
        "Top-1 Constant Shift", 
        "Top-K Mean Shift", 
        "STD-Weighted Shift", 
        "Gaussian Product Shift"
    ]
    
    # Generate a GIF for each method
    for i, (method_pos, name, title) in enumerate(zip(method_positions, method_names, method_titles)):
        print(f"Generating GIF for {name}...")
        create_method_gif(true_positions, drifted_positions, method_pos, 
                          name, title, error_data=df)
    
    # Generate error plots for each method
    print("Generating error plots...")
    plot_error_comparison(df)
    
    # Generate error plots for individual methods
    for i, (name, title) in enumerate(zip(method_names, method_titles)):
        method_col = f'error_method{i+1}'
        plot_individual_error(df, 'error_original', method_col, title, name)
        
    print("All visualizations completed.")

def create_method_gif(true_positions, drift_positions, method_positions, 
                     method_name, method_title, error_data=None, max_frames=50):
    """
    Create an animated GIF showing trajectories for a specific method
    
    Args:
        true_positions: Array of true drone positions
        drift_positions: Array of drifted positions
        method_positions: Array of corrected positions for this method
        method_name: Name of method for filename
        method_title: Title to display on animation
        error_data: DataFrame with error metrics
        max_frames: Maximum number of frames (default 50)
    """
    # Limit to max_frames
    num_frames = min(len(true_positions), max_frames)
    true_positions = true_positions[:num_frames]
    drift_positions = drift_positions[:num_frames]
    method_positions = method_positions[:num_frames-1]  # Method positions may have one less
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set view limits
    all_pos = np.vstack([true_positions, drift_positions])
    if len(method_positions) > 0:
        all_pos = np.vstack([all_pos, method_positions])
    
    x_min, y_min, z_min = np.min(all_pos, axis=0) - 10
    x_max, y_max, z_max = np.max(all_pos, axis=0) + 10
    
    # Initialize plot objects
    true_line, = ax.plot([], [], [], 'g-', linewidth=2, label='True Trajectory')
    drift_line, = ax.plot([], [], [], 'r-', linewidth=2, label='Drifted Trajectory')
    method_line, = ax.plot([], [], [], 'b-', linewidth=2, label=f'Corrected Trajectory')
    
    true_point = ax.scatter([], [], [], color='green', s=100, marker='o')
    drift_point = ax.scatter([], [], [], color='red', s=100, marker='o')
    method_point = ax.scatter([], [], [], color='blue', s=100, marker='o')
    
    # Text for error info
    error_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)
    
    # Animation init function
    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Altitude (meters)')
        ax.set_title(f'Drone Trajectory Correction\n{method_title}')
        ax.legend(loc='upper right')
        
        # Initialize with empty data
        true_line.set_data([], [])
        true_line.set_3d_properties([])
        drift_line.set_data([], [])
        drift_line.set_3d_properties([])
        method_line.set_data([], [])
        method_line.set_3d_properties([])
        
        true_point._offsets3d = ([], [], [])
        drift_point._offsets3d = ([], [], [])
        method_point._offsets3d = ([], [], [])
        
        error_text.set_text("")
        
        return true_line, drift_line, method_line, true_point, drift_point, method_point, error_text
    
    # Animation update function
    def update(frame):
        # Update true trajectory
        true_line.set_data(true_positions[:frame+1, 0], true_positions[:frame+1, 1])
        true_line.set_3d_properties(true_positions[:frame+1, 2])
        true_point._offsets3d = ([true_positions[frame, 0]], 
                               [true_positions[frame, 1]], 
                               [true_positions[frame, 2]])
        
        # Update drifted trajectory
        drift_line.set_data(drift_positions[:frame+1, 0], drift_positions[:frame+1, 1])
        drift_line.set_3d_properties(drift_positions[:frame+1, 2])
        drift_point._offsets3d = ([drift_positions[frame, 0]], 
                                [drift_positions[frame, 1]], 
                                [drift_positions[frame, 2]])
        
        # Update method trajectory - may need to handle edge cases
        if frame > 0 and frame <= len(method_positions):
            method_line.set_data(method_positions[:frame, 0], method_positions[:frame, 1])
            method_line.set_3d_properties(method_positions[:frame, 2])
            method_point._offsets3d = ([method_positions[frame-1, 0]], 
                                     [method_positions[frame-1, 1]], 
                                     [method_positions[frame-1, 2]])
        
        # Update error text if we have error data
        if error_data is not None and frame < len(error_data):
            method_num = int(method_name.split('_')[0][-1])  # Extract method number
            orig_error = error_data['error_original'].iloc[frame]
            method_error = error_data[f'error_method{method_num}'].iloc[frame]
            improvement = (orig_error - method_error) / orig_error * 100
            
            error_text.set_text(
                f"Frame: {frame}/{num_frames-1}\n"
                f"Original Error: {orig_error:.6f}\n"
                f"Corrected Error: {method_error:.6f}\n"
                f"Improvement: {improvement:.2f}%"
            )
        
        # Update view angle for better perspective
        ax.view_init(elev=30, azim=(frame % 360))
        
        return true_line, drift_line, method_line, true_point, drift_point, method_point, error_text
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                init_func=init, blit=False, repeat=True)
    
    # Save animation
    output_path = f'output/drone_inference/methods/{method_name}_trajectory.gif'
    print(f"Saving animation to {output_path}...")
    
    # Try to save as GIF
    try:
        ani.save(output_path, writer='pillow', fps=5, dpi=150)
        print(f"GIF saved successfully")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        # Try alternate writer
        try:
            ani.save(output_path, writer='imagemagick', fps=5, dpi=150)
            print(f"GIF saved with imagemagick")
        except Exception as e2:
            print(f"Error saving with imagemagick: {e2}")
            # Save as MP4 instead
            try:
                mp4_path = output_path.replace('.gif', '.mp4')
                ani.save(mp4_path, writer='ffmpeg', fps=5, dpi=150)
                print(f"Saved as MP4 instead: {mp4_path}")
            except Exception as e3:
                print(f"Could not save animation: {e3}")
    
    plt.close(fig)

def plot_error_comparison(df):
    """Plot comparison of all methods' errors"""
    plt.figure(figsize=(14, 10))
    
    # Extract error columns
    error_cols = [col for col in df.columns if col.startswith('error_')]
    error_labels = {
        'error_original': 'Original Error',
        'error_method1': 'Top-1 Constant',
        'error_method2': 'Top-K Mean',
        'error_method3': 'STD-Weighted',
        'error_method4': 'Gaussian Product'
    }
    
    colors = ['red', 'magenta', 'cyan', 'yellow', 'blue']
    
    # Line plot of errors over time
    plt.subplot(2, 1, 1)
    for col, color in zip(error_cols, colors):
        plt.plot(df['time'], df[col], color=color, linewidth=2, 
                label=error_labels.get(col, col))
    
    plt.xlabel('Time Step')
    plt.ylabel('Error (degrees)')
    plt.title('Error Comparison Across Methods')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Bar plot of average errors
    plt.subplot(2, 1, 2)
    avg_errors = [df[col].mean() for col in error_cols]
    x_pos = np.arange(len(error_cols))
    labels = [error_labels.get(col, col) for col in error_cols]
    
    bars = plt.bar(x_pos, avg_errors, color=colors, alpha=0.7)
    
    # Add percentage improvement labels
    baseline = avg_errors[0]  # Original error
    for i, bar in enumerate(bars[1:], 1):  # Skip the baseline bar
        height = bar.get_height()
        improvement = (baseline - height) / baseline * 100
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.0001,
                f'{improvement:.1f}%',
                ha='center', va='bottom', rotation=0, fontsize=10)
    
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Average Error (degrees)')
    plt.title('Average Error by Method')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('output/drone_inference/error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_individual_error(df, orig_col, method_col, title, method_name):
    """Plot detailed error analysis for a single method"""
    plt.figure(figsize=(14, 12))
    
    # 1. Error over time
    plt.subplot(2, 2, 1)
    plt.plot(df['time'], df[orig_col], 'r-', linewidth=2, label='Original Error')
    plt.plot(df['time'], df[method_col], 'b-', linewidth=2, label='Corrected Error')
    plt.xlabel('Time Step')
    plt.ylabel('Error (degrees)')
    plt.title(f'Error Over Time: {title}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Error reduction percentage over time
    plt.subplot(2, 2, 2)
    reduction = (df[orig_col] - df[method_col]) / df[orig_col] * 100
    plt.plot(df['time'], reduction, 'g-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Error Reduction (%)')
    plt.title(f'Error Reduction Percentage: {title}')
    plt.grid(True, alpha=0.3)
    
    # 3. Error distribution
    plt.subplot(2, 2, 3)
    plt.hist(df[orig_col], bins=15, alpha=0.5, color='red', label='Original Error')
    plt.hist(df[method_col], bins=15, alpha=0.5, color='blue', label='Corrected Error')
    plt.xlabel('Error (degrees)')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative error comparison
    plt.subplot(2, 2, 4)
    cum_orig = df[orig_col].cumsum()
    cum_method = df[method_col].cumsum()
    plt.plot(df['time'], cum_orig, 'r-', linewidth=2, label='Original Cumulative Error')
    plt.plot(df['time'], cum_method, 'b-', linewidth=2, label='Corrected Cumulative Error')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Error (degrees)')
    plt.title(f'Cumulative Error: {title}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'output/drone_inference/methods/{method_name}_error_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_comparison(df, save_path='output/drone_inference/comprehensive_comparison.png'):
    """
    Create a comprehensive comparison plot showing:
    1. Original vs corrected error for each method
    2. Cumulative error comparison
    3. Error reduction percentage
    4. Moving average of error
    
    Args:
        df: DataFrame containing error metrics
        save_path: Path to save the resulting plot
    """
    plt.figure(figsize=(20, 15))
    
    # Color scheme for methods
    colors = {
        'original': 'gray',
        'method1': 'blue',
        'method2': 'green',
        'method3': 'red',
        'method4': 'purple'
    }
    
    method_names = {
        'method1': 'Top-1 Constant',
        'method2': 'Top-K Mean',
        'method3': 'STD-Weighted',
        'method4': 'Gaussian Product'
    }
    
    # 1. Original vs Corrected Error (top left)
    plt.subplot(2, 2, 1)
    plt.plot(df['time'], df['error_original'], color=colors['original'], 
             label='Original Error', alpha=0.3)
    
    window = 20  # Window size for moving average
    for method in ['method1', 'method2', 'method3', 'method4']:
        error_col = f'error_{method}'
        # Plot raw data with low alpha
        plt.plot(df['time'], df[error_col], color=colors[method], alpha=0.2)
        # Plot moving average with high alpha
        plt.plot(df['time'], df[error_col].rolling(window=window).mean(), 
                color=colors[method], label=f'{method_names[method]}', linewidth=2)
    
    plt.xlabel('Frame')
    plt.ylabel('Error (degrees)')
    plt.title('Error Over Time (Moving Average)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Cumulative Error (top right)
    plt.subplot(2, 2, 2)
    plt.plot(df['time'], df['error_original'].cumsum(), color=colors['original'], 
             label='Original', linewidth=2)
    
    for method in ['method1', 'method2', 'method3', 'method4']:
        error_col = f'error_{method}'
        plt.plot(df['time'], df[error_col].cumsum(), color=colors[method], 
                label=method_names[method], linewidth=2)
    
    plt.xlabel('Frame')
    plt.ylabel('Cumulative Error (degrees)')
    plt.title('Cumulative Error Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Error Reduction Percentage (bottom left)
    plt.subplot(2, 2, 3)
    for method in ['method1', 'method2', 'method3', 'method4']:
        error_col = f'error_{method}'
        reduction = (df['error_original'] - df[error_col]) / df['error_original'] * 100
        # Plot raw data with low alpha
        plt.plot(df['time'], reduction, color=colors[method], alpha=0.2)
        # Plot moving average with high alpha
        plt.plot(df['time'], reduction.rolling(window=window).mean(), 
                color=colors[method], label=method_names[method], linewidth=2)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Frame')
    plt.ylabel('Error Reduction (%)')
    plt.title('Error Reduction Percentage Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 4. Method Comparison Box Plot (bottom right)
    plt.subplot(2, 2, 4)
    error_data = [df['error_original']]
    labels = ['Original']
    
    for method in ['method1', 'method2', 'method3', 'method4']:
        error_col = f'error_{method}'
        error_data.append(df[error_col])
        labels.append(method_names[method])
    
    box_colors = [colors['original']] + [colors[f'method{i}'] for i in range(1, 5)]
    bp = plt.boxplot(error_data, labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.xticks(rotation=45)
    plt.ylabel('Error Distribution (degrees)')
    plt.title('Error Distribution by Method')
    plt.grid(True, alpha=0.3)
    
    # Add statistics table
    stats_text = "Statistics Summary:\n"
    for i, method in enumerate(['original'] + [f'method{i}' for i in range(1, 5)]):
        error_col = f'error_{method}'
        mean_error = df[error_col].mean()
        std_error = df[error_col].std()
        total_error = df[error_col].sum()
        if i > 0:
            improvement = ((df['error_original'].mean() - mean_error) / 
                         df['error_original'].mean() * 100)
            stats_text += f"{labels[i]}:\n"
            stats_text += f"  Mean: {mean_error:.4f}° (↓{improvement:.1f}%)\n"
            stats_text += f"  Std: {std_error:.4f}°\n"
            stats_text += f"  Total: {total_error:.4f}°\n"
    
    plt.figtext(1.02, 0.5, stats_text, fontsize=10, va='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def simulate_dual_trajectories(
    start_lat, start_lng, start_altitude, velocity, 
    model_path, config_path,
    drift_std=0.1, drift_velocity=np.array([0.1, -0.05, 0.02]),
    duration=1000, sample_interval=1):
    """
    Simulate trajectories with momentum-based smooth corrections
    """
    # Create output directories
    os.makedirs('output/drone_inference', exist_ok=True)
    
    # Initialize GeoLocalizer
    localizer = GeoLocalizer(model_path, config_path)
    
    # Build database around start point
    localizer.build_database(
        lat_center=start_lat,
        lng_center=start_lng,
        grid_size=15,  # Expanded grid
        base_height=start_altitude
    )
    
    # Initialize true drone flight
    true_drone = DroneFlight(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=start_altitude,
        velocity=velocity,
        noise_std=0.01  # Small noise for true trajectory
    )
    
    # Initialize drifted drone flight
    drifted_drone = DroneFlight(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=start_altitude,
        velocity=velocity + drift_velocity,
        noise_std=drift_std  # Larger noise for drifted trajectory
    )
    
    # Initialize data collectors
    data = []  # Initialize data list here
    true_positions = []
    true_coords = []
    drifted_positions = []
    drifted_coords = []
    
    # Initialize correction methods
    method1_drone_pos = [(start_lat, start_lng, start_altitude)]  # top 1 constant shift
    method2_drone_pos = [(start_lat, start_lng, start_altitude)]  # top K mean shift
    method3_drone_pos = [(start_lat, start_lng, start_altitude)]  # top K weighted by std
    method4_drone_pos = [(start_lat, start_lng, start_altitude)]  # top K gaussian product
    
    # Initialize momentum vectors for each method (lat, lng, alt)
    momentum1 = np.zeros(3)  # For method 1
    momentum2 = np.zeros(3)  # For method 2
    momentum3 = np.zeros(3)  # For method 3
    momentum4 = np.zeros(3)  # For method 4
    
    # Momentum and smoothing parameters
    beta = 0.9  # Momentum factor (how much previous momentum is retained)
    alpha = 0.05  # Base adjustment strength (reduced from 0.3)
    smoothing_window = 5  # Number of previous positions to consider for smoothing
    
    # Previous positions for smoothing
    prev_positions1 = []
    prev_positions2 = []
    prev_positions3 = []
    prev_positions4 = []
    
    # Top K for methods (hyperparameter)
    K = 5
    
    def apply_momentum_correction(current_pos, target_pos, momentum, prev_positions, alpha, beta):
        """Helper function to apply momentum-based correction"""
        # Calculate raw adjustment vector
        raw_adjustment = target_pos - current_pos
        
        # Scale the adjustment (smaller alpha for smoother movement)
        scaled_adjustment = alpha * raw_adjustment
        
        # Update momentum with decay
        momentum = beta * momentum + (1 - beta) * scaled_adjustment
        
        # Apply the momentum-based adjustment
        new_pos = current_pos + momentum
        
        # Add to previous positions for smoothing
        prev_positions.append(new_pos)
        if len(prev_positions) > smoothing_window:
            prev_positions.pop(0)
        
        # Apply smoothing using exponential moving average
        if len(prev_positions) > 1:
            weights = np.exp(np.linspace(-1, 0, len(prev_positions)))
            weights /= weights.sum()
            smoothed_pos = np.average(prev_positions, axis=0, weights=weights)
            return tuple(smoothed_pos), momentum
        
        return tuple(new_pos), momentum
    
    # Run simulation
    for t in range(duration):
        # Step drones forward
        true_state = true_drone.step()
        drifted_state = drifted_drone.step()
        
        # Store positions
        true_pos = true_state.position
        true_positions.append([true_pos[0], true_pos[1], true_pos[2]])
        true_coords.append((true_state.lat, true_state.lng, true_state.altitude))
        
        drifted_pos = drifted_state.position
        drifted_positions.append([drifted_pos[0], drifted_pos[1], drifted_pos[2]])
        drifted_coords.append((drifted_state.lat, drifted_state.lng, drifted_state.altitude))
        
        # Process corrections
        if t % sample_interval == 0:
            print(f"\nTime step: {t}/{duration}")
            
            # Get image for localization
            if t % 2 == 0:
                true_img = get_static_map(
                    true_state.lat, true_state.lng,
                    calculate_google_zoom(true_state.altitude, true_state.lat),
                    scale=2
                )
            else:
                true_img = get_azure_maps_image(
                    true_state.lat, true_state.lng,
                    calculate_azure_zoom(true_state.altitude, true_state.lat),
                    size=256, scale=2
                )
            
            if true_img is None:
                print(f"Warning: Could not get image at {t}, skipping correction")
                continue
            
            # Find top K matches
            top_coords, similarities, indices = localizer.find_top_k_points(true_img, k=K)
            
            # Current positions
            current_drifted = np.array([drifted_state.lat, drifted_state.lng, drifted_state.altitude])
            current_true = np.array([true_state.lat, true_state.lng, true_state.altitude])
            top_coords_array = np.array(top_coords)
            
            # Method 1: Top-1 with momentum
            top_1_coord = np.array(top_coords[0])
            method1_new_pos, momentum1 = apply_momentum_correction(
                current_drifted, top_1_coord, momentum1, prev_positions1, 
                alpha * similarities[0], beta  # Scale alpha by similarity
            )
            method1_drone_pos.append(method1_new_pos)
            
            # Method 2: Top-K mean with momentum
            top_k_mean = np.mean(top_coords_array, axis=0)
            method2_new_pos, momentum2 = apply_momentum_correction(
                current_drifted, top_k_mean, momentum2, prev_positions2,
                alpha * np.mean(similarities), beta  # Scale alpha by mean similarity
            )
            method2_drone_pos.append(method2_new_pos)
            
            # Method 3: STD-weighted with momentum
            if len(top_coords_array) > 1:
                top_k_std = np.std(top_coords_array, axis=0)
                top_k_std = np.where(top_k_std < 1e-6, 1e-6, top_k_std)
                weight = 1.0 / (top_k_std + 1e-6)
                weight_normalized = weight / np.sum(weight)
                target = 0.7 * top_k_mean + 0.3 * current_true
                confidence = np.mean(similarities)
                
                method3_new_pos, momentum3 = apply_momentum_correction(
                    current_drifted, target, momentum3, prev_positions3,
                    alpha * confidence * np.mean(weight_normalized), beta
                )
            else:
                method3_new_pos, momentum3 = apply_momentum_correction(
                    current_drifted, top_k_mean, momentum3, prev_positions3,
                    alpha * similarities[0], beta
                )
            method3_drone_pos.append(method3_new_pos)
            
            # Method 4: Gaussian product with momentum
            if len(top_coords_array) > 1:
                # Create Gaussians
                drone_cov = np.eye(3) * 0.001
                top_k_cov = np.cov(top_coords_array, rowvar=False)
                if np.linalg.det(top_k_cov) < 1e-10:
                    top_k_cov += np.eye(3) * 0.001
                
                # Calculate combined Gaussian
                precision_drone = np.linalg.inv(drone_cov)
                precision_topk = np.linalg.inv(top_k_cov)
                precision_combined = precision_drone + precision_topk
                cov_combined = np.linalg.inv(precision_combined)
                mean_combined = cov_combined @ (
                    precision_drone @ current_drifted + 
                    precision_topk @ top_k_mean
                )
                
                # Apply momentum correction with uncertainty-based scaling
                uncertainty = np.trace(cov_combined)
                alpha_scaled = alpha * np.exp(-uncertainty)  # Reduce alpha when uncertain
                
                method4_new_pos, momentum4 = apply_momentum_correction(
                    current_drifted, mean_combined, momentum4, prev_positions4,
                    alpha_scaled, beta
                )
            else:
                method4_new_pos, momentum4 = apply_momentum_correction(
                    current_drifted, top_k_mean, momentum4, prev_positions4,
                    alpha * similarities[0], beta
                )
            method4_drone_pos.append(method4_new_pos)
            
            # Calculate and store errors
            error_original = np.linalg.norm(current_drifted - current_true)
            error_method1 = np.linalg.norm(np.array(method1_new_pos) - current_true)
            error_method2 = np.linalg.norm(np.array(method2_new_pos) - current_true)
            error_method3 = np.linalg.norm(np.array(method3_new_pos) - current_true)
            error_method4 = np.linalg.norm(np.array(method4_new_pos) - current_true)
            
            # Store data
            data.append({
                'time': t,
                'error_original': error_original,
                'error_method1': error_method1,
                'error_method2': error_method2,
                'error_method3': error_method3,
                'error_method4': error_method4,
                'similarity_top1': similarities[0] if similarities else 0,
                'similarity_mean': np.mean(similarities) if similarities else 0,
                'momentum1_magnitude': np.linalg.norm(momentum1),
                'momentum2_magnitude': np.linalg.norm(momentum2),
                'momentum3_magnitude': np.linalg.norm(momentum3),
                'momentum4_magnitude': np.linalg.norm(momentum4)
            })
    
    # Convert lists to numpy arrays
    true_positions = np.array(true_positions)
    drifted_positions = np.array(drifted_positions)
    
    # Convert coordinate lists to metric spaces (x, y, z) for visualization 
    method1_positions = lat_lng_alt_to_xyz(method1_drone_pos[1:], start_lat, start_lng)
    method2_positions = lat_lng_alt_to_xyz(method2_drone_pos[1:], start_lat, start_lng)
    method3_positions = lat_lng_alt_to_xyz(method3_drone_pos[1:], start_lat, start_lng)
    method4_positions = lat_lng_alt_to_xyz(method4_drone_pos[1:], start_lat, start_lng)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(data)
    
    # Calculate average improvements
    avg_improvements = {
        'Method 1': (df['error_original'] - df['error_method1']).mean() / df['error_original'].mean() * 100,
        'Method 2': (df['error_original'] - df['error_method2']).mean() / df['error_original'].mean() * 100,
        'Method 3': (df['error_original'] - df['error_method3']).mean() / df['error_original'].mean() * 100,
        'Method 4': (df['error_original'] - df['error_method4']).mean() / df['error_original'].mean() * 100
    }
    
    # Print summary statistics
    print("\n=== Performance Summary ===")
    print(f"Average error (original): {df['error_original'].mean():.6f}")
    for method, improvement in avg_improvements.items():
        print(f"Average improvement {method}: {improvement:.2f}%")
    
    # Save statistics to CSV
    df.to_csv('output/drone_inference/trajectory_stats.csv', index=False)
    
    return df, true_positions, drifted_positions, method1_positions, method2_positions, method3_positions, method4_positions

# Add to the simulate_dual_trajectories function to call these visualizations:
def simulate_and_visualize(start_lat, start_lng, start_altitude, velocity, 
                          model_path, config_path, duration=1000):
    """Run simulation and generate all visualizations"""
    # Run the simulation to get trajectories and error data
    df, true_positions, drifted_positions, method1_positions, method2_positions, method3_positions, method4_positions = \
        simulate_dual_trajectories(
            start_lat=start_lat,
            start_lng=start_lng,
            start_altitude=start_altitude,
            velocity=velocity,
            model_path=model_path,
            config_path=config_path,
            duration=duration
        )
    
    # Generate visualizations
    generate_method_visualizations(
        df, true_positions, drifted_positions,
        method1_positions, method2_positions,
        method3_positions, method4_positions
    )
    
    return df

# You can call this function from your main script:
if __name__ == "__main__":
    # Configuration
    start_lat = 46.4825  # Odesa City
    start_lng = 30.7233
    start_altitude = 100  # meters
    velocity = np.array([0.5, 0.5, 0.1])  # meters/second
    
    model_path = "models/siamese_net_best.pth"  # Your model path
    config_path = "config/train_config.yaml"  # Your config path
    
    # Run simulation and generate visualizations
    df = simulate_and_visualize(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=start_altitude,
        velocity=velocity,
        model_path=model_path,
        config_path=config_path,
        duration=1000
    ) 