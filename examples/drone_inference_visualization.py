import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from pathlib import Path
import pyproj  # Needed for coordinate system info potentially

# Import the updated simulation function
# from examples.drone_inference import simulate_dual_trajectories

# GeoLocalizer might not be needed directly here anymore unless rebuilding db for vis
# from examples.drone_inference import GeoLocalizer
# lat_lng_alt_to_xyz is no longer needed
# from examples.drone_inference import lat_lng_alt_to_xyz

# Functions from src might be needed if re-running parts, but generally data comes from simulate_dual_trajectories
# from src.simulation.drone import DroneFlight
# from src.google_maps import get_static_map, calculate_google_zoom
# from src.azure_maps import get_azure_maps_image, calculate_azure_zoom

from scipy import stats  # Keep if used
import yaml  # Keep if used
import torch  # Keep if used
from PIL import Image  # Keep if used
from torchvision import transforms  # Keep if used
from scipy.spatial import distance  # Keep if used
from typing import Dict, Optional  # Added
import json  # <-- Add json import

# Define functions before they are called if needed, or ensure they are defined at module level.


def ecef_to_enu(ecef_coords, ref_lat, ref_lon, ref_alt):
    """Converts ECEF coordinates to local ENU coordinates relative to a reference point."""
    # Ensure input is numpy array
    ecef_coords = np.asarray(ecef_coords)
    if ecef_coords.ndim == 1:  # Handle single point
        ecef_coords = ecef_coords.reshape(1, -1)
    if ecef_coords.shape[1] != 3:
        raise ValueError("Input ecef_coords must have shape (N, 3)")

    # Transformer for LLA -> ECEF
    transformer_to_ecef = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:4978", always_xy=True
    )
    ref_x, ref_y, ref_z = transformer_to_ecef.transform(ref_lon, ref_lat, ref_alt)
    ref_ecef = np.array([ref_x, ref_y, ref_z])

    # Vectorized difference
    delta_ecef = ecef_coords - ref_ecef

    # Rotation matrix (transposed for ECEF -> ENU)
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

    R_transpose = np.array(
        [
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )

    # Apply rotation
    enu_coords = (R_transpose @ delta_ecef.T).T

    return enu_coords  # Shape (N, 3)


# <<< --- REVISED Helper --- >>>
def get_last_top_k_for_frame(frame_index, top_k_history):
    """
    Finds the most recent top-k data from the simulation step that
    *generated* the state displayed at frame_index.
    """
    if frame_index == 0:  # Initial state, no prior step data
        return None

    # The state shown at frame_index resulted from simulation step (frame_index - 1)
    target_sim_step = frame_index - 1

    last_update_step = -1
    # Find the latest update step *at or before* the target simulation step
    for update_step in top_k_history.keys():
        # Ensure keys are integers
        if (
            isinstance(update_step, int)
            and update_step <= target_sim_step
            and update_step > last_update_step
        ):
            last_update_step = update_step

    if last_update_step != -1:
        return top_k_history.get(last_update_step)
    return None


# <<< --- UPDATED Signature --- >>>
def create_comparison_gif(
    true_positions_enu,  # Renamed for clarity
    simulated_positions_enu,  # Renamed for clarity
    # --- ADDED: Baseline Trajectory ---
    baseline_positions_enu,
    # --- END ADDED ---
    db_positions_enu,  # Renamed for clarity
    output_path,
    # --- ADDED: Pass original ECEF for reference point calculation ---
    true_positions_ecef_orig,
    # --- End Add ---
    title="Drone Trajectory Comparison (ENU)",
    error_data=None,
    top_k_history: Optional[Dict] = None,
    update_interval: int = 1,  # Keep for reference, but not directly used in new helper
    max_frames=800,
    transformer_to_lla: Optional[pyproj.Transformer] = None,
    output_filename: str = "trajectory_comparison_enu.mp4",  # Default output filename
):
    """
    Creates an animated MP4 video comparing true and simulated trajectories (ENU).
    Highlights Top-K database matches.

    Args:
        ... (other args) ...
        output_filename (str): The name of the output video file.
    """
    has_db_points = db_positions_enu is not None and db_positions_enu.shape[0] > 0
    has_top_k_data = top_k_history is not None and len(top_k_history) > 0

    num_steps = min(len(true_positions_enu), len(simulated_positions_enu))
    # Adjust max_frames if necessary, ensure it aligns with trajectory lengths
    num_frames = min(num_steps, max_frames)
    # num_frames should correspond to indices 0 to num_frames-1

    if num_frames <= 1:
        print(f"Warning: Not enough frames ({num_frames}) to generate GIF.")
        return

    true_traj = true_positions_enu[:num_frames]
    sim_traj = simulated_positions_enu[:num_frames]

    # --- Calculate ENU Reference Point ONCE ---
    ref_lat, ref_lon, ref_alt = 46.4825, 30.7233, 100  # Default
    if len(true_positions_ecef_orig) > 0:
        if transformer_to_lla is not None:
            try:
                ref_lon, ref_lat, ref_alt = transformer_to_lla.transform(
                    *true_positions_ecef_orig[0]
                )
            except Exception as e:
                print(
                    f"Warning: Failed to get ENU reference LLA using provided transformer: {e}. Using default."
                )
        else:
            print(
                "Warning: No transformer provided for ENU reference LLA calculation. Using default."
            )
    else:
        print(
            "Warning: No true ECEF positions provided for ENU reference. Using default."
        )

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Determine plot limits
    all_traj_pos = np.vstack([true_traj, sim_traj])
    if has_db_points:
        all_pos = np.vstack([all_traj_pos, db_positions_enu])
    else:
        all_pos = all_traj_pos
    x_min, x_max = np.min(all_pos[:, 0]), np.max(all_pos[:, 0])
    y_min, y_max = np.min(all_pos[:, 1]), np.max(all_pos[:, 1])
    z_min, z_max = np.min(all_pos[:, 2]), np.max(all_pos[:, 2])
    padding_x = max(10, (x_max - x_min) * 0.1)
    padding_y = max(10, (y_max - y_min) * 0.1)
    padding_z = max(10, (z_max - z_min) * 0.1)
    x_min, x_max = (x_min - padding_x, x_max + padding_x)
    y_min, y_max = (y_min - padding_y, y_max + padding_y)
    # --- ADJUSTED: Ensure Z limit starts at or below 0 if ground is visible ---
    # Find the minimum altitude in the trajectories
    min_traj_z = min(np.min(true_traj[:, 2]), np.min(sim_traj[:, 2]))
    if baseline_positions_enu is not None:
        min_traj_z = min(min_traj_z, np.min(baseline_positions_enu[:num_frames, 2]))
    # If database points exist and are lower, consider them too
    if has_db_points:
        min_db_z = np.min(db_positions_enu[:, 2])
        z_min = min(z_min, min_traj_z, min_db_z)
    else:
        z_min = min(z_min, min_traj_z)
    # Ensure z_min doesn't go above 0 unnecessarily, but allow negative if data is below 0
    z_min = min(z_min, 0)
    # --- END ADJUSTED ---
    z_max = z_max + padding_z  # Keep padding for upper limit

    # Plot elements
    (true_line,) = ax.plot([], [], [], "g-", linewidth=2, label="True Trajectory")
    (sim_line,) = ax.plot(
        [], [], [], "b-", linewidth=2, label="Simulated (Corrected)"
    )  # Updated label
    # --- ADDED: Baseline plot elements ---
    (baseline_line,) = ax.plot(
        [], [], [], "r--", linewidth=2, label="Baseline (Uncorrected)"
    )
    baseline_point = ax.scatter(
        [], [], [], color="red", s=100, marker="x", label="Baseline Pos"
    )
    # --- END ADDED ---
    true_point = ax.scatter(
        [], [], [], color="green", s=100, marker="o", label="True Pos"
    )
    # --- ADDED: Restore missing sim_point definition ---
    sim_point = ax.scatter(
        [], [], [], color="blue", s=100, marker="^", label="Simulated Pos"
    )
    # --- END ADDED ---

    db_scatter = None
    if has_db_points:
        db_scatter = ax.scatter(
            db_positions_enu[:, 0],
            db_positions_enu[:, 1],
            db_positions_enu[:, 2],
            color="gray",
            s=15,
            marker=".",
            alpha=0.5,
            label="DB Points",
        )

    error_text = ax.text2D(
        0.02,
        0.90,
        "",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
    )

    # --- ADDED: Top-K scatter plots ---
    top_k_scatters = []
    # Define hot colors (adjust as needed)
    top_k_colors = ["red", "orange", "yellow", "magenta", "cyan"]
    if has_top_k_data:
        for i in range(5):  # Create 5 plot objects
            scatter = ax.scatter(
                [],
                [],
                [],
                color=top_k_colors[i % len(top_k_colors)],
                s=50,
                marker="x",
                label=f"Top-{i+1} Match",
                alpha=0.9,
            )
            top_k_scatters.append(scatter)

    # Store the last valid top-k ENU coordinates to persist between updates
    last_top_k_enu = [np.empty((0, 3))] * 5

    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("ENU East (meters)")
        ax.set_ylabel("ENU North (meters)")
        ax.set_zlabel("ENU Up (meters)")
        ax.set_title(title)
        ax.legend(loc="best", fontsize="small")  # Adjust legend size

        true_line.set_data_3d([], [], [])
        sim_line.set_data_3d([], [], [])  # Corrected variable name
        true_point._offsets3d = ([], [], [])
        sim_point._offsets3d = ([], [], [])
        error_text.set_text("")

        # --- ADDED: Initialize top-k scatters ---
        for scatter in top_k_scatters:
            scatter._offsets3d = ([], [], [])

        # --- UPDATED: Initialize baseline plot consistently ---
        baseline_line.set_data_3d([], [], [])  # Use set_data_3d
        # --- END UPDATED ---

        # --- ADDED: Initialize baseline point ---
        baseline_point._offsets3d = ([], [], [])
        # --- END ADDED ---

        elements = [true_line, sim_line, true_point, sim_point, error_text]
        if db_scatter:
            elements.append(db_scatter)
        elements.extend(top_k_scatters)
        return tuple(elements)

    def update(frame):
        nonlocal last_top_k_enu  # Allow modification of the outer scope variable
        current_index = frame  # frame 0 to num_frames-1

        # Update main trajectory lines and points
        true_line.set_data_3d(
            true_traj[: current_index + 1, 0],
            true_traj[: current_index + 1, 1],
            true_traj[: current_index + 1, 2],
        )
        sim_line.set_data_3d(
            sim_traj[: current_index + 1, 0],
            sim_traj[: current_index + 1, 1],
            sim_traj[: current_index + 1, 2],
        )
        true_point._offsets3d = (
            [true_traj[current_index, 0]],
            [true_traj[current_index, 1]],
            [true_traj[current_index, 2]],
        )
        sim_point._offsets3d = (
            [sim_traj[current_index, 0]],
            [sim_traj[current_index, 1]],
            [sim_traj[current_index, 2]],
        )

        # --- ADDED: Update baseline trajectory ---
        if baseline_positions_enu is not None:
            baseline_line.set_data(
                baseline_positions_enu[: current_index + 1, 0],
                baseline_positions_enu[: current_index + 1, 1],
            )
            baseline_line.set_3d_properties(
                baseline_positions_enu[: current_index + 1, 2]
            )
        # --- END ADDED ---

        # --- ADDED: Update Top-K points ---
        if has_top_k_data:
            # Use revised helper function
            current_top_k_data = get_last_top_k_for_frame(frame, top_k_history)

            if current_top_k_data and "ecef_coords" in current_top_k_data:
                top_k_ecef = np.array(current_top_k_data["ecef_coords"])
                num_found = len(top_k_ecef)

                if num_found > 0:
                    try:
                        # Use the pre-calculated reference point
                        top_k_enu_current = ecef_to_enu(
                            top_k_ecef, ref_lat, ref_lon, ref_alt
                        )
                        # Store the results correctly
                        last_top_k_enu = [
                            (
                                top_k_enu_current[i : i + 1]
                                if i < num_found
                                else np.empty((0, 3))
                            )
                            for i in range(5)
                        ]
                    except Exception as e:
                        # print(f"Frame {frame}: Error converting top-k ECEF to ENU: {e}") # Debug
                        last_top_k_enu = [np.empty((0, 3))] * 5  # Clear on error
                else:  # No points found in this update
                    last_top_k_enu = [np.empty((0, 3))] * 5
            # If no relevant data found by helper, last_top_k_enu remains unchanged (persists)

            # Update scatter plots using the latest valid data
            for i, scatter in enumerate(top_k_scatters):
                coords = last_top_k_enu[i]
                if coords.shape[0] > 0:
                    scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
                else:
                    scatter._offsets3d = ([], [], [])

        # Update error text
        error_info_str = f"Frame: {frame+1}/{num_frames}"
        if (
            error_data is not None
            # --- UPDATED: Check frame index against error_data length ---
            and frame < len(error_data)
            # --- END UPDATED ---
            and "simulated_error" in error_data.columns
        ):
            current_error = error_data["simulated_error"].iloc[frame]
            if isinstance(current_error, (int, float)):
                error_info_str += f"\nSim Error: {current_error:.2f} m"
            else:
                error_info_str += "\nSim Error: N/A"
        else:
            error_info_str += "\nSim Error: No Data"
        error_text.set_text(error_info_str)

        # Rotate view
        ax.view_init(elev=30, azim=(frame % 360))

        elements = [true_line, sim_line, true_point, sim_point, error_text]
        if db_scatter is not None:
            elements.append(db_scatter)
        elements.extend(top_k_scatters)  # Add top-k scatters to returned elements
        return tuple(elements)

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=False, interval=100
    )

    # Ensure output directory exists (assuming output_path is a Path object)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / output_filename  # Use the provided filename

    print(f"Saving animation to {final_output_path}...")
    print("(This may take a while and requires ffmpeg to be installed)")
    try:
        # Use ffmpeg writer for MP4, increase fps for smoother video
        ani.save(str(final_output_path), writer="ffmpeg", fps=20, dpi=150)
        print(f"Video saved successfully.")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Please ensure ffmpeg is installed and accessible in your system's PATH.")
    finally:
        plt.close(fig)


def plot_error_comparison(df, save_path):
    """Plot comparison of original and corrected errors (in meters)."""
    if (
        df is None
        or df.empty
        or "error_original" not in df.columns
        or "error_corrected" not in df.columns
    ):
        print("Insufficient data for error comparison plot.")
        return

    plt.figure(figsize=(12, 6))  # Smaller figure size

    # Line plot
    plt.plot(
        df["time"],
        df["error_original"],
        color="red",
        alpha=0.8,
        linewidth=2,
        label="Original Error",
    )
    plt.plot(
        df["time"],
        df["error_corrected"],
        color="blue",
        alpha=0.8,
        linewidth=2,
        label="Corrected Error (TopK-Momentum)",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Error (meters)")
    plt.title("Error Comparison: Original vs. Corrected (ECEF Distance)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale("log")  # Keep log scale

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_simulation_error(df, save_path):
    """Plots the simulation error over time."""
    if df is None or df.empty:
        print("Error data is empty. Skipping error plot.")
        return

    plt.figure(figsize=(12, 6))
    # --- UPDATED: Plot both simulated and baseline errors ---
    if "simulated_error" in df.columns:
        plt.plot(
            df["time"],
            df["simulated_error"],
            label="Corrected Trajectory Error",
            color="blue",
        )
    else:
        print("Warning: 'simulated_error' column not found in DataFrame.")

    if "baseline_error" in df.columns:
        plt.plot(
            df["time"],
            df["baseline_error"],
            label="Baseline Trajectory Error",
            color="red",
            linestyle="--",
        )
    else:
        # This is expected if baseline wasn't run, so not necessarily a warning
        print(
            "Info: 'baseline_error' column not found in DataFrame. Baseline plot skipped."
        )
    # --- END UPDATED ---

    plt.xlabel("Time Step")
    plt.ylabel("ECEF Position Error (meters)")
    plt.title("Simulation Error (ECEF Distance: True vs Simulated Path)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    # Use log scale if errors vary widely, otherwise linear might be clearer
    mean_error = df["simulated_error"].mean()
    std_error = df["simulated_error"].std()
    # Check for std_error > 0 to avoid division by zero or NaN comparison
    if (
        std_error > 0 and mean_error / std_error < 10
    ):  # Heuristic: Use log if std dev is large relative to mean
        plt.yscale("log")
        plt.title(
            "Simulation Error (ECEF Distance: True vs Simulated Path) - Log Scale"
        )

    plt.tight_layout()
    print(f"Saving error plot to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# <<< --- UPDATED Signature --- >>>
def generate_method_visualizations(
    true_positions_ecef,
    simulated_positions_ecef,
    # --- ADDED: Baseline Trajectory ---
    baseline_positions_ecef,
    # --- END ADDED ---
    db_positions_ecef,
    error_data=None,
    top_k_history: Optional[Dict] = None,  # Added
    update_interval: int = 1,  # Added
    transformer_to_lla: Optional[pyproj.Transformer] = None,
):
    """
    Generates GIF and error plots comparing true path vs simulated path (with corrections).
    Includes Top-K highlighting.
    """
    output_vis_dir = Path("output/drone_inference/visualization")
    output_vis_dir.mkdir(parents=True, exist_ok=True)
    output_main_dir = Path("output/drone_inference")

    print("Generating trajectory GIF...")
    vis_title = "Simulated Drone Path vs True Path (ENU) with Top-K Matches"

    # Calculate ENU reference ONCE
    ref_lat, ref_lon, ref_alt = 46.4825, 30.7233, 100  # Default
    if len(true_positions_ecef) > 0:
        if transformer_to_lla is not None:
            try:
                ref_lon, ref_lat, ref_alt = transformer_to_lla.transform(
                    *true_positions_ecef[0]
                )
            except Exception as e:
                print(
                    f"Warning: Failed to get ENU reference LLA using provided transformer: {e}. Using default."
                )
        else:
            print(
                "Warning: No transformer provided for ENU reference LLA calculation. Using default."
            )
    else:
        print(
            "Warning: No true positions to determine ENU reference. Using default (Odesa)."
        )

    # Convert main trajectories to ENU
    true_positions_enu = ecef_to_enu(true_positions_ecef, ref_lat, ref_lon, ref_alt)
    simulated_positions_enu = ecef_to_enu(
        simulated_positions_ecef, ref_lat, ref_lon, ref_alt
    )

    # --- ADDED: Convert Baseline to ENU ---
    if baseline_positions_ecef is not None and len(baseline_positions_ecef) > 0:
        baseline_positions_enu = ecef_to_enu(
            baseline_positions_ecef, ref_lat, ref_lon, ref_alt
        )
    else:
        print("Warning: No baseline ECEF positions provided for ENU conversion.")
        baseline_positions_enu = None  # Handle potential None value later
    # --- END ADDED ---

    # Convert DB points to ENU
    db_positions_enu = np.empty((0, 3))
    if db_positions_ecef is not None and db_positions_ecef.shape[0] > 0:
        # --- ADDED: Check for NaNs/Infs in DB ECEF before conversion ---
        if np.any(np.isnan(db_positions_ecef)) or np.any(np.isinf(db_positions_ecef)):
            print(
                "ERROR: NaN or Inf values found in db_positions_ecef. Skipping ENU conversion."
            )
        else:
            # --- END ADDED ---
            try:
                db_positions_enu = ecef_to_enu(
                    db_positions_ecef, ref_lat, ref_lon, ref_alt
                )
            except Exception as e:
                print(f"Warning: Could not convert DB points to ENU: {e}")

    # --- ADDED: Sanity checks for NaN/Inf in ENU data ---
    if np.any(np.isnan(true_positions_enu)) or np.any(np.isinf(true_positions_enu)):
        print(
            "ERROR: NaN or Inf values found in true_positions_enu. Aborting GIF generation."
        )
        return
    if np.any(np.isnan(simulated_positions_enu)) or np.any(
        np.isinf(simulated_positions_enu)
    ):
        print(
            "ERROR: NaN or Inf values found in simulated_positions_enu. Aborting GIF generation."
        )
        return
    if baseline_positions_enu is not None and (
        np.any(np.isnan(baseline_positions_enu))
        or np.any(np.isinf(baseline_positions_enu))
    ):
        print(
            "ERROR: NaN or Inf values found in baseline_positions_enu. Aborting GIF generation."
        )
        return
    if db_positions_enu is not None and (
        np.any(np.isnan(db_positions_enu)) or np.any(np.isinf(db_positions_enu))
    ):
        print(
            "ERROR: NaN or Inf values found in db_positions_enu. Aborting GIF generation."
        )
        return
    # --- END ADDED ---

    # Call create_comparison_gif with all necessary data
    create_comparison_gif(
        true_positions_enu=true_positions_enu,  # Pass ENU
        simulated_positions_enu=simulated_positions_enu,  # Pass ENU
        db_positions_enu=(
            db_positions_enu if db_positions_enu.shape[0] > 0 else None
        ),  # Pass ENU
        output_path=output_vis_dir,
        # --- ADDED: Pass original ECEF for internal reference calculation ---
        true_positions_ecef_orig=true_positions_ecef,
        # --- End Add ---
        title=vis_title,
        error_data=error_data,
        top_k_history=top_k_history,
        update_interval=update_interval,  # Pass interval (though helper changed)
        transformer_to_lla=transformer_to_lla,
        # --- ADDED: Pass Baseline ENU data ---
        baseline_positions_enu=baseline_positions_enu,
        # --- END ADDED ---
    )

    print("Generating error plot...")
    if error_data is not None and "simulated_error" in error_data.columns:
        plot_simulation_error(error_data, output_main_dir / "simulation_error_ecef.png")
    else:
        print("Skipping error plot generation: Missing 'simulated_error' data.")

    print("All visualizations completed.")


# =============================================================================
# Main Execution Block (Example)
# =============================================================================
if __name__ == "__main__":
    # Define paths (assuming script is run from workspace root)
    output_dir = Path("output/drone_inference")
    stats_path = output_dir / "trajectory_stats_ecef.csv"
    history_path = output_dir / "top_k_history.json"

    # --- Load Data ---
    print(f"Loading data from {output_dir}...")
    if not stats_path.exists():
        print(f"Error: Statistics file not found at {stats_path}")
        print("Please run drone_inference.py first to generate the data.")
        exit()

    # Load trajectory data (Assuming saved from simulate_dual_trajectories previously)
    # This part needs adjustment based on how you save/load the ECEF trajectories.
    # For now, let's assume they are loaded from somewhere or re-run simulation if needed.
    # Placeholder: Load from CSV if saved, otherwise use dummy data or re-run.
    # Example: Load the *error* data which IS saved by the current sim script.
    try:
        df_stats = pd.read_csv(stats_path)
        print(f"Loaded stats DataFrame: {df_stats.shape}")
    except Exception as e:
        print(f"Error loading stats CSV: {e}")
        df_stats = None

    # Placeholder: You MUST load or generate these ECEF arrays
    # true_positions_ecef = np.load(output_dir / "true_ecef.npy") # Example load
    # simulated_positions_ecef = np.load(output_dir / "corrected_ecef.npy") # Example load
    # db_positions_ecef = np.load(output_dir / "db_ecef.npy") # Example load

    # Create dummy data if loading fails, just to run the visualization code
    # Replace this with actual loading/generation!
    print("\n--- WARNING: Using DUMMY trajectory data for visualization --- ")
    print("--- Please modify the script to load actual ECEF trajectory data ---")
    duration = 100
    start_pos = np.array([4038317.5, 2223962.5, 4603325.0])  # Approx ECEF Odesa
    true_positions_ecef = start_pos + np.cumsum(
        np.random.randn(duration + 1, 3) * 5, axis=0
    )
    simulated_positions_ecef = start_pos + np.cumsum(
        np.random.randn(duration + 1, 3) * 7, axis=0
    )
    db_positions_ecef = start_pos + np.random.randn(50, 3) * 500
    print("-------------------------------------------------------------")

    # Load Top-K history
    top_k_history_loaded = None
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                loaded_json = json.load(f)
                # Convert string keys back to int, and lists back to numpy arrays
                top_k_history_loaded = {}
                for str_step, data in loaded_json.items():
                    step = int(str_step)
                    if "ecef_coords" in data and isinstance(data["ecef_coords"], list):
                        data["ecef_coords"] = [
                            np.array(coord) for coord in data["ecef_coords"]
                        ]
                    top_k_history_loaded[step] = data
            print(f"Loaded Top-K history: {len(top_k_history_loaded)} items")
        except Exception as e:
            print(f"Error loading or processing Top-K history JSON: {e}")
    else:
        print(
            f"Warning: Top-K history file not found at {history_path}. Visualization will not show matches."
        )

    # --- Create Transformer (Needed for ENU conversion) ---
    # You might need to instantiate this or get it from the simulation results if saved.
    try:
        transformer_to_lla = pyproj.Transformer.from_crs(
            "EPSG:4978", "EPSG:4326", always_xy=True
        )
    except Exception as e:
        print(f"Error creating pyproj transformer: {e}")
        transformer_to_lla = None

    # --- Generate Visualizations ---
    if true_positions_ecef is not None and simulated_positions_ecef is not None:
        generate_method_visualizations(
            true_positions_ecef=true_positions_ecef,
            simulated_positions_ecef=simulated_positions_ecef,
            db_positions_ecef=db_positions_ecef,
            error_data=df_stats,  # Pass the loaded error dataframe
            top_k_history=top_k_history_loaded,  # Pass the loaded history
            update_interval=1,  # Example interval, adjust if needed
            transformer_to_lla=transformer_to_lla,  # Pass the transformer
            # --- ADDED: Pass Baseline ECEF data ---
            baseline_positions_ecef=np.array(
                [start_pos]
            ),  # Pass the baseline ECEF data
            # --- END ADDED ---
        )
    else:
        print("Skipping visualization generation due to missing trajectory data.")

    print("\n--- Visualization Script Finished ---")
