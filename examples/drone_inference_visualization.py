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
):
    """
    Creates an animated GIF comparing true and simulated trajectories (ENU).
    Highlights Top-K database matches.
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
        try:
            transformer_to_lla = pyproj.Transformer.from_crs(
                "EPSG:4978", "EPSG:4326", always_xy=True
            )
            # Use the *original* ECEF coordinate for the reference
            ref_lon, ref_lat, ref_alt = transformer_to_lla.transform(
                *true_positions_ecef_orig[0]
            )
            # print(f"Using ENU Reference (LLA): {ref_lat}, {ref_lon}, {ref_alt}") # Debug print
        except Exception as e:
            print(f"Error calculating ENU reference from ECEF: {e}. Using default.")
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
    z_min, z_max = (z_min - padding_z, z_max + padding_z)

    # Plot elements
    (true_line,) = ax.plot([], [], [], "g-", linewidth=2, label="True Trajectory")
    (sim_line,) = ax.plot([], [], [], "b-", linewidth=2, label="Simulated Trajectory")
    true_point = ax.scatter(
        [], [], [], color="green", s=100, marker="o", label="True Pos"
    )
    sim_point = ax.scatter(
        [], [], [], color="blue", s=100, marker="^", label="Simulated Pos"
    )

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
        sim_line.set_data_3d([], [], [])
        true_point._offsets3d = ([], [], [])
        sim_point._offsets3d = ([], [], [])
        error_text.set_text("")

        # --- ADDED: Initialize top-k scatters ---
        for scatter in top_k_scatters:
            scatter._offsets3d = ([], [], [])

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
            and frame < len(error_data)
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
        if db_scatter:
            elements.append(db_scatter)
        elements.extend(top_k_scatters)  # Add top-k scatters to returned elements
        return tuple(elements)

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=False, interval=100
    )
    print(f"Saving animation to {output_path}...")
    try:
        ani.save(str(output_path), writer="pillow", fps=10, dpi=150)
        print(f"GIF saved successfully.")
    except Exception as e:
        print(f"Error saving GIF: {e}")
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
    """Plots the simulation error (true vs simulated ECEF distance) over time."""
    if (
        df is None
        or df.empty
        or "simulated_error" not in df.columns
        or "time" not in df.columns
    ):
        print("Insufficient data for simulation error plot.")
        return

    plt.figure(figsize=(10, 5))

    plt.plot(
        df["time"],
        df["simulated_error"],
        color="purple",
        alpha=0.9,
        linewidth=2,
        label="Simulation Error (True vs Sim)",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Error (meters)")
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
    db_positions_ecef,
    error_data=None,
    top_k_history: Optional[Dict] = None,  # Added
    update_interval: int = 1,  # Added
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
        try:
            transformer_to_lla = pyproj.Transformer.from_crs(
                "EPSG:4978", "EPSG:4326", always_xy=True
            )
            ref_lon, ref_lat, ref_alt = transformer_to_lla.transform(
                *true_positions_ecef[0]
            )
        except Exception as e:
            print(f"Warning: Failed to get ENU reference LLA: {e}. Using default.")
    else:
        print(
            "Warning: No true positions to determine ENU reference. Using default (Odesa)."
        )

    # Convert main trajectories to ENU
    true_positions_enu = ecef_to_enu(true_positions_ecef, ref_lat, ref_lon, ref_alt)
    simulated_positions_enu = ecef_to_enu(
        simulated_positions_ecef, ref_lat, ref_lon, ref_alt
    )

    # Convert DB points to ENU
    db_positions_enu = np.empty((0, 3))
    if db_positions_ecef is not None and db_positions_ecef.shape[0] > 0:
        try:
            db_positions_enu = ecef_to_enu(db_positions_ecef, ref_lat, ref_lon, ref_alt)
        except Exception as e:
            print(f"Warning: Could not convert DB points to ENU: {e}")

    # Call create_comparison_gif with all necessary data
    create_comparison_gif(
        true_positions_enu=true_positions_enu,  # Pass ENU
        simulated_positions_enu=simulated_positions_enu,  # Pass ENU
        db_positions_enu=(
            db_positions_enu if db_positions_enu.shape[0] > 0 else None
        ),  # Pass ENU
        output_path=output_vis_dir / "trajectory_comparison_enu.gif",
        # --- ADDED: Pass original ECEF for internal reference calculation ---
        true_positions_ecef_orig=true_positions_ecef,
        # --- End Add ---
        title=vis_title,
        error_data=error_data,
        top_k_history=top_k_history,
        update_interval=update_interval,  # Pass interval (though helper changed)
    )

    print("Generating error plot...")
    if error_data is not None and "simulated_error" in error_data.columns:
        plot_simulation_error(error_data, output_main_dir / "simulation_error_ecef.png")
    else:
        print("Skipping error plot generation: Missing 'simulated_error' data.")

    print("All visualizations completed.")


def create_method_gif(
    true_positions_ecef,
    drifted_positions_ecef,
    corrected_positions_ecef,
    method_name,
    method_title,
    db_positions_ecef,
    error_data=None,
    max_frames=10000,  # Adjust this to handle more frames
):
    """
    Create an animated GIF showing trajectories, DB points, and top-K matches.
    """
    # Convert ECEF to ENU
    ref_lat, ref_lon, ref_alt = 46.4825, 30.7233, 100  # Reference point (Odesa City)
    true_positions_enu = ecef_to_enu(true_positions_ecef, ref_lat, ref_lon, ref_alt)
    drifted_positions_enu = ecef_to_enu(
        drifted_positions_ecef, ref_lat, ref_lon, ref_alt
    )
    corrected_positions_enu = ecef_to_enu(
        corrected_positions_ecef, ref_lat, ref_lon, ref_alt
    )
    db_positions_enu = ecef_to_enu(db_positions_ecef, ref_lat, ref_lon, ref_alt)

    has_db_points = (
        db_positions_enu.ndim == 2
        and db_positions_enu.shape[0] > 0
        and db_positions_enu.shape[1] == 3
    )
    if not has_db_points:
        print(
            "Warning: No valid database points provided. Skipping database point plotting."
        )

    num_steps = min(
        len(true_positions_enu),
        len(drifted_positions_enu),
        len(corrected_positions_enu),
    )
    num_frames = min(num_steps, max_frames)

    if num_frames <= 1:
        print(
            f"Warning: Not enough frames ({num_frames}) to generate GIF for {method_name}."
        )
        return

    # Use ENU coordinates for plotting
    true_positions = true_positions_enu[:num_frames]
    drift_positions = drifted_positions_enu[:num_frames]
    corrected_positions = corrected_positions_enu[:num_frames]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    all_traj_pos = np.vstack([true_positions, drift_positions, corrected_positions])
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

    x_min, x_max = (
        (x_min - padding_x, x_max + padding_x)
        if x_max > x_min
        else (x_min - 10, x_max + 10)
    )
    y_min, y_max = (
        (y_min - padding_y, y_max + padding_y)
        if y_max > y_min
        else (y_min - 10, y_max + 10)
    )
    z_min, z_max = (
        (z_min - padding_z, z_max + padding_z)
        if z_max > z_min
        else (z_min - 10, z_max + 10)
    )

    (true_line,) = ax.plot([], [], [], "g-", linewidth=2, label="True Trajectory")
    (drift_line,) = ax.plot([], [], [], "r-", linewidth=2, label="Drifted Trajectory")
    (method_line,) = ax.plot(
        [], [], [], "b-", linewidth=2, label="Corrected Trajectory"
    )

    true_point = ax.scatter(
        [], [], [], color="green", s=100, marker="o", label="True Pos"
    )
    drift_point = ax.scatter(
        [], [], [], color="red", s=100, marker="x", label="Drifted Pos"
    )
    method_point = ax.scatter(
        [], [], [], color="blue", s=100, marker="^", label="Corrected Pos"
    )

    db_scatter_default = None
    db_scatter_highlight = None
    if has_db_points:
        db_scatter_default = ax.scatter(
            [], [], [], color="gray", s=15, marker=".", alpha=0.5, label="DB Points"
        )
        db_scatter_highlight = ax.scatter(
            [],
            [],
            [],
            color="orangered",
            s=30,
            marker="o",
            alpha=0.8,
            label="Top-K Matches",
        )

    error_text = ax.text2D(
        0.02,
        0.90,
        "",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
    )

    def init():
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("ENU East (meters)")
        ax.set_ylabel("ENU North (meters)")
        ax.set_zlabel("ENU Up (meters)")
        ax.set_title(f"Drone Trajectory Correction (ENU)\n{method_title}")
        ax.legend(loc="best")

        true_line.set_data_3d([], [], [])
        drift_line.set_data_3d([], [], [])
        method_line.set_data_3d([], [], [])
        true_point._offsets3d = ([], [], [])
        drift_point._offsets3d = ([], [], [])
        method_point._offsets3d = ([], [], [])

        if (
            has_db_points
            and db_scatter_default is not None
            and db_scatter_highlight is not None
        ):
            db_scatter_default._offsets3d = (
                db_positions_enu[:, 0],
                db_positions_enu[:, 1],
                db_positions_enu[:, 2],
            )
            db_scatter_highlight._offsets3d = ([], [], [])

        error_text.set_text("")
        elements = [
            true_line,
            drift_line,
            method_line,
            true_point,
            drift_point,
            method_point,
            error_text,
        ]
        if (
            has_db_points
            and db_scatter_default is not None
            and db_scatter_highlight is not None
        ):
            elements.extend([db_scatter_default, db_scatter_highlight])
        return tuple(elements)

    def update(frame):
        current_index = frame

        true_line.set_data_3d(
            true_positions[: current_index + 1, 0],
            true_positions[: current_index + 1, 1],
            true_positions[: current_index + 1, 2],
        )
        true_point._offsets3d = (
            [true_positions[current_index, 0]],
            [true_positions[current_index, 1]],
            [true_positions[current_index, 2]],
        )

        drift_line.set_data_3d(
            drift_positions[: current_index + 1, 0],
            drift_positions[: current_index + 1, 1],
            drift_positions[: current_index + 1, 2],
        )
        drift_point._offsets3d = (
            [drift_positions[current_index, 0]],
            [drift_positions[current_index, 1]],
            [drift_positions[current_index, 2]],
        )

        method_line.set_data_3d(
            corrected_positions[: current_index + 1, 0],
            corrected_positions[: current_index + 1, 1],
            corrected_positions[: current_index + 1, 2],
        )
        method_point._offsets3d = (
            [corrected_positions[current_index, 0]],
            [corrected_positions[current_index, 1]],
            [corrected_positions[current_index, 2]],
        )

        highlight_indices = None
        if (
            has_db_points
            and error_data is not None
            and "top_k_indices" in error_data.columns
            and frame < len(error_data)
        ):
            indices_val = error_data["top_k_indices"].iloc[frame]
            if isinstance(indices_val, list) and len(indices_val) > 0:
                highlight_indices = np.array(indices_val, dtype=int)

        if (
            has_db_points
            and db_scatter_default is not None
            and db_scatter_highlight is not None
        ):
            if highlight_indices is not None and len(highlight_indices) > 0:
                is_highlighted = np.isin(
                    np.arange(len(db_positions_enu)), highlight_indices
                )
                highlight_points = db_positions_enu[is_highlighted]
                db_scatter_highlight._offsets3d = (
                    highlight_points[:, 0],
                    highlight_points[:, 1],
                    highlight_points[:, 2],
                )
                default_points = db_positions_enu[~is_highlighted]
                db_scatter_default._offsets3d = (
                    default_points[:, 0],
                    default_points[:, 1],
                    default_points[:, 2],
                )
            else:
                db_scatter_default._offsets3d = (
                    db_positions_enu[:, 0],
                    db_positions_enu[:, 1],
                    db_positions_enu[:, 2],
                )
                db_scatter_highlight._offsets3d = ([], [], [])

        error_info_str = f"Frame: {frame+1}/{num_frames}"
        if error_data is not None and not error_data.empty and frame < len(error_data):
            orig_error_col = "error_original"
            method_error_col = "error_corrected"

            if (
                orig_error_col in error_data.columns
                and method_error_col in error_data.columns
            ):
                orig_error = error_data[orig_error_col].iloc[frame]
                method_error = error_data[method_error_col].iloc[frame]

                improvement_str = "N/A"
                if isinstance(orig_error, (int, float)) and isinstance(
                    method_error, (int, float)
                ):
                    if orig_error > 1e-9:
                        improvement = (orig_error - method_error) / orig_error * 100
                        improvement_str = f"{improvement:.1f}%"
                    else:
                        improvement_str = "Inf" if method_error < orig_error else "0.0%"
                    error_info_str += (
                        f"\nOriginal Error: {orig_error:.2f} m"
                        f"\nCorrected Error: {method_error:.2f} m"
                        f"\nImprovement: {improvement_str}"
                    )
                else:
                    error_info_str += "\nError values not numeric."
            else:
                error_info_str += (
                    f"\nError columns missing ({orig_error_col} or {method_error_col})"
                )

        error_text.set_text(error_info_str)

        ax.view_init(elev=30, azim=(frame % 360))

        elements = [
            true_line,
            drift_line,
            method_line,
            true_point,
            drift_point,
            method_point,
            error_text,
        ]
        if (
            has_db_points
            and db_scatter_default is not None
            and db_scatter_highlight is not None
        ):
            elements.extend([db_scatter_default, db_scatter_highlight])
        return tuple(elements)

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=False, interval=100
    )
    output_vis_dir = Path("output/drone_inference/methods")
    output_path = output_vis_dir / f"{method_name}_trajectory_enu.gif"
    print(f"Saving animation to {output_path}...")
    try:
        ani.save(output_path, writer="pillow", fps=10, dpi=100)
        print(f"GIF saved successfully")
    except Exception as e:
        print(f"Error saving GIF: {e}")
    plt.close(fig)
