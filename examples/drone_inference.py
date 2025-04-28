import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import distance
import time
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv, dotenv_values
import torch
from PIL import Image
from torchvision import transforms

# Use pyproj for accurate coordinate transformations
import pyproj
from src.google_maps import get_static_map, calculate_google_zoom
from src.azure_maps import get_azure_maps_image, calculate_azure_zoom
from src.simulation.drone import DroneFlight
from src.models.siamese_net import SiameseNet
import pandas as pd
from src.elevation import get_google_elevation

# Import visualization functions (will be updated next)
from examples.drone_inference_visualization import (
    generate_method_visualizations,
    plot_simulation_error,
)

# =============================================================================
# Helper Functions (like in drone_simulation.py)
# =============================================================================


def enu_to_ecef_velocity(lat_deg, lon_deg, velocity_enu) -> np.ndarray:
    """Transforms a velocity vector from local ENU to ECEF frame."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

    # Rotation matrix from ENU to ECEF
    R = np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0, cos_lat, sin_lat],
        ]
    )

    velocity_ecef = R @ np.asarray(velocity_enu)
    return velocity_ecef


# Transformer for LLA -> ECEF coordinate conversion (use class member in GeoLocalizer)
# _transformer_to_ecef = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)


# =============================================================================
# GeoLocalizer Class Modifications
# =============================================================================
class GeoLocalizer:
    def __init__(self, model_path, config_path, secrets_path=".env"):
        # Load config with explicit encoding handling
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except UnicodeDecodeError:
            print(
                f"Warning: Failed to load {config_path} with UTF-8 encoding. Trying latin-1."
            )
            try:
                with open(config_path, "r", encoding="latin-1") as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                print(
                    f"Error loading config file {config_path} with fallback encoding: {e}"
                )
                raise  # Re-raise the exception if fallback also fails
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            raise  # Re-raise other potential errors

        # Load environment variables
        load_dotenv(secrets_path)
        self.secrets = dotenv_values(secrets_path)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model (assumed SiameseNet)
        model_config = self.config.get("model", {})
        backbone_name = model_config.get("backbone", "shufflenet_v2_x1_0")
        pretrained = model_config.get("pretrained", True)
        embedding_dim = model_config.get("embedding_dim", 128)

        print(
            f"Initializing model with {backbone_name}, embedding_dim: {embedding_dim}"
        )
        self.model = SiameseNet(
            backbone_name=backbone_name,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
        )
        # Load checkpoint
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print("Loading model from 'model_state_dict' key")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("Loading model directly from checkpoint")
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Set up image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Transformer for coordinate conversions
        self._transformer_to_ecef = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:4978", always_xy=True
        )

        # Database containers
        self.db_embeddings = None
        self.db_coords_ecef = None  # Store ECEF coordinates (x, y, z)
        self.db_images = None

    def extract_embedding(self, img_tensor):
        """Extract embedding using the model's forward_one method."""
        with torch.no_grad():
            if hasattr(self.model, "forward_one"):
                embedding = self.model.forward_one(img_tensor)
                return embedding.cpu().numpy()
            else:
                try:
                    if hasattr(self.model, "get_embedding"):
                        return self.model.get_embedding(img_tensor).cpu().numpy()
                    elif hasattr(self.model, "embedding"):
                        return self.model.embedding(img_tensor).cpu().numpy()
                    elif hasattr(self.model, "encode"):
                        return self.model.encode(img_tensor).cpu().numpy()
                    else:
                        return self.model(img_tensor).cpu().numpy()
                except Exception as e:
                    print(f"Error extracting embedding: {e}")
                    # Return a default shape or None if needed
                    embedding_dim = getattr(
                        self.model, "embedding_dim", 128
                    )  # Get dim if possible
                    return np.zeros((1, embedding_dim))

    def build_database(
        self,
        lat_center,
        lng_center,
        grid_size=9,
        step_meters=50,
        height_range=30,
        base_height=100,
    ):
        """Build a database of embeddings and coordinates (LLA & ECEF)."""
        print(f"Building database around ({lat_center}, {lng_center})...")

        lat_deg, lng_deg = self._meters_to_degrees(step_meters, lat_center)
        lat_range = np.linspace(
            lat_center - lat_deg * grid_size / 2,
            lat_center + lat_deg * grid_size / 2,
            grid_size,
        )
        lng_range = np.linspace(
            lng_center - lng_deg * grid_size / 2,
            lng_center + lng_deg * grid_size / 2,
            grid_size,
        )
        height_values = np.array(
            [base_height - height_range, base_height, base_height + height_range]
        )

        coordinates = []
        for height in height_values:
            for lng in lng_range:
                for lat in lat_range:
                    coordinates.append((lat, lng, height))

        embeddings = []
        valid_coords_ecef = []  # Store ECEF coordinates
        images = []

        print(f"Fetching {len(coordinates)} images for database...")
        for i, (lat, lng, height) in enumerate(coordinates):
            # Use print with flush to see progress immediately
            print(
                f"\rFetching database image {i+1}/{len(coordinates)}...",
                end="",
                flush=True,
            )
            # Alternate between providers if needed (here we call get_static_map)
            img = get_static_map(lat, lng, calculate_google_zoom(height, lat))
            if img is not None:
                img_tensor = (
                    self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
                )
                try:
                    embedding = self.extract_embedding(img_tensor)
                    # Check if embedding is valid before proceeding
                    if (
                        embedding is not None
                        and embedding.ndim == 2
                        and embedding.shape[0] > 0
                        and embedding.shape[1] > 0
                    ):
                        # Convert LLA to ECEF
                        x, y, z = self._transformer_to_ecef.transform(lng, lat, height)
                        ecef_coord = np.array([x, y, z])

                        embeddings.append(embedding[0])  # Assuming batch size 1
                        valid_coords_ecef.append(ecef_coord)
                        images.append(
                            img
                        )  # Append image only if embedding and coords are successful
                    else:
                        # This else belongs to the inner embedding check
                        print(
                            f"\nWarning: Got invalid embedding for point {i+1} ({lat:.4f}, {lng:.4f}). Skipping."
                        )
                except Exception as e:
                    # This except belongs to the try block
                    print(f"\nError processing point {i+1} ({lat:.4f}, {lng:.4f}): {e}")
            else:
                # This else belongs to the outer "if img is not None"
                print(
                    f"\nWarning: Failed to fetch image for point {i+1} ({lat:.4f}, {lng:.4f}). Skipping."
                )

            time.sleep(0.2)  # Reduced sleep slightly

        print(
            f"\nDatabase built with {len(valid_coords_ecef)} points"
        )  # Newline after progress indicator

        self.db_embeddings = np.array(embeddings)
        self.db_coords_ecef = np.array(valid_coords_ecef)
        self.db_images = images
        # No return needed, data stored in self

    def find_closest_point(self, query_img):
        """Return the single best match (LLA, ECEF, similarity, index)."""
        if self.db_embeddings is None or len(self.db_embeddings) == 0:
            raise ValueError(
                "Database not built or is empty. Call build_database first."
            )

        query_tensor = (
            self.transform(query_img.convert("RGB")).unsqueeze(0).to(self.device)
        )
        try:
            query_embedding = self.extract_embedding(query_tensor)
            if query_embedding is None or query_embedding.shape[0] == 0:
                raise ValueError("Empty query embedding")
            query_embedding = query_embedding[0]  # Assuming batch size 1
        except Exception as e:
            print(f"Error extracting query embedding: {e}")
            idx = np.random.randint(0, len(self.db_coords_ecef))
            return (
                None,
                self.db_coords_ecef[idx],
                0.0,
                idx,
            )  # Return random point

        distances = distance.cdist([query_embedding], self.db_embeddings, "cosine")[0]
        best_idx = np.argmin(distances)
        similarity = 1 - distances[best_idx]
        return (
            None,
            self.db_coords_ecef[best_idx],
            similarity,
            best_idx,
        )

    def find_top_k_points(self, query_img, k=5):
        """Find the top K closest points (LLA coords, ECEF coords, similarities, indices)."""
        if self.db_embeddings is None or len(self.db_embeddings) == 0:
            raise ValueError(
                "Database not built or is empty. Call build_database first."
            )

        # Process query image
        query_tensor = (
            self.transform(query_img.convert("RGB")).unsqueeze(0).to(self.device)
        )

        # Get query embedding
        try:
            query_embedding = self.extract_embedding(query_tensor)
            if query_embedding is None or query_embedding.shape[0] == 0:
                raise ValueError("Empty query embedding")
            query_embedding = query_embedding[0]  # Assuming batch size 1
        except Exception as e:
            print(f"Error extracting query embedding: {e}")
            # Return random points if failed
            num_available = len(self.db_coords_ecef)
            actual_k = min(k, num_available)
            indices = np.random.choice(num_available, actual_k, replace=False)
            return (
                None,
                [self.db_coords_ecef[i] for i in indices],  # Also return random ECEF
                [0.0] * actual_k,
                indices,
            )

        # Compute distances
        distances = distance.cdist([query_embedding], self.db_embeddings, "cosine")[0]

        # Find top k points
        # Ensure k is not larger than the database size
        num_available = len(self.db_coords_ecef)
        actual_k = min(k, num_available)
        if actual_k < k:
            print(
                f"Warning: Requested k={k}, but database only has {num_available} points. Returning {actual_k} points."
            )

        top_indices = np.argsort(distances)[:actual_k]

        # Retrieve corresponding ECEF coordinates
        top_coords_ecef = [self.db_coords_ecef[i] for i in top_indices]

        similarities = [1 - distances[i] for i in top_indices]

        print(f"Found {len(top_indices)} matches with similarities: {similarities}")

        return None, top_coords_ecef, similarities, top_indices

    def _meters_to_degrees(self, meters, latitude):
        """Approximate conversion of meters to degrees."""
        lat_deg = meters / 111111
        lng_deg = meters / (111111 * np.cos(np.radians(latitude)))
        return lat_deg, lng_deg


# =============================================================================
# Simplified Simulation Function
# =============================================================================
def simulate_dual_trajectories(
    start_lat,
    start_lng,
    start_altitude,
    velocity_enu,
    model_path,
    config_path,
    drift_std=0.1,
    drift_velocity_enu=np.array([0.1, -0.05, 0.02]),
    duration=50,
    sample_interval=1,
):
    """
    Simulates true/drifted trajectories, applies Top-K Mean Momentum correction,
    and returns ECEF coordinates, error metrics, and database info.

    Noise (drift_std) is applied to the drifted trajectory's velocity in the ENU frame at each step.

    Returns:
        tuple: (df, true_ecef, drifted_ecef, corrected_ecef, db_ecef, transformer)
    """
    os.makedirs("output/drone_inference", exist_ok=True)

    # --- Ensure correct arguments are passed to GeoLocalizer ---
    # The first argument should be the model path (.pth)
    # The second argument should be the config path (.yaml)
    print(f"Initializing GeoLocalizer with model: {model_path}, config: {config_path}")
    localizer = GeoLocalizer(model_path=model_path, config_path=config_path)
    # ----------------------------------------------------------

    # Build database
    grid_size_db = localizer.config.get("database", {}).get("grid_size", 15)
    # ... (load other db params) ...
    localizer.build_database(
        lat_center=start_lat,
        lng_center=start_lng,
        grid_size=grid_size_db,
        base_height=start_altitude,  # Use start_altitude as base_height default
        height_range=localizer.config.get("database", {}).get("height_range", 30),
        step_meters=localizer.config.get("database", {}).get("step_meters", 50),
    )

    db_positions_ecef = localizer.db_coords_ecef
    if db_positions_ecef is None or len(db_positions_ecef) == 0:
        print("Warning: Database ECEF coordinates are empty.")
        db_positions_ecef = np.empty((0, 3))

    transformer_to_ecef = localizer._transformer_to_ecef
    start_x, start_y, start_z = transformer_to_ecef.transform(
        start_lng, start_lat, start_altitude
    )
    initial_pos_ecef = np.array([start_x, start_y, start_z])

    # Calculate intended ECEF velocities (deterministic part)
    true_velocity_ecef = enu_to_ecef_velocity(start_lat, start_lng, velocity_enu)
    intended_drifted_velocity_ecef = enu_to_ecef_velocity(
        start_lat, start_lng, velocity_enu + drift_velocity_enu
    )

    # --- Initialize DroneFlight using correct parameters (NO noise_std) ---
    print("Initializing drone objects...")
    try:
        true_drone = DroneFlight(
            start_lat=start_lat,
            start_lng=start_lng,
            start_altitude=start_altitude,
            velocity_ecef=true_velocity_ecef,
        )
        drifted_drone = DroneFlight(
            start_lat=start_lat,
            start_lng=start_lng,
            start_altitude=start_altitude,
            velocity_ecef=intended_drifted_velocity_ecef,  # Store the base drift velocity
        )
    except TypeError as e:
        print(f"Error initializing DroneFlight: {e}")
        print(
            "Please ensure DroneFlight.__init__ accepts 'start_lat', 'start_lng', 'start_altitude', 'velocity_ecef' keyword arguments."
        )
        raise  # Re-raise the error after printing context
    # --------------------------------------------------

    # Data collectors
    data = []
    true_positions_ecef = [initial_pos_ecef.copy()]
    drifted_positions_ecef = [initial_pos_ecef.copy()]
    corrected_positions_ecef = [initial_pos_ecef.copy()]  # Renamed from method2

    # Momentum vector (only one needed now)
    momentum = np.zeros(3)  # Renamed from momentum2

    correction_params = localizer.config.get("correction", {})
    beta = correction_params.get("momentum_beta", 0.9)
    alpha = correction_params.get("adjustment_alpha", 0.05)
    smoothing_window = correction_params.get("smoothing_window", 5)
    K = correction_params.get("top_k", 5)

    prev_positions = []  # Renamed from prev_positions2

    def apply_momentum_correction_ecef(
        current_pos_ecef, target_pos_ecef, momentum, prev_positions, alpha, beta
    ):
        # (This helper function remains the same as before)
        raw_adjustment = np.asarray(target_pos_ecef) - np.asarray(current_pos_ecef)
        scaled_adjustment = alpha * raw_adjustment
        momentum = beta * momentum + (1 - beta) * scaled_adjustment
        new_pos = current_pos_ecef + momentum

        prev_positions.append(new_pos)
        if len(prev_positions) > smoothing_window:
            prev_positions.pop(0)
        if len(prev_positions) > 1:
            weights = np.exp(
                np.linspace(
                    -1.0 * (smoothing_window - 1) / smoothing_window,
                    0,
                    len(prev_positions),
                )
            )
            weights /= weights.sum()
            smoothed_pos = np.average(prev_positions, axis=0, weights=weights)
            return smoothed_pos, momentum
        return new_pos, momentum

    last_query_img_failed = False
    print(f"Running simulation for {duration} steps...")
    for t in range(duration):
        # --- Step 1: Move the true drone (deterministic) ---
        true_drone.step()

        # --- Step 2: Calculate noisy velocity for the drifted drone ---
        # Get current drifted state for noise calculation reference
        drift_lat, drift_lng, _, intended_vel_ecef = drifted_drone.get_state()

        # Convert intended ECEF velocity to ENU at current drifted location
        intended_vel_enu = DroneFlight._ecef_to_enu_velocity(
            drift_lat, drift_lng, intended_vel_ecef
        )

        # Add Gaussian noise in the ENU frame
        noise_enu = np.random.normal(0, drift_std, 3)
        noisy_vel_enu = intended_vel_enu + noise_enu

        # Convert noisy ENU velocity back to ECEF
        noisy_vel_ecef_step = DroneFlight._enu_to_ecef_velocity(
            drift_lat, drift_lng, noisy_vel_enu
        )

        # --- Step 3: Move the drifted drone using the noisy velocity ---
        drifted_drone.step(velocity_step_ecef=noisy_vel_ecef_step)

        # --- Step 4: Record positions ---
        true_pos_ecef = true_drone.position
        drifted_pos_ecef = drifted_drone.position
        true_positions_ecef.append(true_pos_ecef.copy())
        drifted_positions_ecef.append(drifted_pos_ecef.copy())

        # --- Step 5: Perform Localization and Correction (using drifted LLA) ---
        drift_lat_curr, drift_lng_curr, drift_alt_curr, _ = drifted_drone.get_state()

        current_data_step = {
            "time": t,
            "error_original": np.linalg.norm(drifted_pos_ecef - true_pos_ecef),
        }
        top_k_indices_current = []

        if t % sample_interval == 0 and not last_query_img_failed:
            query_img = None
            try:
                # Use current drifted LLA for fetching images
                zoom_level_g = calculate_google_zoom(drift_alt_curr, drift_lat_curr)
                zoom_level_a = calculate_azure_zoom(drift_alt_curr, drift_lat_curr)

                if t % (2 * sample_interval) == 0:
                    query_img = get_static_map(
                        drift_lat_curr, drift_lng_curr, zoom_level_g, scale=2
                    )
                else:
                    query_img = get_azure_maps_image(
                        drift_lat_curr, drift_lng_curr, zoom_level_a, size=256, scale=2
                    )

                if query_img is None:
                    print(f"\nWarning: Failed to fetch query image at step {t}.")
                    last_query_img_failed = True
                    # Predict next corrected position based on momentum only
                    corrected_positions_ecef.append(
                        corrected_positions_ecef[-1] + momentum
                    )
                else:
                    last_query_img_failed = False
                    print(f"\rStep {t}/{duration}: Localizing...", end="", flush=True)
                    try:
                        # Find top K matches (returns LLA, ECEF, similarities, indices)
                        _, top_coords_ecef, similarities, indices = (
                            localizer.find_top_k_points(query_img, k=K)
                        )
                        top_k_indices_current = indices.tolist()

                        current_drifted_ecef = drifted_pos_ecef
                        top_coords_ecef_array = np.array(top_coords_ecef)

                        # --- Top-K Mean Shift (ECEF) --- ONLY METHOD NOW ---
                        top_k_mean_ecef = np.mean(top_coords_ecef_array, axis=0)
                        corrected_new_pos, momentum = apply_momentum_correction_ecef(
                            current_drifted_ecef,
                            top_k_mean_ecef,
                            momentum,
                            prev_positions,
                            alpha
                            * np.mean(similarities),  # Scale alpha by mean similarity
                            beta,
                        )
                        corrected_positions_ecef.append(corrected_new_pos)
                        # -----------------------------------------------------

                        # Store similarities and momentum
                        current_data_step["similarity_top1"] = (
                            similarities[0] if similarities else 0
                        )
                        current_data_step["similarity_mean"] = (
                            np.mean(similarities) if similarities else 0
                        )
                        current_data_step["momentum_magnitude"] = np.linalg.norm(
                            momentum
                        )  # Renamed
                    except Exception as e:
                        print(
                            f"\nError during localization/correction at step {t}: {e}"
                        )
                        # Predict next corrected position based on momentum only
                        corrected_positions_ecef.append(
                            corrected_positions_ecef[-1] + momentum
                        )
            except Exception as e:
                print(f"\nError fetching image at step {t}: {e}")
                last_query_img_failed = True
                # Predict next corrected position based on momentum only
                corrected_positions_ecef.append(corrected_positions_ecef[-1] + momentum)
        else:  # Not a sample interval or image failed previously
            # Predict next corrected position based on momentum only
            corrected_positions_ecef.append(corrected_positions_ecef[-1] + momentum)

        # --- Step 6: Calculate and store corrected error ---
        current_true_ecef = true_positions_ecef[-1]  # Get the latest true position
        current_data_step["error_corrected"] = np.linalg.norm(
            corrected_positions_ecef[-1] - current_true_ecef
        )

        # Store top_k_indices
        current_data_step["top_k_indices"] = (
            top_k_indices_current if top_k_indices_current else None
        )
        data.append(current_data_step)

    print("\nSimulation finished.")

    # Convert lists to numpy arrays
    true_positions_ecef_np = np.array(true_positions_ecef)
    drifted_positions_ecef_np = np.array(drifted_positions_ecef)
    corrected_positions_ecef_np = np.array(corrected_positions_ecef)  # Renamed

    # Pad/Trim arrays
    target_len = duration + 1

    def pad_or_trim(arr, length):
        # (This helper function remains the same)
        if len(arr) < length:
            padding = np.repeat(arr[-1:], length - len(arr), axis=0)
            return np.vstack((arr, padding))
        elif len(arr) > length:
            return arr[:length]
        return arr

    true_positions_ecef_np = pad_or_trim(true_positions_ecef_np, target_len)
    drifted_positions_ecef_np = pad_or_trim(drifted_positions_ecef_np, target_len)
    corrected_positions_ecef_np = pad_or_trim(
        corrected_positions_ecef_np, target_len
    )  # Renamed

    df = pd.DataFrame(data)

    # Print summary statistics
    if (
        not df.empty
        and "error_original" in df.columns
        and "error_corrected" in df.columns
    ):
        avg_orig_error = df["error_original"].mean()
        avg_corr_error = df["error_corrected"].mean()
        improvement = (
            (avg_orig_error - avg_corr_error) / avg_orig_error * 100
            if avg_orig_error > 1e-9
            else 0
        )
        print("\n=== Performance Summary (ECEF Errors - meters) ===")
        print(f"Average error (original): {avg_orig_error:.2f}")
        print(
            f"Corrected (TopK-Mean-Momentum): Avg Error={avg_corr_error:.2f}, Improvement={improvement:.1f}%"
        )
    else:
        print(
            "\nWarning: DataFrame missing error columns, cannot calculate statistics."
        )

    df.to_csv("output/drone_inference/trajectory_stats_ecef.csv", index=False)

    # --- Return ECEF data, db points, and transformer ---
    return (
        df,
        true_positions_ecef_np,
        drifted_positions_ecef_np,
        corrected_positions_ecef_np,  # Single corrected trajectory
        db_positions_ecef,
        transformer_to_ecef,
    )


# =============================================================================
# Main entry point (Example Usage)
# =============================================================================
if __name__ == "__main__":
    # Configuration parameters
    start_lat = 46.4825  # Odesa City latitude
    start_lng = 30.7233  # Odesa City longitude
    start_height = 100  # Local height above ground (meters)

    # Base velocity in local ENU frame (m/step)
    velocity_enu = np.array([1.0, 1.0, 0.1])

    # Noise parameters (standard deviation in ENU frame m/step)
    drift_std_enu = 0.2  # Noise applied each step

    # Simulation parameters
    duration = 500  # Number of steps
    update_interval = 5  # Run drone.update() every N steps

    # Paths
    model_path = Path("models/siamese_net_best.pth")
    config_path = Path("config/train_config.yaml")
    output_dir = Path("output/drone_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Input Validation ---
    if not model_path.is_file():
        print(f"Error: Model file not found at {model_path}")
        exit()
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        exit()

    # --- Initialize Drone ---
    print("--- Initializing Drone ---")
    try:
        drone = DroneFlight(
            model_path=str(model_path),
            config_path=str(config_path),
            start_lat=start_lat,
            start_lng=start_lng,
            start_height=start_height,
        )
    except Exception as e:
        print(f"Error initializing DroneFlight: {e}")
        exit()
    print("--- Drone Initialized ---")

    # --- Initialize True Path Simulation ---
    # Need a separate way to track the ideal path without noise/correction
    # Re-use drone's coordinate transformers and helpers for consistency
    print("--- Initializing True Path ---")
    true_ground_elev = get_google_elevation(start_lat, start_lng) or 0
    true_start_alt = true_ground_elev + start_height
    true_start_ecef = drone._transformer_to_ecef.transform(
        start_lng, start_lat, true_start_alt
    )
    true_position_ecef = np.array(true_start_ecef)
    true_positions_history = [true_position_ecef.copy()]
    print(f"True Start ECEF: {true_position_ecef}")
    print("--- True Path Initialized ---")

    # --- Simulation Loop ---
    print(f"--- Running Simulation ({duration} steps) ---")
    simulation_errors = []  # Store error between true and simulated path
    # --- ADDED: Store top K results per update step ---
    top_k_results_history = {}  # Dict: {step_index: results_dict}
    image_fetch_status = {}  # Dict: {step_index: bool (True if fetch succeeded)}

    for t in range(duration):
        print(f"\rStep {t+1}/{duration}...", end="", flush=True)

        # 1. Calculate True Path step (deterministic)
        # Get LLA *before* updating position for image fetching
        true_lat, true_lng, true_alt = drone._transformer_to_lla.transform(
            *true_position_ecef
        )
        true_velocity_ecef_step = drone._enu_to_ecef_velocity(
            true_lat, true_lng, velocity_enu
        )
        true_position_ecef += true_velocity_ecef_step
        true_positions_history.append(true_position_ecef.copy())

        # 2. Fetch Image for Update (if on interval) using PREVIOUS step's true LLA
        image_for_update = None
        fetch_succeeded = False
        if (t + 1) % update_interval == 0:
            print(
                f"\n - Fetching image for update at step {t+1} using true coords ({true_lat:.4f}, {true_lng:.4f}, alt {true_alt:.1f}) - "
            )
            try:
                # Decide which map provider to use (logic moved from drone.update)
                use_azure_map = (t // update_interval) % 2 != 0
                if use_azure_map:
                    zoom_level_a = calculate_azure_zoom(true_alt, true_lat)
                    image_for_update = get_azure_maps_image(
                        true_lat, true_lng, zoom_level_a, size=256
                    )
                    print("   (Using Azure Maps)")
                else:
                    zoom_level_g = calculate_google_zoom(true_alt, true_lat)
                    image_for_update = get_static_map(true_lat, true_lng, zoom_level_g)
                    print("   (Using Google Maps)")

                if image_for_update is None:
                    print("   Warning: Failed to fetch map image.")
                    fetch_succeeded = False
                else:
                    fetch_succeeded = True
                    print("   Image fetched successfully.")

            except Exception as e:
                print(f"   Error fetching map image: {e}")
                fetch_succeeded = False
                image_for_update = None

            image_fetch_status[t] = fetch_succeeded

        # 3. Calculate Noisy Velocity for Drone Step
        noise_enu = np.random.normal(0, drift_std_enu, 3)
        current_step_velocity_enu = velocity_enu + noise_enu

        # 4. Step the Drone (apply noisy movement)
        drone.step(current_step_velocity_enu)

        # 5. Update Drone State (if on interval)
        if (t + 1) % update_interval == 0:
            print(f" - Running Drone Internal Update at step {t+1} -")
            drone.update(query_img=image_for_update)  # Pass the fetched image (or None)
            # --- ADDED: Retrieve and store results ---
            last_results = drone.get_last_top_k_results()
            if last_results is not None:
                top_k_results_history[t] = last_results  # Store results for this step t
            print(f" - Update finished -")

        # 6. Calculate Error
        current_simulated_pos = drone.position_history[-1]
        error = np.linalg.norm(current_simulated_pos - true_position_ecef)
        simulation_errors.append(error)

    print("\n--- Simulation Finished ---")

    # --- Prepare Data for Visualization ---
    # Drone history already includes initial position
    simulated_positions_ecef_np = np.array(drone.position_history)
    # True history also includes initial position
    true_positions_ecef_np = np.array(true_positions_history)
    # Get database points
    db_positions_ecef_np = drone.get_database_ecef()

    # Create a simple DataFrame for error plotting
    error_df = pd.DataFrame(
        {
            "time": np.arange(duration),  # 0 to duration-1
            "simulated_error": simulation_errors,
        }
    )
    error_df.to_csv(output_dir / "simulation_error_stats_ecef.csv", index=False)

    print("\n--- Generating Visualizations ---")
    # Check data shapes
    print(f"True positions shape: {true_positions_ecef_np.shape}")
    print(f"Simulated positions shape: {simulated_positions_ecef_np.shape}")
    if db_positions_ecef_np is not None:
        print(f"Database positions shape: {db_positions_ecef_np.shape}")
    else:
        print("Database positions: None")

    # --- UPDATED: Call with top_k_results_history ---
    generate_method_visualizations(
        true_positions_ecef=true_positions_ecef_np,
        simulated_positions_ecef=simulated_positions_ecef_np,
        db_positions_ecef=(
            db_positions_ecef_np
            if db_positions_ecef_np is not None
            else np.empty((0, 3))
        ),
        error_data=error_df,
        top_k_history=top_k_results_history,  # Pass the new data
        update_interval=update_interval,  # Pass interval info
    )

    # Also call the error plot function if it exists and you want it
    # (Assuming plot_simulation_error was also intended to be called)
    # Check if error_df exists and has the required columns
    if (
        "error_df" in locals()
        and error_df is not None
        and "simulated_error" in error_df.columns
    ):
        plot_simulation_error(error_df, output_dir / "simulation_error_ecef.png")
    else:
        print("Skipping error plot: Data not available.")

    print("\n--- Visualization Script Finished ---")
    print(f"Output files generated in: {output_dir}")
