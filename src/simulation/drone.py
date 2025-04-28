import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import pyproj
import yaml
from PIL import Image
import torch  # Needed for device check potentially

# Import necessary components from the project
from src.elevation import get_google_elevation
from src.localization.geo_localizer import GeoLocalizer  # Updated import
from src.google_maps import get_static_map, calculate_google_zoom
from src.azure_maps import get_azure_maps_image, calculate_azure_zoom


@dataclass
class DroneFlight:
    """
    Simulates a drone's flight primarily in ECEF coordinates.
    Includes onboard GeoLocalizer for position updates.
    Accepts velocity commands in the local ENU frame for steps.
    """

    # --- Fields without defaults (for __init__) ---
    model_path: str
    config_path: str
    start_lat: float
    start_lng: float
    start_height: float  # Local height above ground

    # --- Fields with defaults (for __init__) ---
    secrets_path: str = ".env"

    # --- Fields excluded from __init__ or with default_factory ---
    position: np.ndarray = field(init=False)  # ECEF [x, y, z] meters
    velocity: np.ndarray = field(
        init=False
    )  # Last applied ECEF velocity [vx, vy, vz] m/step
    position_history: List[np.ndarray] = field(default_factory=list, init=False)
    localizer: GeoLocalizer = field(init=False)
    momentum: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    prev_corrected_positions: List[np.ndarray] = field(default_factory=list, init=False)
    _correction_params: dict = field(
        init=False, default_factory=dict
    )  # Store alpha, beta, K etc.
    _last_query_img_failed: bool = field(default=False, init=False)
    _last_top_k_results: Optional[Dict] = field(default=None, init=False)

    # --- Coordinate Systems & Transformers ---
    _ecef_crs = pyproj.CRS("EPSG:4978")
    _lla_crs = pyproj.CRS("EPSG:4326")
    _transformer_to_ecef = pyproj.Transformer.from_crs(
        _lla_crs, _ecef_crs, always_xy=True
    )
    _transformer_to_lla = pyproj.Transformer.from_crs(
        _ecef_crs, _lla_crs, always_xy=True
    )

    def __post_init__(self):
        """Initialize ECEF state and GeoLocalizer after dataclass setup."""
        print("Initializing DroneFlight...")

        # 1. Initialize GeoLocalizer
        self.localizer = GeoLocalizer(
            self.model_path, self.config_path, self.secrets_path
        )
        # Load correction parameters from localizer's config
        self._correction_params = self.localizer.config.get("correction", {})
        print(f"Correction params loaded: {self._correction_params}")

        # 2. Determine initial ECEF position
        print(
            f"Getting ground elevation for start point ({self.start_lat}, {self.start_lng})..."
        )
        ground_elevation = get_google_elevation(self.start_lat, self.start_lng)
        if ground_elevation is None:
            print("Warning: Failed to get start elevation, using 0m.")
            ground_elevation = 0
        start_altitude = ground_elevation + self.start_height
        print(
            f"Start ground elevation: {ground_elevation:.1f}m, Start height: {self.start_height:.1f}m -> Start Altitude: {start_altitude:.1f}m"
        )

        x, y, z = self._transformer_to_ecef.transform(
            self.start_lng, self.start_lat, start_altitude
        )
        self.position = np.array([x, y, z])
        print(f"Initial ECEF Position: {self.position}")

        # 3. Set initial velocity to zero (will be set by first step)
        self.velocity = np.zeros(3)

        # 4. Initialize history
        self.position_history = [self.position.copy()]

        # 5. Build database (can be slow, do it during init)
        db_params = self.localizer.config.get("database", {})
        self.localizer.build_database(
            lat_center=self.start_lat,
            lng_center=self.start_lng,
            grid_size=db_params.get("grid_size", 15),
            step_meters=db_params.get("step_meters", 50),
            height_range=db_params.get("height_range", 30),
            base_height=db_params.get(
                "base_height", 100
            ),  # Use configured base height for DB
        )

    @staticmethod
    def _rotation_matrix_enu_to_ecef(lat_rad, lon_rad) -> np.ndarray:
        """Rotation matrix from local ENU to ECEF frame."""
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
        return np.array(
            [
                [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                [0, cos_lat, sin_lat],
            ]
        )

    def _enu_to_ecef_velocity(self, lat_deg, lon_deg, velocity_enu) -> np.ndarray:
        """Transforms a velocity vector from local ENU to ECEF frame at the given LLA."""
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        R = self._rotation_matrix_enu_to_ecef(lat_rad, lon_rad)
        velocity_ecef = R @ np.asarray(velocity_enu)
        return velocity_ecef

    def get_current_lla(self) -> Tuple[float, float, float]:
        """Converts current ECEF position to LLA."""
        lon, lat, alt = self._transformer_to_lla.transform(*self.position)
        return lat, lon, alt

    def step(self, velocity_step_enu: np.ndarray):
        """
        Moves the drone one time step based on an ENU velocity command.

        Args:
            velocity_step_enu: The desired velocity for this step in the local ENU frame [East, North, Up] (m/step).
        """
        # 1. Get current LLA needed for ENU->ECEF conversion
        current_lat, current_lng, _ = self.get_current_lla()

        # 2. Convert ENU velocity command to ECEF velocity step
        velocity_step_ecef = self._enu_to_ecef_velocity(
            current_lat, current_lng, velocity_step_enu
        )

        # 3. Update ECEF position
        self.position += velocity_step_ecef

        # 4. Store the applied ECEF velocity for this step
        self.velocity = velocity_step_ecef

        # 5. Record history
        self.position_history.append(self.position.copy())
        # print(f"Step: New Pos ECEF: {self.position}, Last Vel ECEF: {self.velocity}") # Debug

    def _apply_momentum_correction(
        self, target_pos_ecef: np.ndarray, mean_similarity: float
    ):
        """Applies the Top-K Mean Momentum correction logic in ECEF."""
        alpha = self._correction_params.get("adjustment_alpha", 0.1)
        beta = self._correction_params.get("momentum_beta", 0.9)
        smoothing_window = self._correction_params.get("smoothing_window", 5)

        # Scale adjustment by similarity and alpha
        raw_adjustment = target_pos_ecef - self.position
        scaled_adjustment = alpha * mean_similarity * raw_adjustment

        # Update momentum
        self.momentum = beta * self.momentum + (1 - beta) * scaled_adjustment

        # Apply momentum
        new_pos = self.position + self.momentum

        # Apply smoothing
        self.prev_corrected_positions.append(new_pos)
        if len(self.prev_corrected_positions) > smoothing_window:
            self.prev_corrected_positions.pop(0)

        if len(self.prev_corrected_positions) > 1:
            # Use simple moving average for now, could use exponential weights
            smoothed_pos = np.mean(self.prev_corrected_positions, axis=0)
        else:
            smoothed_pos = new_pos  # No smoothing possible yet

        # Update drone's position
        self.position = smoothed_pos
        # Update history with the corrected position
        if self.position_history:  # Should always be true after init
            self.position_history[-1] = self.position.copy()

    def update(self, query_img: Optional[Image.Image]):
        """
        Performs localization using the onboard GeoLocalizer and updates
        the drone's ECEF position based on the Top-K Mean Momentum method,
        using the provided query image.

        Args:
            query_img: The satellite image (PIL.Image) corresponding to the drone's
                       assumed true location at this update step. If None, indicates
                       image fetching failed, and only momentum drift should be applied.
        Stores the top-k results.
        """
        # Reset last results before attempting update
        self._last_top_k_results = None

        if (
            self.localizer.db_embeddings is None
            or len(self.localizer.db_embeddings) == 0
        ):
            print("Update skipped: GeoLocalizer database is empty.")
            return

        # --- Check if a valid query image was provided ---
        if query_img is None:
            print(
                "Update Warning: No query image provided (fetch failed?). Applying momentum drift only."
            )
            self.position += self.momentum
            if self.position_history:
                self.position_history[-1] = self.position.copy()
            return

        # --- Localization using the provided image ---
        # 1. Find Top-K matches (ECEF coordinates)
        K = self._correction_params.get("top_k", 5)
        try:
            # Use the provided query_img directly
            _, top_coords_ecef, similarities, indices = (
                self.localizer.find_top_k_points(query_img, k=K)
            )
            # Store results BEFORE check for failure
            if top_coords_ecef is not None:  # Store even if empty list was returned
                self._last_top_k_results = {
                    "ecef_coords": [
                        c.tolist() for c in top_coords_ecef
                    ],  # Store as lists
                    "similarities": similarities,
                    "indices": indices,
                }
            else:  # Ensure reset if localization returns None
                self._last_top_k_results = None

            if top_coords_ecef is None or not top_coords_ecef:
                print(
                    "Update Warning: Localization failed to find matches using provided image."
                )
                # Apply momentum drift if no matches
                self.position += self.momentum
                if self.position_history:
                    self.position_history[-1] = self.position.copy()
                return  # Skip correction

        except Exception as e:
            print(f"Error during localization in drone.update: {e}")
            self._last_top_k_results = None  # Reset results on error
            # Apply momentum drift on error
            self.position += self.momentum
            if self.position_history:
                self.position_history[-1] = self.position.copy()
            return  # Skip correction

        # 2. Calculate target position (mean of Top-K ECEF points)
        top_k_mean_ecef = np.mean(np.array(top_coords_ecef), axis=0)
        mean_similarity = np.mean(similarities) if similarities else 0

        # 3. Apply Correction
        self._apply_momentum_correction(top_k_mean_ecef, mean_similarity)
        # print(f"Update: Corrected Pos ECEF: {self.position}, Momentum: {self.momentum}") # Debug

    def get_last_top_k_results(self) -> Optional[Dict]:
        """Returns the ECEF coords, similarities, and indices from the last update attempt."""
        # Return a copy to prevent external modification
        return (
            self._last_top_k_results.copy()
            if self._last_top_k_results is not None
            else None
        )

    def get_state_ecef(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns current ECEF position and last applied ECEF velocity."""
        return self.position.copy(), self.velocity.copy()

    def get_database_ecef(self) -> np.ndarray | None:
        """Returns the ECEF coordinates of the database points."""
        return (
            self.localizer.db_coords_ecef.copy()
            if self.localizer.db_coords_ecef is not None
            else None
        )
