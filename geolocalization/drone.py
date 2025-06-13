import numpy as np
import random
from PIL import Image
from geolocalization import config

class Drone:
    """
    Simulates a drone with both desired (ground truth) and noisy trajectories.
    Implements VIO measurements, dynamic trajectory generation, and control corrections.
    """
    def __init__(self, database, start_world_pos_m: tuple = None):
        self.database = database
        self.m_per_pixel = config.M_PER_PIXEL
        
        # Initialize positions at map center
        map_center_x = 0.0  # Map center in world coordinates
        map_center_y = 0.0
        if start_world_pos_m is None:
            start_world_pos_m = (map_center_x, map_center_y)
        
        self.desired_pos_m = np.array(start_world_pos_m, dtype=np.float64)  # Ground truth trajectory
        self.current_pos_m = np.array(start_world_pos_m, dtype=np.float64)  # Noisy trajectory (VIO)
        
        # Calculate map bounds in world coordinates
        map_width_m = self.database.map_w * self.m_per_pixel
        map_height_m = self.database.map_h * self.m_per_pixel
        
        # Zig-zag trajectory parameters with larger safety margin
        self.zig_zag_margin = 500.0  # Stay 500m from edges for safety
        self.x_min = -map_width_m/2 + self.zig_zag_margin
        self.x_max = map_width_m/2 - self.zig_zag_margin
        self.y_min = -map_height_m/2 + self.zig_zag_margin
        self.y_max = map_height_m/2 - self.zig_zag_margin
        
        # Zig-zag state
        self.zig_zag_rows = 8  # Number of horizontal rows
        self.current_row = 0
        self.going_right = True
        self.row_height = (self.y_max - self.y_min) / (self.zig_zag_rows - 1)
        
        # Movement parameters
        self.speed_m_per_step = 2.0  # Constant speed
        self.total_distance_traveled = 0.0
        
        # Current target
        self._update_zig_zag_target()
        
        # VIO error model (realistic drift)
        self.vio_bias = np.array([1.0, 0.5], dtype=np.float64)  # Initial bias
        self.vio_accumulated_error = np.array([0.0, 0.0], dtype=np.float64)
        
        # Correction system
        self.correction_active = False
        self.correction_target_m = None

    def _update_zig_zag_target(self):
        """Update the target position for zig-zag pattern."""
        # Calculate current row y-position
        target_y = self.y_min + self.current_row * self.row_height
        
        # Calculate target x-position based on direction
        if self.going_right:
            target_x = self.x_max
        else:
            target_x = self.x_min
            
        self.target_pos = np.array([target_x, target_y], dtype=np.float64)

    def step(self):
        """Advance the drone one time step with zig-zag movement."""
        # Calculate direction to target
        direction = self.target_pos - self.desired_pos_m
        distance_to_target = np.linalg.norm(direction)
        
        # Check if we've reached the target
        if distance_to_target < self.speed_m_per_step:
            # Move to exact target
            self.desired_pos_m = self.target_pos.copy()
            
            # Update to next target
            if self.going_right:
                # Reached right edge, go to next row on left
                self.going_right = False
                self.current_row += 1
            else:
                # Reached left edge, go to next row on right
                self.going_right = True
                self.current_row += 1
            
            # Check if we've completed all rows
            if self.current_row >= self.zig_zag_rows:
                self.current_row = 0  # Start over
                
            self._update_zig_zag_target()
        else:
            # Move toward target
            normalized_direction = direction / distance_to_target
            self.desired_pos_m += normalized_direction * self.speed_m_per_step
        
        # Update VIO with realistic drift
        self._update_vio_position()
        
        # Update statistics
        self.total_distance_traveled += self.speed_m_per_step

    def _update_vio_position(self):
        """Update VIO position with realistic drift error."""
        # Update VIO bias (slow drift)
        bias_change = np.random.normal(0, config.VIO_BIAS_DRIFT_RATE, 2)
        self.vio_bias = np.clip(
            self.vio_bias + bias_change,
            -config.VIO_BIAS_MAX, config.VIO_BIAS_MAX
        )
        
        # Add random noise
        random_noise = np.random.normal(0, config.VIO_RANDOM_NOISE_STD, 2)
        
        # Accumulate error
        step_error = self.vio_bias * 0.01 + random_noise  # Bias grows slowly
        self.vio_accumulated_error += step_error
        
        # Update VIO position
        self.current_pos_m = self.desired_pos_m + self.vio_accumulated_error

    def get_exploration_stats(self) -> dict:
        """Get exploration statistics."""
        return {
            'total_distance': self.total_distance_traveled,
            'current_row': self.current_row,
            'going_right': self.going_right,
            'target_pos': self.target_pos.tolist()
        }

    def _generate_random_start_position(self) -> tuple:
        """Generate starting position for rectangular coverage pattern."""
        # Start at top-left corner of the coverage area
        map_w_m = self.database.map_w * self.m_per_pixel
        map_h_m = self.database.map_h * self.m_per_pixel
        
        margin_m = 600  # Safe margin from edges
        start_x = -map_w_m/2 + margin_m
        start_y = -map_h_m/2 + margin_m
        
        return (start_x, start_y)

    def _update_vio_bias(self):
        """Update VIO bias with realistic drift behavior."""
        # Bias evolves as a random walk (realistic VIO drift)
        drift_change = np.random.normal(0, config.VIO_BIAS_DRIFT_RATE, size=2)
        self.vio_bias += drift_change
        
        # Clamp bias to maximum values
        self.vio_bias = np.clip(self.vio_bias, -config.VIO_BIAS_MAX, config.VIO_BIAS_MAX)

    def _check_bounds(self, position_m: np.ndarray) -> np.ndarray:
        """Ensure position stays within map bounds with strict safety margins."""
        map_w_m = self.database.map_w * self.m_per_pixel
        map_h_m = self.database.map_h * self.m_per_pixel
        
        # Strict margin to prevent going out of bounds
        margin_m = 400  # 400m safety margin
        max_x = map_w_m/2 - margin_m
        min_x = -map_w_m/2 + margin_m
        max_y = map_h_m/2 - margin_m
        min_y = -map_h_m/2 + margin_m
        
        # Clamp to safe bounds
        position_m[0] = np.clip(position_m[0], min_x, max_x)
        position_m[1] = np.clip(position_m[1], min_y, max_y)
        
        return position_m

    def move_step(self) -> tuple:
        """
        Execute one movement step with realistic VIO drift error.
        Returns: (vio_delta_m, epsilon_m, should_update)
        """
        # Store previous position for delta calculation
        prev_pos = self.current_pos_m.copy()
        
        # Execute one zig-zag step
        self.step()
        
        # Calculate VIO delta (what VIO observes as movement)
        vio_delta_m = self.current_pos_m - prev_pos
        
        # Calculate epsilon (VIO measurement error magnitude)
        true_delta_m = self.desired_pos_m - (prev_pos - self.vio_accumulated_error + 
                      (self.current_pos_m - self.desired_pos_m))
        epsilon_m = np.linalg.norm(vio_delta_m - true_delta_m)
        
        # Always update (simple approach)
        should_update = True
        
        return vio_delta_m, epsilon_m, should_update

    def apply_correction(self, correction_target_world_m: tuple):
        """Apply position correction based on confident localization."""
        self.correction_target_m = correction_target_world_m
        self.correction_active = True
        print(f"Correction target set: {correction_target_world_m}")

    def get_camera_view(self, view_size_m: float = 50.0) -> Image.Image:
        """
        Get the current camera view from the drone's true position.
        Returns a cropped image centered on the drone's true position.
        """
        # Use true position for camera (this simulates perfect camera positioning)
        true_pos_px = self.database.world_to_pixel(self.desired_pos_m[0], self.desired_pos_m[1])
        
        # Calculate view size in pixels
        view_size_px = int(view_size_m / self.m_per_pixel)

        # Calculate crop bounds
        left = true_pos_px[0] - view_size_px // 2
        top = true_pos_px[1] - view_size_px // 2
        right = left + view_size_px
        bottom = top + view_size_px
        
        # Ensure bounds are within map
        left = max(0, left)
        top = max(0, top)
        right = min(self.database.map_w, right)
        bottom = min(self.database.map_h, bottom)
        
        # Crop and resize to standard patch size
        camera_view = self.database.map_image.crop((left, top, right, bottom))
        
        # Resize to standard patch size for embedding
        target_size = (self.database.patch_size_px, self.database.patch_size_px)
        camera_view = camera_view.resize(target_size, Image.Resampling.LANCZOS)
        
        return camera_view

    def get_true_position_world(self) -> tuple:
        """Get the true world position in meters."""
        return tuple(self.desired_pos_m)

    def get_vio_position_world(self) -> tuple:
        """Get the VIO-estimated world position in meters."""
        return tuple(self.current_pos_m)

    def get_true_position_px(self) -> tuple:
        """Get the true position in pixel coordinates."""
        return self.database.world_to_pixel(self.desired_pos_m[0], self.desired_pos_m[1])

    def get_vio_position_px(self) -> tuple:
        """Get the VIO-estimated position in pixel coordinates."""
        return self.database.world_to_pixel(self.current_pos_m[0], self.current_pos_m[1])

    def get_position_error_m(self) -> float:
        """Get the current position error magnitude in meters."""
        return np.linalg.norm(self.desired_pos_m - self.current_pos_m)

    def get_vio_bias(self) -> np.ndarray:
        """Get the current VIO bias vector."""
        return self.vio_bias.copy()

    def get_accumulated_drift(self) -> np.ndarray:
        """Get the accumulated VIO drift error."""
        return self.vio_accumulated_error.copy()

    def is_within_bounds(self) -> bool:
        """Check if drone is within safe map bounds."""
        map_w_m = self.database.map_w * self.m_per_pixel
        map_h_m = self.database.map_h * self.m_per_pixel
        margin_m = 400  # Warning threshold - match zig-zag margin
        
        max_x = map_w_m/2 - margin_m
        min_x = -map_w_m/2 + margin_m
        max_y = map_h_m/2 - margin_m
        min_y = -map_h_m/2 + margin_m
        
        return (min_x <= self.desired_pos_m[0] <= max_x and 
                min_y <= self.desired_pos_m[1] <= max_y) 