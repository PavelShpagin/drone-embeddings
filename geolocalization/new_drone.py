import numpy as np
import random
import math
from PIL import Image

class NewDrone:
    """
    Implements a drone with two trajectories:
    1. Desired trajectory (ground truth)
    2. Current trajectory (noisy VIO measurements)
    
    Supports dynamic trajectory generation and control corrections.
    """
    def __init__(self, config, localizer, start_world_pos_m: tuple = None):
        self.config = config
        self.localizer = localizer
        
        # Calculate map bounds for zigzag pattern
        self.map_w_m = self.localizer.map_w * self.localizer.m_per_pixel
        self.map_h_m = self.localizer.map_h * self.localizer.m_per_pixel
        self.margin_m = 400  # Stay away from edges
        
        # Initialize positions - start at top-left corner
        if start_world_pos_m is None:
            start_world_pos_m = (-self.map_w_m/2 + self.margin_m, -self.map_h_m/2 + self.margin_m)
        
        # Two trajectories as specified
        self.desired_pos_m = np.array(start_world_pos_m, dtype=np.float64)  # Ground truth
        self.current_pos_m = np.array(start_world_pos_m, dtype=np.float64)  # Noisy VIO
        
        # Zigzag trajectory state
        self.zigzag_segment_length_m = 800.0  # Much longer straight segments
        self.zigzag_row_height_m = 400.0      # Larger distance between rows
        self.current_direction = np.array([1.0, 0.0])  # Start moving right
        self.segment_progress = 0.0
        self.row_number = 0
        self.moving_right = True
        
        # Movement state
        self.total_distance_traveled = 0.0
        self.step_size_m = 3.0  # Faster movement for larger coverage
        
        # Control state
        self.correction_active = False
        self.correction_target_m = None
        
        print(f"NewDrone initialized at {start_world_pos_m} for zigzag coverage")

    def _generate_random_start_position(self) -> tuple:
        """Generate a random starting position within safe map bounds."""
        margin_m = 500
        map_w_m = self.localizer.map_w * self.localizer.m_per_pixel
        map_h_m = self.localizer.map_h * self.localizer.m_per_pixel
        
        x_m = random.uniform(-map_w_m/2 + margin_m, map_w_m/2 - margin_m)
        y_m = random.uniform(-map_h_m/2 + margin_m, map_h_m/2 - margin_m)
        
        return (x_m, y_m)

    def _generate_zigzag_trajectory_move(self) -> np.ndarray:
        """Generate the next movement vector using systematic zigzag pattern."""
        
        # Check if we need to turn (reached end of current segment)
        if self.segment_progress >= self.zigzag_segment_length_m:
            # Determine next direction based on current state
            if abs(self.current_direction[1]) < 0.1:  # Currently moving horizontally
                # Switch to vertical movement (down to next row)
                self.current_direction = np.array([0.0, 1.0])
                self.segment_progress = 0.0
                self.zigzag_segment_length_m = self.zigzag_row_height_m
            else:  # Currently moving vertically
                # Switch to horizontal movement for next row
                self.row_number += 1
                self.moving_right = not self.moving_right  # Alternate direction
                self.current_direction = np.array([1.0 if self.moving_right else -1.0, 0.0])
                self.segment_progress = 0.0
                self.zigzag_segment_length_m = 800.0  # Reset to normal segment length
        
        # Calculate movement vector
        move_vector = self.current_direction * self.step_size_m
        
        # Update progress
        self.segment_progress += self.step_size_m
        self.total_distance_traveled += self.step_size_m
        
        return move_vector

    def _check_bounds(self, position_m: np.ndarray) -> np.ndarray:
        """Ensure position stays within map bounds."""
        position_m[0] = np.clip(position_m[0], -self.map_w_m/2 + self.margin_m, self.map_w_m/2 - self.margin_m)
        position_m[1] = np.clip(position_m[1], -self.map_h_m/2 + self.margin_m, self.map_h_m/2 - self.margin_m)
        
        return position_m

    def move_step(self) -> tuple:
        """
        Execute one movement step.
        Returns: (vio_delta_m, epsilon_m, should_update)
        """
        # Generate desired movement
        if self.correction_active and self.correction_target_m is not None:
            # Move towards correction target
            target_vector = np.array(self.correction_target_m) - self.desired_pos_m
            distance_to_target = np.linalg.norm(target_vector)
            
            if distance_to_target > self.step_size_m:
                move_vector = (target_vector / distance_to_target) * self.step_size_m
            else:
                move_vector = target_vector
                self.correction_active = False
                self.correction_target_m = None
                print("Correction completed")
        else:
            # Normal zigzag trajectory
            move_vector = self._generate_zigzag_trajectory_move()
        
        # Update desired (ground truth) position
        self.desired_pos_m += move_vector
        self.desired_pos_m = self._check_bounds(self.desired_pos_m)
        
        # Generate VIO error - bias it in the direction of movement for realism
        move_direction = move_vector / np.linalg.norm(move_vector) if np.linalg.norm(move_vector) > 0 else np.array([0, 0])
        
        # Base error (small random component)
        base_error = np.random.normal(0, self.config.VIO_ERROR_STD_M * 0.5, size=2)
        
        # Directional bias error (larger component aligned with movement)
        directional_bias = np.random.normal(0, self.config.VIO_ERROR_STD_M * 0.8) * move_direction
        
        # Combine errors
        vio_error_m = base_error + directional_bias
        epsilon_m = np.linalg.norm(vio_error_m)
        
        # VIO observes the movement with error
        vio_observed_movement = move_vector + vio_error_m
        
        # Update current (noisy VIO) position
        self.current_pos_m += vio_observed_movement
        self.current_pos_m = self._check_bounds(self.current_pos_m)
        
        # Check if we should update localization (every UPDATE_INTERVAL_M)
        should_update = (self.total_distance_traveled % self.config.UPDATE_INTERVAL_M) < self.step_size_m
        
        return vio_observed_movement, epsilon_m, should_update

    def apply_correction(self, correction_target_world_m: tuple):
        """Apply correction to move towards estimated true position."""
        self.correction_active = True
        self.correction_target_m = correction_target_world_m
        print(f"Correction initiated towards {correction_target_world_m}")

    def get_camera_view(self, view_size_m: float = 50.0) -> Image.Image:
        """Get camera view from true position (simulates perfect camera)."""
        true_pos_px = self._world_to_pixel(self.desired_pos_m[0], self.desired_pos_m[1])
        
        view_size_px = int(view_size_m / self.localizer.m_per_pixel)
        
        # Calculate crop bounds
        left = max(0, true_pos_px[0] - view_size_px // 2)
        top = max(0, true_pos_px[1] - view_size_px // 2)
        right = min(self.localizer.map_w, left + view_size_px)
        bottom = min(self.localizer.map_h, top + view_size_px)
        
        # Crop and resize
        camera_view = self.localizer.map_image.crop((left, top, right, bottom))
        camera_view = camera_view.resize((self.config.CROP_SIZE_PX, self.config.CROP_SIZE_PX),
                                        Image.Resampling.LANCZOS)
        
        return camera_view

    def _world_to_pixel(self, x_m: float, y_m: float) -> tuple:
        """Convert world coordinates to pixel coordinates."""
        x_px = int(x_m / self.localizer.m_per_pixel + self.localizer.map_w // 2)
        y_px = int(y_m / self.localizer.m_per_pixel + self.localizer.map_h // 2)
        return (x_px, y_px)

    def get_true_position_world(self) -> tuple:
        """Get true world position."""
        return tuple(self.desired_pos_m)

    def get_vio_position_world(self) -> tuple:
        """Get VIO-estimated world position."""
        return tuple(self.current_pos_m)

    def get_position_error_m(self) -> float:
        """Get position error magnitude."""
        return np.linalg.norm(self.desired_pos_m - self.current_pos_m)

    def is_within_bounds(self) -> bool:
        """Check if drone is within safe bounds."""
        safe_margin = self.margin_m + 100  # Extra safety margin for bounds checking
        
        return (abs(self.desired_pos_m[0]) < self.map_w_m/2 - safe_margin and 
                abs(self.desired_pos_m[1]) < self.map_h_m/2 - safe_margin) 