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
        
        # Calculate map bounds for trajectory generation
        self.map_w_m = self.localizer.map_w * self.localizer.m_per_pixel
        self.map_h_m = self.localizer.map_h * self.localizer.m_per_pixel
        self.margin_m = 400  # Stay away from edges
        
        # Initialize positions
        if start_world_pos_m is None:
            # Use random start if not provided
            start_world_pos_m = self._generate_random_start_position()
        
        # Two trajectories as specified
        self.desired_pos_m = np.array(start_world_pos_m, dtype=np.float64)  # Ground truth
        self.current_pos_m = np.array(start_world_pos_m, dtype=np.float64)  # Noisy VIO
        
        # Dynamic trajectory state
        self.current_heading_rad = random.uniform(0, 2 * math.pi) # Initial random heading
        self.current_segment_length = 0.0
        self.target_segment_length = random.uniform(self.config.MIN_SEGMENT_LENGTH_M, self.config.MAX_SEGMENT_LENGTH_M)
        
        # Movement state with speed and acceleration
        self.current_speed_mps = 1.0  # Start at mean speed
        self.current_accel_mps2 = 0.0  # Initial acceleration
        self.max_accel_mps2 = 0.5  # Maximum acceleration change per step
        self.speed_bias = 0.0  # Bias to maintain mean speed around 1 m/s
        self.total_distance_traveled = 0.0
        self.distance_traveled_since_last_update = 0.0
        
        # Accumulated VIO data for batched updates
        self._accumulated_vio_delta_m = np.array([0.0, 0.0], dtype=np.float64)
        self._accumulated_vio_error_m = np.array([0.0, 0.0], dtype=np.float64)
        self._accumulated_epsilon_m = 0.0
        
        # Control state
        self.correction_active = False
        self.correction_target_m = None
        self.correction_path_length = 0.0
        self.correction_progress = 0.0
        
        print(f"NewDrone initialized at {start_world_pos_m} with dynamic trajectory.")

    def _generate_random_start_position(self) -> tuple:
        """Generate a random starting position within safe map bounds."""
        margin_m = 500
        map_w_m = self.localizer.map_w * self.localizer.m_per_pixel
        map_h_m = self.localizer.map_h * self.localizer.m_per_pixel
        
        x_m = random.uniform(-map_w_m/2 + margin_m, map_w_m/2 - margin_m)
        y_m = random.uniform(-map_h_m/2 + margin_m, map_h_m/2 - margin_m)
        
        return (x_m, y_m)

    def _update_speed(self):
        """Update speed with random acceleration while maintaining bounds and mean."""
        # Calculate speed bias to maintain mean around 1 m/s
        if self.total_distance_traveled > 0:
            current_mean = self.total_distance_traveled / (len(self.true_trajectory) if hasattr(self, 'true_trajectory') else 1)
            self.speed_bias = 0.1 * (1.0 - current_mean)  # Adjust bias based on current mean
        
        # Generate random acceleration change
        accel_change = random.uniform(-self.max_accel_mps2, self.max_accel_mps2)
        self.current_accel_mps2 = np.clip(self.current_accel_mps2 + accel_change, -self.max_accel_mps2, self.max_accel_mps2)
        
        # Update speed with acceleration and bias
        new_speed = self.current_speed_mps + self.current_accel_mps2 + self.speed_bias
        self.current_speed_mps = np.clip(new_speed, 0.0, 5.0)  # Keep speed between 0 and 5 m/s

    def _generate_dynamic_trajectory_move(self) -> np.ndarray:
        """Generate the next movement vector for a dynamic trajectory with variable speed."""
        # Update speed with random acceleration
        self._update_speed()
        
        # Check if current segment is complete or if we need to turn
        if self.current_segment_length >= self.target_segment_length or \
           random.random() < self.config.TURN_PROBABILITY:
            
            # Start a new segment
            self.target_segment_length = random.uniform(self.config.MIN_SEGMENT_LENGTH_M, self.config.MAX_SEGMENT_LENGTH_M)
            self.current_segment_length = 0.0
            
            # Apply a random turn
            turn_angle = random.uniform(-self.config.MAX_TURN_ANGLE_RAD, self.config.MAX_TURN_ANGLE_RAD)
            self.current_heading_rad += turn_angle
            self.current_heading_rad = self.current_heading_rad % (2 * math.pi)
            
            print(f"Drone new segment: length {self.target_segment_length:.1f}m, heading {math.degrees(self.current_heading_rad):.1f} deg, speed {self.current_speed_mps:.1f} m/s")
        
        # Calculate movement vector based on current heading and speed
        step_distance = self.current_speed_mps  # Use current speed as step distance
        move_vector = np.array([
            step_distance * math.cos(self.current_heading_rad),
            step_distance * math.sin(self.current_heading_rad)
        ])
        
        self.current_segment_length += step_distance
        self.total_distance_traveled += step_distance
        
        return move_vector

    def _check_bounds(self, position_m: np.ndarray) -> np.ndarray:
        """Ensure position stays within map bounds."""
        position_m[0] = np.clip(position_m[0], -self.map_w_m/2 + self.margin_m, self.map_w_m/2 - self.margin_m)
        position_m[1] = np.clip(position_m[1], -self.map_h_m/2 + self.margin_m, self.map_h_m/2 - self.margin_m)
        
        return position_m

    def move_step(self) -> bool:
        """
        Execute one movement step.
        Accumulates VIO measurements and returns True if localization update is due.
        Returns: should_update_localization (bool)
        """
        # Determine the movement vector based on correction or dynamic trajectory
        if self.correction_active and self.correction_target_m is not None:
            remaining_vector = np.array(self.correction_target_m) - self.desired_pos_m
            remaining_distance = np.linalg.norm(remaining_vector)
            
            # Use current speed for correction movement
            correction_speed = self.current_speed_mps * self.config.CORRECTION_SPEED_FACTOR
            if remaining_distance > correction_speed:
                move_vector = (remaining_vector / remaining_distance) * correction_speed
                self.correction_progress += np.linalg.norm(move_vector)
            else:
                move_vector = remaining_vector
                self.correction_active = False
                self.correction_target_m = None
                self.correction_progress = 0.0 # Reset progress
                print("Correction completed by drone.")
        else:
            move_vector = self._generate_dynamic_trajectory_move()
        
        # Update desired (ground truth) position
        self.desired_pos_m += move_vector
        self.desired_pos_m = self._check_bounds(self.desired_pos_m)
        
        # Generate VIO error for this step
        vio_error_m_step = np.array([0.0, 0.0]) # Error vector for this step
        if np.linalg.norm(move_vector) > 0:
            noise_x = np.random.normal(0, self.config.VIO_ERROR_STD_M * 0.5)
            noise_y = np.random.normal(0, self.config.VIO_ERROR_STD_M * 0.5)
            vio_error_m_step = np.array([noise_x, noise_y])

        # VIO observes the movement with error
        vio_observed_movement_step = move_vector + vio_error_m_step
        
        # Update current (noisy VIO) position
        self.current_pos_m += vio_observed_movement_step
        self.current_pos_m = self._check_bounds(self.current_pos_m)

        # Accumulate VIO data
        self._accumulated_vio_delta_m += vio_observed_movement_step
        self._accumulated_vio_error_m += vio_error_m_step # Accumulate error vector
        self.distance_traveled_since_last_update += np.linalg.norm(move_vector) # Accumulate true distance for update trigger
        
        should_update_localization = False
        if self.distance_traveled_since_last_update >= self.config.UPDATE_INTERVAL_M:
            should_update_localization = True
        
        # Also trigger update if correction just ended, regardless of distance
        if not self.correction_active and self.correction_target_m is None and self.correction_progress > 0:
            should_update_localization = True

        return should_update_localization

    def get_accumulated_vio_data(self) -> tuple:
        """
        Returns accumulated VIO delta (displacement vector) and epsilon (magnitude of accumulated error vector).
        """
        # Calculate epsilon as the norm of the accumulated error vector
        accumulated_epsilon_m = np.linalg.norm(self._accumulated_vio_error_m)
        return self._accumulated_vio_delta_m, accumulated_epsilon_m

    def reset_accumulated_vio_data(self):
        """
        Resets accumulated VIO data and distance traveled for update trigger.
        """
        self._accumulated_vio_delta_m = np.array([0.0, 0.0], dtype=np.float64)
        self._accumulated_vio_error_m = np.array([0.0, 0.0], dtype=np.float64) # Reset error vector
        self._accumulated_epsilon_m = 0.0
        self.distance_traveled_since_last_update = 0.0

    def apply_correction(self, correction_target_world_m: tuple):
        """
        Drone initiates correction movement towards an estimated true position.
        """
        self.correction_active = True
        self.correction_target_m = np.array(correction_target_world_m, dtype=np.float64)
        # Calculate the total distance to travel during correction
        self.correction_path_length = np.linalg.norm(self.correction_target_m - self.desired_pos_m)
        self.correction_progress = 0.0 # Reset progress
        print(f"Correction initiated towards {correction_target_world_m}. Path length: {self.correction_path_length:.1f}m")

    def get_camera_view(self, view_size_m: float = 50.0) -> Image.Image:
        """
        Get camera view from true position (simulates perfect camera).
        """
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
        """
        Convert world coordinates to pixel coordinates.
        """
        x_px = int(x_m / self.localizer.m_per_pixel + self.localizer.map_w / 2.0)
        y_px = int(y_m / self.localizer.m_per_pixel + self.localizer.map_h / 2.0)
        
        return (x_px, y_px)

    def get_true_position_world(self) -> tuple:
        """
        Get the drone's true position in world coordinates (meters).
        """
        return tuple(self.desired_pos_m)

    def get_vio_position_world(self) -> tuple:
        """
        Get the drone's noisy VIO position in world coordinates (meters).
        """
        return tuple(self.current_pos_m)

    def get_position_error_m(self) -> float:
        """
        Calculate the Euclidean distance between true and VIO positions.
        """
        return np.linalg.norm(self.desired_pos_m - self.current_pos_m)

    def is_within_bounds(self) -> bool:
        """
        Check if the drone's true position is within the safe map bounds.
        """
        x, y = self.desired_pos_m
        map_w_m = self.localizer.map_w * self.localizer.m_per_pixel
        map_h_m = self.localizer.map_h * self.localizer.m_per_pixel
        
        return (-map_w_m/2 + self.margin_m <= x <= map_w_m/2 - self.margin_m and
                -map_h_m/2 + self.margin_m <= y <= map_h_m/2 - self.margin_m) 