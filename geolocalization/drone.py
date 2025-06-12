import numpy as np
import random
from geolocalization import config

class Drone:
    """
    Simulates the drone's flight, including its true position,
    VIO-estimated position, and the generation of a flight path.
    """
    def __init__(self, map_w_px: int, map_h_px: int, m_per_pixel: float):
        self.map_w_px = map_w_px
        self.map_h_px = map_h_px
        self.m_per_pixel = m_per_pixel
        self.num_steps = config.NUM_STEPS
        self.speed_mps = config.FLIGHT_SPEED_MPS
        self.vio_error_std_m = config.VIO_ERROR_STD_M

        # True state
        self.x_px, self.y_px = self._generate_start_position()
        self.path_waypoints_px = self._generate_segmented_path()
        self.current_waypoint_index = 0

        # VIO state (starts perfectly aligned with true state)
        self.vio_x_px, self.vio_y_px = self.x_px, self.y_px

    def _generate_start_position(self):
        """Generates a random starting position away from the edges."""
        border = 100 # pixels
        x = random.randint(border, self.map_w_px - border)
        y = random.randint(border, self.map_h_px - border)
        return x, y

    def _generate_segmented_path(self):
        """
        Generates a path with longer, straighter segments and random turns.
        """
        path = [(self.x_px, self.y_px)]
        num_segments = 10 # Create 10 major segments for the flight
        steps_per_segment = self.num_steps // num_segments

        current_pos = np.array([self.x_px, self.y_px], dtype=np.float64)
        
        # Start with a random direction
        angle = random.uniform(0, 2 * np.pi)
        
        for i in range(self.num_steps - 1):
            # After each segment, pick a new (slightly different) direction
            if i > 0 and i % steps_per_segment == 0:
                angle += random.uniform(-np.pi/4, np.pi/4) # Turn up to 45 degrees

            # Move in the current direction
            step_size_m = self.speed_mps # Assumes 1 step = 1 second
            step_size_px = step_size_m / self.m_per_pixel
            
            move_vec = np.array([np.cos(angle), np.sin(angle)]) * step_size_px
            
            # Add some minor randomness to the angle to make it less robotic
            angle += random.uniform(-0.1, 0.1) # Radians
            
            next_pos = current_pos + move_vec

            # Boundary checks
            if not (0 <= next_pos[0] < self.map_w_px and 0 <= next_pos[1] < self.map_h_px):
                # If hitting a boundary, make a significant turn and retry
                angle += random.uniform(np.pi/2, 3*np.pi/2) # Turn between 90 and 270 degrees
                move_vec = np.array([np.cos(angle), np.sin(angle)]) * step_size_px
                next_pos = current_pos + move_vec
                # Ensure the bounce doesn't immediately go out of bounds again
                next_pos[0] = np.clip(next_pos[0], 0, self.map_w_px - 1)
                next_pos[1] = np.clip(next_pos[1], 0, self.map_h_px - 1)


            current_pos = next_pos
            path.append((int(current_pos[0]), int(current_pos[1])))

        print(f"Generated a segmented flight path with {len(path)} waypoints.")
        return path

    def _generate_random_walk_path(self):
        """
        Generates a random walk flight path within the image boundaries.
        Returns a list of (x, y) pixel coordinates for each step.
        """
        path = [self.true_pos_m.copy()]
        current_pos = self.true_pos_m.copy()

        # Define possible moves in meters (e.g., 10m steps)
        step_size_m = 10.0
        moves = [np.array([step_size_m, 0]), np.array([-step_size_m, 0]),
                 np.array([0, step_size_m]), np.array([0, -step_size_m])]

        for _ in range(self.num_steps - 1):
            # Choose a random direction
            move = random.choice(moves)
            next_pos = current_pos + move

            # Check if the next position is within map boundaries (with a margin)
            if (0 < next_pos[0] < self.map_w_px * self.m_per_pixel and
                0 < next_pos[1] < self.map_h_px * self.m_per_pixel):
                current_pos = next_pos
                path.append(current_pos.copy())
        
        print(f"Generated a flight path with {len(path)} waypoints.")
        return path

    def move(self):
        """
        Moves the drone to the next waypoint, simulates VIO error, and updates state.
        Returns the VIO delta (motion vector in meters) and the VIO error magnitude for this step.
        """
        # Get the true current and next positions from the pre-computed path
        if self.current_waypoint_index >= len(self.path_waypoints_px) - 1:
            print("End of flight path reached.")
            return None, None
            
        true_current_pos_px = np.array(self.path_waypoints_px[self.current_waypoint_index])
        self.current_waypoint_index += 1
        true_next_pos_px = np.array(self.path_waypoints_px[self.current_waypoint_index])

        # This is the actual, perfect movement vector
        true_delta_px = true_next_pos_px - true_current_pos_px
        true_delta_m = true_delta_px * self.m_per_pixel
        
        # Simulate VIO error (epsilon) as a random noise vector
        vio_error_m = np.random.normal(0, self.vio_error_std_m, size=2)
        
        # The VIO system observes a motion vector that is the sum of the true motion and the error
        vio_delta_m = true_delta_m + vio_error_m
        
        # Update the drone's true position to the next waypoint
        self.x_px, self.y_px = true_next_pos_px[0], true_next_pos_px[1]
        
        # Update the drone's VIO-estimated position
        vio_delta_px = vio_delta_m / self.m_per_pixel
        self.vio_x_px += vio_delta_px[0]
        self.vio_y_px += vio_delta_px[1]
        
        # The magnitude of the error vector is used to grow the confidence radius
        epsilon_m = np.linalg.norm(vio_error_m)
        
        return vio_delta_m, epsilon_m

    def get_true_position_px(self):
        """Returns the drone's true position in pixels."""
        return (int(self.x_px), int(self.y_px))

    def get_vio_position_px(self):
        """Returns the drone's VIO-estimated position in pixels."""
        return (int(self.vio_x_px), int(self.vio_y_px)) 