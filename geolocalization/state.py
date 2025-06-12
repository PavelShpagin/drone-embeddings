import numpy as np
import cv2
from scipy.stats import norm
from PIL import Image
from geolocalization import config

class LocalizationState:
    """
    Manages the probability grid and confidence circle for localization.
    """
    def __init__(self, map_w_px: int, map_h_px: int, initial_pos_px: tuple, m_per_pixel: float):
        # Grid dimensions are based on the number of patches that fit in the map
        self.patch_size_px = int(config.GRID_PATCH_SIZE_M / m_per_pixel)
        self.grid_w = map_w_px // self.patch_size_px
        self.grid_h = map_h_px // self.patch_size_px

        # The probability grid stores a probability for each patch
        self.prob_grid = np.ones((self.grid_h, self.grid_w), dtype=np.float64)
        self._normalize_probabilities()

        self.m_per_pixel = m_per_pixel
        self.config = config

        # Confidence circle state
        self.center_x_px, self.center_y_px = initial_pos_px
        self.radius_m = self.config.INITIAL_RADIUS_M

    def _get_circle_mask(self):
        """Creates a boolean mask for the grid, True for patches inside the confidence circle."""
        y_grid, x_grid = np.ogrid[:self.grid_h, :self.grid_w]
        
        # Calculate pixel coordinates of each grid cell's center
        x_centers_px = (x_grid + 0.5) * self.patch_size_px
        y_centers_px = (y_grid + 0.5) * self.patch_size_px
        
        dist_sq = (x_centers_px - self.center_x_px)**2 + (y_centers_px - self.center_y_px)**2
        radius_px_sq = (self.radius_m / self.m_per_pixel)**2
        
        return dist_sq <= radius_px_sq

    def update_motion(self, vio_delta_m: np.ndarray, epsilon_m: float):
        """
        Updates the state based on VIO motion.
        1. Shifts the confidence circle center.
        2. Shifts the probability grid to model movement.
        3. Blurs the grid to account for motion uncertainty.
        4. Grows the confidence circle radius.
        """
        # 1. Shift the confidence circle center by the VIO delta
        delta_px = vio_delta_m / self.m_per_pixel
        self.center_x_px += delta_px[0]
        self.center_y_px += delta_px[1]

        # 2. Shift the entire probability grid to reflect the drone's movement
        shift_y_grid = delta_px[1] / self.patch_size_px
        shift_x_grid = delta_px[0] / self.patch_size_px

        M = np.float32([[1, 0, shift_x_grid], [0, 1, shift_y_grid]])
        self.prob_grid = cv2.warpAffine(self.prob_grid, M, (self.grid_w, self.grid_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        # 3. Blur the now-shifted grid to account for motion uncertainty (diffusion)
        std_dev_m = self.config.MOTION_UNCERTAINTY_STD_DEV_M + epsilon_m
        std_dev_grid = std_dev_m / (self.m_per_pixel * self.patch_size_px)
        
        kernel_size = int(std_dev_grid * 6)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if kernel_size > 1:
            self.prob_grid = cv2.GaussianBlur(self.prob_grid, (kernel_size, kernel_size), 0)

        # 4. Grow the confidence radius by the VIO error magnitude, with a cap.
        self.radius_m += epsilon_m
        self.radius_m = min(self.radius_m, 1500.0)
        
        self._normalize_probabilities()

    def update_measurement(self, current_drone_image: Image.Image, embedding_db):
        """
        Updates the probability grid based on a new image measurement.
        This now uses a k-NN search via FAISS for efficiency.
        """
        # 1. Get embedding for the current drone view
        current_embedding = embedding_db.get_embedding(current_drone_image)

        # 2. Find the k most similar embeddings
        k = 50
        distances, candidate_indices, _ = embedding_db.get_closest_embeddings(current_embedding, k)
        
        if len(candidate_indices) == 0:
            return

        # 3. Calculate likelihoods and update the grid
        likelihoods = self._calculate_likelihood(distances)
        
        likelihood_grid = np.zeros_like(self.prob_grid)
        gy_coords = embedding_db.grid_coordinates[candidate_indices, 0]
        gx_coords = embedding_db.grid_coordinates[candidate_indices, 1]

        np.maximum.at(likelihood_grid, (gy_coords, gx_coords), likelihoods)

        # Bayesian update: apply likelihood
        self.prob_grid *= likelihood_grid
        
        # We only trust the probabilities inside the circle
        mask = self._get_circle_mask()
        self.prob_grid[~mask] = 0 

        self._normalize_probabilities()
        
        # After measurement, re-center the confidence circle on the new peak probability
        new_center_x_px, new_center_y_px = self.get_most_likely_position_px()
        
        if new_center_x_px is not None:
             self.center_x_px, self.center_y_px = new_center_x_px, new_center_y_px
        
        # Shrink the circle only if we have a high confidence match AND the circle is very large
        max_prob = np.max(self.prob_grid) if self.prob_grid.size > 0 else 0
        if self.radius_m > 1000 and max_prob > 0.05:
            self.radius_m = 250.0
            print(f"High confidence point found. Shrinking radius to {self.radius_m}m.")

    def _calculate_likelihood(self, distances: np.ndarray) -> np.ndarray:
        """
        Converts embedding distances into likelihoods using a Gaussian distribution.
        """
        likelihood = norm.pdf(distances, loc=0, scale=self.config.LIKELIHOOD_STD_DEV)
        return likelihood

    def get_most_likely_position_px(self):
        """Finds the grid cell with the highest probability and returns its center in pixels."""
        if self.prob_grid.size == 0 or np.sum(self.prob_grid) < 1e-9:
            return None, None

        max_idx_flat = np.argmax(self.prob_grid)
        gy, gx = np.unravel_index(max_idx_flat, self.prob_grid.shape)

        px_x = gx * self.patch_size_px + self.patch_size_px // 2
        px_y = gy * self.patch_size_px + self.patch_size_px // 2
        return px_x, px_y

    def _normalize_probabilities(self):
        """Ensures the sum of the probability grid is 1."""
        total_prob = np.sum(self.prob_grid)
        if total_prob > 0:
            self.prob_grid /= total_prob
        else:
            self.prob_grid.fill(0)
            mask = self._get_circle_mask()
            num_cells_in_circle = np.sum(mask)
            if num_cells_in_circle > 0:
                self.prob_grid[mask] = 1.0 / num_cells_in_circle

    def get_active_patches(self):
        """Returns grid coordinates of patches with non-zero probability."""
        return np.argwhere(self.prob_grid > 1e-9) 