import numpy as np
import cv2
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from PIL import Image
from geolocalization import config

class LocalizationState:
    """
    Manages the probabilistic localization state using a confidence circle and
    implements the algorithm from the paper reference with VIO prediction and
    image measurement updates.
    """
    def __init__(self, database, initial_world_pos_m: tuple):
        self.database = database
        self.m_per_pixel = config.M_PER_PIXEL
        self.patch_size_m = config.GRID_PATCH_SIZE_M
        self.patch_size_px = self.database.patch_size_px
        
        # Convert initial position to world coordinates
        self.center_world_m = np.array(initial_world_pos_m, dtype=np.float64)
        self.radius_m = config.INITIAL_RADIUS_M

        # Probability grid for patches within the confidence circle
        # We use a dictionary to store probabilities for active patches only
        self.patch_probabilities = {}  # key: (grid_row, grid_col), value: probability
        
        # Initialize probabilities within the initial circle
        self._initialize_circle_probabilities()
        
        # VIO prediction parameters from paper
        self.vio_x_variance = config.VIO_X_VARIANCE
        self.vio_y_variance = config.VIO_Y_VARIANCE
        
        print(f"Localization initialized at {initial_world_pos_m}, radius: {self.radius_m}m")

    def _initialize_circle_probabilities(self):
        """Initialize uniform probabilities for patches within the confidence circle."""
        self.patch_probabilities.clear()
        
        # Find all patches within the circle
        patch_indices, _ = self.database.get_embeddings_in_circle(
            tuple(self.center_world_m), self.radius_m
        )
        
        if len(patch_indices) > 0:
            # Uniform probability distribution
            prob_per_patch = 1.0 / len(patch_indices)
            for idx in patch_indices:
                grid_coord = self.database.get_grid_coordinates(idx)
                self.patch_probabilities[tuple(grid_coord)] = prob_per_patch
        
        print(f"Initialized {len(self.patch_probabilities)} patches with probabilities")

    def _update_circle_patches(self):
        """
        Update the set of patches within the confidence circle.
        Only include patches that are within valid map bounds.
        """
        # Clear current probabilities
        self.patch_probabilities.clear()
        
        # Get map bounds for validation
        map_w_px = self.database.map_w
        map_h_px = self.database.map_h
        patch_size_px = self.database.patch_size_px
        
        # Calculate valid grid bounds
        max_grid_col = map_w_px // patch_size_px
        max_grid_row = map_h_px // patch_size_px
        
        # Convert center position to grid coordinates
        center_px = self.database.world_to_pixel(self.center_world_m[0], self.center_world_m[1])
        center_grid_col = center_px[0] // patch_size_px
        center_grid_row = center_px[1] // patch_size_px
        
        # Calculate radius in grid coordinates
        radius_grid = self.radius_m / self.patch_size_m
        
        # Find all patches within the circle that are also within map bounds
        grid_radius_int = int(np.ceil(radius_grid))
        
        patches_added = 0
        for dr in range(-grid_radius_int, grid_radius_int + 1):
            for dc in range(-grid_radius_int, grid_radius_int + 1):
                # Calculate grid coordinates
                grid_row = int(center_grid_row + dr)
                grid_col = int(center_grid_col + dc)
        
                # Check if patch is within map bounds
                if (grid_row < 0 or grid_row >= max_grid_row or 
                    grid_col < 0 or grid_col >= max_grid_col):
                    continue  # Skip patches outside map bounds
                
                # Check if patch is within circle
                distance_grid = np.sqrt(dr*dr + dc*dc)
                if distance_grid <= radius_grid:
                    # Add patch with uniform probability (will be normalized later)
                    self.patch_probabilities[(grid_row, grid_col)] = 1.0
                    patches_added += 1
        
        # Normalize probabilities
        if patches_added > 0:
            uniform_prob = 1.0 / patches_added
            for coord in self.patch_probabilities:
                self.patch_probabilities[coord] = uniform_prob
        
        print(f"Updated circle: {patches_added} valid patches within radius {self.radius_m:.1f}m")

    def update_motion_prediction(self, vio_delta_m: np.ndarray, epsilon_m: float):
        """
        Implements the motion prediction step from the paper using VIO measurements.
        Uses 1D convolutions for x and y directions as described in equations (9a-9c).
        """
        # Update circle center based on VIO measurement
        self.center_world_m += vio_delta_m
        
        # Increase circle radius by epsilon (Q3 correction)
        self.radius_m += epsilon_m
        self.radius_m = min(self.radius_m, config.MAX_RADIUS_M)
        
        # Update the set of active patches
        self._update_circle_patches()
        
        # Apply VIO prediction to probability distribution
        # This implements the convolution-based approach from the paper
        if self.patch_probabilities:
            self._apply_vio_prediction_convolution(vio_delta_m, epsilon_m)
        
        print(f"Motion update: center={self.center_world_m}, radius={self.radius_m:.1f}m, "
              f"active_patches={len(self.patch_probabilities)}")

    def _apply_vio_prediction_convolution(self, vio_delta_m: np.ndarray, epsilon_m: float):
        """
        Apply VIO prediction using 1D convolutions as described in the paper.
        This spreads probability mass according to motion uncertainty.
        """
        if not self.patch_probabilities:
            return
        
        # Convert probabilities to a dense grid for convolution
        # Find bounds of active region
        coords = list(self.patch_probabilities.keys())
        if not coords:
            return
            
        min_row = min(coord[0] for coord in coords)
        max_row = max(coord[0] for coord in coords)
        min_col = min(coord[1] for coord in coords)
        max_col = max(coord[1] for coord in coords)
        
        # Create dense grid
        grid_h = max_row - min_row + 1
        grid_w = max_col - min_col + 1
        prob_grid = np.zeros((grid_h, grid_w))
        
        # Fill grid with current probabilities
        for coord, prob in self.patch_probabilities.items():
            row_idx = coord[0] - min_row
            col_idx = coord[1] - min_col
            prob_grid[row_idx, col_idx] = prob
        
        # Calculate motion offset in grid coordinates
        motion_x_grid = vio_delta_m[0] / self.patch_size_m
        motion_y_grid = vio_delta_m[1] / self.patch_size_m
        
        # Apply translation (shift the probability mass)
        if abs(motion_x_grid) > 0.1 or abs(motion_y_grid) > 0.1:
            shift_matrix = np.float32([[1, 0, motion_x_grid], [0, 1, motion_y_grid]])
            prob_grid = cv2.warpAffine(prob_grid, shift_matrix, (grid_w, grid_h), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Apply Gaussian blur to model motion uncertainty (sigma based on epsilon)
        motion_uncertainty_grid = (epsilon_m + config.VIO_X_VARIANCE) / self.patch_size_m
        if motion_uncertainty_grid > 0.1:
            prob_grid = gaussian_filter(prob_grid, sigma=motion_uncertainty_grid)
        
        # Update probabilities dictionary
        self.patch_probabilities.clear()
        for row in range(grid_h):
            for col in range(grid_w):
                if prob_grid[row, col] > 1e-9:
                    coord = (row + min_row, col + min_col)
                    self.patch_probabilities[coord] = prob_grid[row, col]
        
        self._normalize_probabilities()

    def update_measurement(self, drone_camera_view: Image.Image):
        """
        Updates probabilities based on image measurement using embedding similarity.
        Implements proper Bayesian measurement update.
        """
        if not self.patch_probabilities:
            return
        
        # Get embedding for current camera view
        current_embedding = self.database.get_embedding(drone_camera_view)

        # Find top-k similar patches in the database
        distances, indices, _ = self.database.get_closest_embeddings(
            current_embedding, config.TOP_K_MATCHES
        )
        
        if len(indices) == 0:
            print("Warning: No embedding matches found")
            return

        # Create a mapping from grid coordinates to similarity scores
        grid_similarities = {}
        for distance, patch_idx in zip(distances, indices):
            # Convert database index to grid coordinates
            grid_coord = tuple(self.database.get_grid_coordinates(patch_idx))
            
            # Convert distance to similarity (closer = higher similarity)
            similarity = np.exp(-distance * 2.0)  # More sensitive to differences
            
            # Use maximum similarity if multiple matches map to same grid coordinate
            if grid_coord in grid_similarities:
                grid_similarities[grid_coord] = max(grid_similarities[grid_coord], similarity)
            else:
                grid_similarities[grid_coord] = similarity

        # Calculate likelihoods for all active patches
        likelihoods = {}
        max_similarity = max(grid_similarities.values()) if grid_similarities else 1.0
        
        for patch_coord in self.patch_probabilities.keys():
            if patch_coord in grid_similarities:
                # Normalize and amplify differences
                normalized_sim = grid_similarities[patch_coord] / max_similarity
                likelihood = normalized_sim ** 3  # Amplify differences more
            else:
                # Much lower likelihood for non-matching patches
                likelihood = 0.01  # Strong penalty for no match
            
            likelihoods[patch_coord] = likelihood

        # Apply Bayesian update: P(patch|measurement) ∝ P(measurement|patch) * P(patch)
        updated_probabilities = {}
        for patch_coord, prior_prob in self.patch_probabilities.items():
            if patch_coord in likelihoods:
                updated_probabilities[patch_coord] = likelihoods[patch_coord] * prior_prob
            else:
                updated_probabilities[patch_coord] = 0.01 * prior_prob

        # Normalize to ensure probabilities sum to 1
        total_prob = sum(updated_probabilities.values())
        if total_prob > 0:
            for patch_coord in updated_probabilities:
                updated_probabilities[patch_coord] /= total_prob
        
        # Update probabilities
        self.patch_probabilities = updated_probabilities
        
        print(f"Measurement update: {len(grid_similarities)} grid matches found, "
              f"max similarity: {max_similarity:.3f}")
        
        # Debug: Print top probabilities to verify discrimination
        sorted_probs = sorted(self.patch_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_probs = sorted_probs[:5]
        prob_values = [f"{prob:.4f}" for _, prob in top_probs]
        print(f"Top 5 patch probabilities: {prob_values}")
        
        # Check if we have good discrimination (top probability should be much higher)
        if len(sorted_probs) > 1:
            ratio = sorted_probs[0][1] / sorted_probs[1][1] if sorted_probs[1][1] > 0 else float('inf')
            print(f"Discrimination ratio (1st/2nd): {ratio:.2f}")

    def _calculate_likelihood(self, distances: np.ndarray) -> np.ndarray:
        """Convert embedding distances to likelihoods using Gaussian distribution."""
        # Use L2 distances - lower distance = higher likelihood
        # Apply a more selective conversion to create better discrimination
        likelihoods = np.exp(-distances**2 / (2 * config.LIKELIHOOD_STD_DEV**2))
        
        # Normalize to [0, 1] range for better interpretation
        if len(likelihoods) > 0:
            min_val = np.min(likelihoods)
            max_val = np.max(likelihoods)
            if max_val > min_val:
                likelihoods = (likelihoods - min_val) / (max_val - min_val)
        
        return likelihoods

    def _check_correction_trigger(self):
        """
        Check if conditions are met to trigger a position correction.
        Triggers correction if radius is large and we have confident localization.
        """
        if self.radius_m < config.CORRECTION_THRESHOLD_M:
            return False
        
        if not self.patch_probabilities:
            return False
        
        # Calculate peak-to-average ratio (Q2)
        probs = list(self.patch_probabilities.values())
        max_prob = max(probs)
        avg_prob = np.mean(probs)
        peak_to_avg_ratio = max_prob / avg_prob if avg_prob > 0 else 0
        
        # Trigger correction if we have high confidence (peak-to-average ratio > threshold)
        confidence_threshold = 5.0  # Adjustable parameter
        
        if peak_to_avg_ratio > confidence_threshold:
            return True
        
        return False

    def get_most_confident_position(self) -> tuple:
        """
        Returns the world position of the most confident patch.
        This is used for triggering corrections.
        """
        if not self.patch_probabilities:
            return tuple(self.center_world_m)
        
        # Find patch with highest probability
        best_coord = max(self.patch_probabilities.keys(), 
                        key=lambda k: self.patch_probabilities[k])
        
        # Convert grid coordinates to world coordinates
        # Calculate world position directly from grid coordinates
        row, col = best_coord
        
        # Convert grid coordinates to pixel coordinates
        center_x_px = (col + 0.5) * self.database.patch_size_px
        center_y_px = (row + 0.5) * self.database.patch_size_px
        
        # Convert pixel coordinates to world coordinates
        world_pos = self.database.pixel_to_world(center_x_px, center_y_px)
        return world_pos

    def apply_correction(self, new_center_world_m: tuple):
        """
        Apply position correction by moving the confidence circle and shrinking radius.
        """
        self.center_world_m = np.array(new_center_world_m, dtype=np.float64)
        
        # Shrink radius after successful correction
        self.radius_m = max(config.MIN_RADIUS_M, self.radius_m * 0.5)
        
        # Update patches in the new circle
        self._update_circle_patches()
        
        print(f"Correction applied: new center={new_center_world_m}, new radius={self.radius_m:.1f}m")

    def _normalize_probabilities(self):
        """Normalize probabilities to sum to 1."""
        if not self.patch_probabilities:
            return
        
        total_prob = sum(self.patch_probabilities.values())
        if total_prob > 0:
            for coord in self.patch_probabilities:
                self.patch_probabilities[coord] /= total_prob

    def get_active_patches_for_visualization(self) -> tuple:
        """
        Returns data for visualization: coordinates, probabilities, and circle info.
        """
        if not self.patch_probabilities:
            return np.array([]), np.array([]), self.center_world_m, self.radius_m
        
        coords = np.array(list(self.patch_probabilities.keys()))
        probs = np.array(list(self.patch_probabilities.values()))
        
        return coords, probs, self.center_world_m, self.radius_m

    def get_confidence_metrics(self) -> dict:
        """Return metrics about the current localization confidence."""
        if not self.patch_probabilities:
            return {"entropy": float('inf'), "peak_to_avg": 0, "num_patches": 0}
        
        probs = np.array(list(self.patch_probabilities.values()))
        
        # Calculate entropy (lower = more confident)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Calculate peak-to-average ratio (higher = more confident)
        peak_to_avg = np.max(probs) / np.mean(probs) if len(probs) > 0 else 0
        
        return {
            "entropy": entropy,
            "peak_to_avg": peak_to_avg,
            "num_patches": len(self.patch_probabilities),
            "radius_m": self.radius_m
        } 