import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d, shift
from scipy.stats import norm
import cv2
from tqdm import tqdm
import random
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from train_encoder import SiameseNet, get_eval_transforms
from simple_superpoint import SuperPoint

def match_descriptors(desc1, desc2, threshold=0.8):
    """Match descriptors using ratio test. Duplicated from visualize_superpoint_clean.py"""
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    # Compute distances
    dists = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)
    
    # Find best and second best matches for each descriptor in desc1
    matches = []
    for i in range(len(desc1)):
        sorted_indices = np.argsort(dists[i])
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1] if len(sorted_indices) > 1 else best_idx
        
        best_dist = dists[i, best_idx]
        second_best_dist = dists[i, second_best_idx]
        
        # Ratio test
        if best_dist < threshold * second_best_dist:
            matches.append((i, best_idx, best_dist))
    
    return matches


class NewGlobalLocalizer:
    """
    Implements the new global GPS-denied geolocalization algorithm with:
    1. Dynamic patch-based database (only for confidence circle)
    2. Circle-based probability tracking 
    3. VIO prediction with 1D convolutions
    4. Correction triggers based on circle radius
    """
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Load map image
        self.map_image = Image.open(config.MAP_IMAGE_PATH).convert('RGB')
        self.map_w, self.map_h = self.map_image.size
        self.m_per_pixel = config.M_PER_PIXEL
        self.patch_size_m = config.GRID_PATCH_SIZE_M
        self.patch_size_px = int(self.patch_size_m / self.m_per_pixel)
        
        # Load trained encoder model
        self.model = self._load_model()
        self.eval_transforms = get_eval_transforms()

        # Load SuperPoint model
        print(f"Loading SuperPoint model from: {self.config.SUPERPOINT_WEIGHTS_PATH}")
        self.superpoint = SuperPoint(self.config.SUPERPOINT_WEIGHTS_PATH)
        
        # Initialize confidence circle and probability map
        self.center_world_m = np.array([0.0, 0.0], dtype=np.float64)  # Will be set by drone
        self.radius_m = config.INITIAL_RADIUS_M
        self.patch_probabilities = {}  # {(grid_row, grid_col): probability}
        self.patch_embeddings = {}     # {(grid_row, grid_col): embedding}
        self.patch_images = {}         # {(grid_row, grid_col): PIL Image}
        
        # For visualization
        self.last_camera_view = None
        self.last_superpoint_vis_data = None # Stores best SP match image, kpts, matches
        self.last_top5_patches = [] # Stores coords, probs, embeddings, images for top 5 candidates
        
        print(f"GlobalLocalizer initialized with {self.map_w}x{self.map_h} map")

    def _load_model(self):
        """Load the trained SiameseNet model."""
        model = SiameseNet(self.config.BACKBONE_NAME).to(self.device)
        
        if Path(self.config.MODEL_WEIGHTS_PATH).exists():
            print(f"Loading model weights from {self.config.MODEL_WEIGHTS_PATH}")
            try:
                # Load checkpoint with weights_only=True for security
                checkpoint = torch.load(self.config.MODEL_WEIGHTS_PATH, map_location=self.device, weights_only=True)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                
                # Filter out unexpected keys that don't match the model
                model_keys = set(model.state_dict().keys())
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                
                # Load the filtered state dict with strict=False to ignore missing keys
                missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                
                if missing_keys:
                    print(f"Warning: Missing keys in checkpoint: {missing_keys}")
                if unexpected_keys:
                    print(f"Info: Ignoring unexpected keys: {unexpected_keys}")
                    
                print("Model weights loaded successfully!")
                
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Using untrained model.")
        else:
            print("Warning: No trained model found. Using untrained model.")
        
        model.eval()
        return model

    def initialize_circle(self, start_world_pos_m: tuple):
        """Initialize the confidence circle at the starting position."""
        self.center_world_m = np.array(start_world_pos_m, dtype=np.float64)
        self.radius_m = self.config.INITIAL_RADIUS_M
        
        # Build initial database for patches in circle
        self._update_circle_database()
        
        # Initialize uniform probabilities
        if self.patch_probabilities:
            uniform_prob = 1.0 / len(self.patch_probabilities)
            for coord in self.patch_probabilities:
                self.patch_probabilities[coord] = uniform_prob
        
        print(f"Circle initialized at {start_world_pos_m}, radius: {self.radius_m}m, "
              f"patches: {len(self.patch_probabilities)}")

    def _update_circle_database(self):
        """Update the patch database for current circle position and radius."""
        # Find all patches within the circle
        circle_patches = self._get_patches_in_circle()
        
        # Remove embeddings, images for patches no longer in circle
        current_coords = set(self.patch_probabilities.keys())
        new_coords = set(circle_patches)
        
        # Set probabilities to ZERO for patches that exit circle, and remove associated data
        for coord in current_coords - new_coords:
            self.patch_probabilities[coord] = 0.0  # Set probability to zero
            # Optionally remove other data if memory is a concern, but keep probability entry for now for normalization
            self.patch_embeddings.pop(coord, None)
            self.patch_images.pop(coord, None)
        
        # Add new patches and compute embeddings and store images
        patches_to_add = new_coords - current_coords
        if patches_to_add:
            self._compute_embeddings_for_patches(patches_to_add)
            
            # Set probabilities for new patches based on neighbors
            self._set_new_patch_probabilities(patches_to_add)

    def _get_patches_in_circle(self) -> list:
        """Get grid coordinates of all patches within the confidence circle."""
        patches = []
        
        # Calculate grid bounds
        radius_patches = math.ceil(self.radius_m / self.patch_size_m)
        center_grid = self._world_to_grid(self.center_world_m[0], self.center_world_m[1])
        
        for dr in range(-radius_patches, radius_patches + 1):
            for dc in range(-radius_patches, radius_patches + 1):
                grid_row = center_grid[0] + dr
                grid_col = center_grid[1] + dc
                
                # Check if patch is within map bounds
                if (0 <= grid_row < self.map_h // self.patch_size_px and 
                    0 <= grid_col < self.map_w // self.patch_size_px):
                    
                    # Check if patch center is within circle
                    patch_center = self._grid_to_world(grid_row, grid_col)
                    distance = np.linalg.norm(np.array(patch_center) - self.center_world_m)
                    
                    if distance <= self.radius_m:
                        patches.append((grid_row, grid_col))
        
        return patches

    def _compute_embeddings_for_patches(self, patch_coords: set):
        """Compute embeddings and store images for a set of patch coordinates."""
        if not patch_coords:
            return
            
        patches_batch = []
        coords_batch = []
        images_to_store = {}
        
        for grid_row, grid_col in patch_coords:
            # Extract patch from map
            x_px = grid_col * self.patch_size_px
            y_px = grid_row * self.patch_size_px
            
            # Ensure bounds
            x_end = min(x_px + self.patch_size_px, self.map_w)
            y_end = min(y_px + self.patch_size_px, self.map_h)
            
            patch_image = self.map_image.crop((x_px, y_px, x_end, y_end))
            
            # Store the original-sized cropped image
            images_to_store[(grid_row, grid_col)] = patch_image.copy()

            # Resize for model input
            patch_image_resized = patch_image.resize((self.config.CROP_SIZE_PX, self.config.CROP_SIZE_PX), 
                                           Image.Resampling.LANCZOS)
            
            patches_batch.append(self.eval_transforms(patch_image_resized))
            coords_batch.append((grid_row, grid_col))
        
        # Compute embeddings in batch
        with torch.no_grad():
            batch_tensor = torch.stack(patches_batch).to(self.device)
            embeddings = self.model.get_embedding(batch_tensor).cpu().numpy()
            
            for i, coord in enumerate(coords_batch):
                self.patch_embeddings[coord] = embeddings[i].astype('float32')
                self.patch_images[coord] = images_to_store[coord] # Store the image after embedding computation

    def _set_new_patch_probabilities(self, new_patches: set):
        """Set probabilities for new patches based on neighbors (Q1 requirement)."""
        for coord in new_patches:
            # Find neighboring patches with existing probabilities
            neighbor_probs = []
            row, col = coord
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbor_coord = (row + dr, col + dc)
                    if neighbor_coord in self.patch_probabilities:
                        neighbor_probs.append(self.patch_probabilities[neighbor_coord])
            
            # Set probability based on neighbors
            if neighbor_probs:
                # Set new probability to be average of neighbors, scaled by a factor
                self.patch_probabilities[coord] = np.mean(neighbor_probs) * self.config.NEW_PATCH_PROBABILITY_FACTOR
            else:
                # If no neighbors, set to a small default probability
                self.patch_probabilities[coord] = self.config.NEW_PATCH_DEFAULT_PROB

    def update_motion_prediction(self, vio_delta_m: np.ndarray, epsilon_m: float):
        """
        Update state using VIO measurements and motion prediction.
        Implements 1D convolutions as described in the paper.
        NOTE: Circle center is NOT updated with VIO - it should be centered on ground truth
        """
        # DON'T update circle center with VIO - circle stays centered on ground truth
        # self.center_world_m += vio_delta_m  # REMOVED - this was the bug
        
        # Update confidence circle radius based on VIO error
        self.radius_m = min(self.config.MAX_RADIUS_M, self.radius_m + epsilon_m * self.config.RADIUS_GROWTH_FACTOR)
        
        # Ensure the circle database is up-to-date with new radius/center
        self._update_circle_database() # This will add/remove patches and reset their probabilities accordingly
        
        # Apply VIO prediction using 1D convolutions
        if self.patch_probabilities:
            self._apply_vio_prediction_convolution(vio_delta_m) # epsilon_m is no longer passed for sigma
        
        # Normalize probabilities
        self._normalize_probabilities()

    def _apply_vio_prediction_convolution(self, vio_delta_m: np.ndarray): # Removed epsilon_m argument
        """Apply VIO prediction using 1D convolutions."""
        if not self.patch_probabilities:
            return
        
        coords = list(self.patch_probabilities.keys())
        if not coords:
            return
        
        # Create dense grid for convolution
        min_row = min(coord[0] for coord in coords)
        max_row = max(coord[0] for coord in coords)
        min_col = min(coord[1] for coord in coords)
        max_col = max(coord[1] for coord in coords)
        
        grid_h = max_row - min_row + 1
        grid_w = max_col - min_col + 1
        prob_grid = np.zeros((grid_h, grid_w))
        
        # Fill grid
        for coord, prob in self.patch_probabilities.items():
            row_idx = coord[0] - min_row
            col_idx = coord[1] - min_col
            prob_grid[row_idx, col_idx] = prob
        
        # Calculate motion offset in grid coordinates
        motion_x_grid = vio_delta_m[0] / self.patch_size_m
        motion_y_grid = vio_delta_m[1] / self.patch_size_m
        
        # Apply translation using scipy.ndimage.shift
        # The shift represents the mean of the distribution, so we shift by -motion_grid
        # as scipy.ndimage.shift shifts the input array, not the coordinate system.
        # Order 1 for linear interpolation. Boundary condition: fill with zeros.
        prob_grid = shift(prob_grid, (-motion_y_grid, -motion_x_grid), mode='constant', cval=0.0, order=1)
        
        # Apply Gaussian blur for motion uncertainty
        # Use VIO_ERROR_STD_M for the standard deviation of the Gaussian filter
        # We assume isotropic noise for the blur kernel for x and y
        sigma_grid_units = self.config.VIO_ERROR_STD_M / self.patch_size_m

        if sigma_grid_units > 0.01: # Use a small threshold to avoid very small sigmas
            # Convolve along columns (x-direction)
            prob_grid = gaussian_filter1d(prob_grid, sigma=sigma_grid_units, axis=1, mode='constant', cval=0.0)
            # Convolve along rows (y-direction)
            prob_grid = gaussian_filter1d(prob_grid, sigma=sigma_grid_units, axis=0, mode='constant', cval=0.0)
        
        # Update probabilities dictionary
        self.patch_probabilities.clear()
        for row in range(grid_h):
            for col in range(grid_w):
                # Convert back to original grid coordinates
                    coord = (row + min_row, col + min_col)
                if prob_grid[row, col] > self.config.MIN_PROBABILITY_THRESHOLD and coord in self.patch_embeddings:
                        self.patch_probabilities[coord] = prob_grid[row, col]

        # After convolution, prune very low probabilities to manage memory
        # Patches that exit the circle have their probabilities set to 0 in _update_circle_database
        # We only need to prune based on the probability threshold here.
        coords_to_delete = [coord for coord, prob in self.patch_probabilities.items() if prob < self.config.MIN_PROBABILITY_THRESHOLD]
        for coord in coords_to_delete:
            self.patch_probabilities.pop(coord)
            self.patch_embeddings.pop(coord, None)
            self.patch_images.pop(coord, None)

        self._normalize_probabilities()

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV (BGR) format."""
        # Convert PIL image to NumPy array
        np_image = np.array(pil_image)
        
        # If it's RGB, convert to BGR (OpenCV default)
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        # If it's grayscale, keep as is (SuperPoint expects grayscale too)
        elif len(np_image.shape) == 2:
            return np_image
        else:
            raise ValueError("Unsupported image format for conversion.")

    def update_measurement(self, camera_image: Image.Image):
        """Update probabilities based on camera image measurement and store top-5 retrievals for visualization."""
        if not self.patch_probabilities:
            print("No patch probabilities - skipping measurement update")
            return
        
        print(f"Measurement update: {len(self.patch_probabilities)} patches")
        
        # Store initial probabilities for comparison
        initial_probs = dict(self.patch_probabilities)
        
        # Store the current camera view for visualization
        self.last_camera_view = camera_image.copy()
        
        # Get embedding for camera image
        camera_image_resized = camera_image.resize((self.config.CROP_SIZE_PX, self.config.CROP_SIZE_PX),
                                                  Image.Resampling.LANCZOS)
        
        with torch.no_grad():
            img_tensor = self.eval_transforms(camera_image_resized).unsqueeze(0).to(self.device)
            query_embedding = self.model.get_embedding(img_tensor).squeeze().cpu().numpy()
        
        # Calculate likelihood for each patch in circle
        distances = {}
        for coord in self.patch_probabilities:
            if coord in self.patch_embeddings:
                patch_embedding = self.patch_embeddings[coord]
                # Calculate L2 distance
                distance = np.linalg.norm(query_embedding - patch_embedding)
                distances[coord] = distance

        if not distances:
            print("No patches with embeddings found in circle. Skipping measurement update.")
            return
        
        min_embedding_distance = min(distances.values())
        print(f"Embedding Distance range: {min_embedding_distance:.4f} to {max(distances.values()):.4f}")
        
        # Update probabilities based on embedding similarity (lower distance = higher likelihood)
            # Use temperature parameter to control sharpness of probability distribution
            temperature = 0.5  # Lower = sharper distribution
            
            likelihoods = {}
            for coord in self.patch_probabilities:
                if coord in distances:
                    distance = distances[coord]
                    # Convert to likelihood using exponential with temperature
                # Subtract min_embedding_distance for numerical stability
                likelihood = np.exp(-(distance - min_embedding_distance) / temperature)
                    likelihoods[coord] = likelihood
                    # Update probability (multiply prior by likelihood)
                    self.patch_probabilities[coord] *= likelihood
            
        print(f"Likelihood range (embedding): {min(likelihoods.values()):.6f} to {max(likelihoods.values()):.6f}")
        
        # --- SuperPoint Matching and Probability Update --- 
        # Store SP results for top 5 candidates for detailed application
        sp_results_top5 = {} # {(coord): avg_desc_dist}
        all_top5_sp_avg_dists = [] # To find the max among successful top-5 matches

        sorted_by_embedding_dist = sorted(distances.items(), key=lambda item: item[1])
        top_5_candidates_coords = [item[0] for item in sorted_by_embedding_dist[:5]]

        camera_image_cv2 = self._pil_to_cv2(camera_image.convert('L'))

        self.last_top5_patches = [] # Reset for current visualization

        # First pass: Get SP matches for top 5 candidates and find the max_avg_desc_dist_top5
        best_sp_overall_dist = float('inf')
        best_sp_overall_coord = None
        best_sp_overall_kpts_camera = None
        best_sp_overall_kpts_patch = None
        best_sp_overall_matches = None
        best_sp_overall_patch_image = None

        for rank, coord in enumerate(top_5_candidates_coords):
            if coord not in self.patch_images:
                continue # Should not happen if _update_circle_database is correct

            patch_image_pil = self.patch_images[coord]
            patch_image_cv2 = self._pil_to_cv2(patch_image_pil.convert('L'))

            # Store for visualization even if not the best SP match
            self.last_top5_patches.append({
                'coord': coord,
                'prob': self.patch_probabilities.get(coord, 0.0), # Get current prob
                'embedding_dist': distances.get(coord, float('inf')),
                'image': patch_image_pil
            })

            # Detect SuperPoint keypoints and descriptors
            kpts_camera, scores_camera, desc_camera = self.superpoint.detect(camera_image_cv2)
            kpts_patch, scores_patch, desc_patch = self.superpoint.detect(patch_image_cv2)

            # Match descriptors
            sp_matches = match_descriptors(desc_camera, desc_patch)

            if sp_matches:
                current_sp_avg_dist = np.mean([m[2] for m in sp_matches])
                sp_results_top5[coord] = current_sp_avg_dist
                all_top5_sp_avg_dists.append(current_sp_avg_dist)

                print(f"SuperPoint for {coord} (Rank {rank+1}): Matches {len(sp_matches)}, Avg Desc Dist: {current_sp_avg_dist:.3f}")

                # Keep track of the overall best SP match for visualization
                if current_sp_avg_dist < best_sp_overall_dist:
                    best_sp_overall_dist = current_sp_avg_dist
                    best_sp_overall_coord = coord
                    best_sp_overall_kpts_camera = kpts_camera
                    best_sp_overall_kpts_patch = kpts_patch
                    best_sp_overall_matches = sp_matches
                    best_sp_overall_patch_image = patch_image_pil
            else:
                print(f"SuperPoint for {coord} (Rank {rank+1}): No matches found.")

        # Determine the highest 'd' from top 5 successful SP matches
        max_avg_desc_dist_top5_or_default = 1.0 # Default if no SP matches in top 5 (leads to 0.5 likelihood)
        if all_top5_sp_avg_dists:
            max_avg_desc_dist_top5_or_default = max(all_top5_sp_avg_dists)

        # Update SuperPoint visualization data with the overall best match
        if best_sp_overall_coord is not None:
            self.last_superpoint_vis_data = {
                'best_match_image': best_sp_overall_patch_image,
                'kpts_camera': best_sp_overall_kpts_camera,
                'kpts_patch': best_sp_overall_kpts_patch,
                'matches': best_sp_overall_matches,
                'avg_desc_dist': best_sp_overall_dist
            }
        else:
            self.last_superpoint_vis_data = None
            print("No strong SP match found among top 5 candidates. Skipping SP visualization.")

        # Second pass: Apply SuperPoint likelihood to ALL active patches
        for coord in list(self.patch_probabilities.keys()): # Iterate over a copy as dict will be modified
            if coord not in self.patch_embeddings: # Skip if embedding is somehow missing
                continue

            c = 0.0
            if coord in sp_results_top5: # This patch was a top-5 and had a successful SP match
                c = sp_results_top5[coord]
            else: # For non-top5 patches, or top5 without SP matches, use the max from top5 or default
                c = max_avg_desc_dist_top5_or_default

            c = max(0.0, min(1.0, c)) # Clamp c to [0, 1]
            superpoint_likelihood = (2.0 - c) / 2.0

            if self.patch_probabilities[coord] > 0: # Only update if current probability is positive
                self.patch_probabilities[coord] *= superpoint_likelihood
            # If probability is 0, leave it as 0 (it was likely pruned or never active) 

        # Normalize probabilities after all updates
        self._normalize_probabilities()
        
        # Show how probabilities changed
        final_probs = dict(self.patch_probabilities)
        prob_changes = [(coord, initial_probs[coord], final_probs[coord]) 
                       for coord in initial_probs if coord in final_probs]
        
        # Show a few examples
        if prob_changes:
            print("Probability changes (coord, before, after):")
            # Sort by absolute change for more informative examples
            prob_changes_sorted = sorted(prob_changes, key=lambda x: abs(x[2] - x[1]), reverse=True)
            for i, (coord, before, after) in enumerate(prob_changes_sorted[:3]):
                print(f"  {coord}: {before:.6f} -> {after:.6f} (×{after/before:.2f})")
            if len(prob_changes_sorted) > 3:
                print(f"  ... and {len(prob_changes_sorted)-3} more")
        
        print(f"Final prob sum: {sum(self.patch_probabilities.values()):.6f}")
        print("Measurement update completed")

    def check_correction_trigger(self) -> tuple:
        """Check if correction should be triggered and return correction target."""
        # Q2: Peak-to-Average Ratio check (Implicitly handled by finding max_prob)
        # The primary trigger remains radius > CORRECTION_THRESHOLD_M

        if self.radius_m <= self.config.CORRECTION_THRESHOLD_M:
            return False, None
        
        # Find the most confident cluster (patch with highest probability)
        max_prob = 0
        best_coord = None
        
        for coord, prob in self.patch_probabilities.items():
            if prob > max_prob:
                max_prob = prob
                best_coord = coord
        
        if best_coord is None or max_prob < self.config.MIN_PROBABILITY_THRESHOLD:
            print("No confident cluster found for correction.")
            return False, None
        
        # Convert to world coordinates
        target_world = self._grid_to_world(best_coord[0], best_coord[1])
        
        # Check if correction distance is significant
        distance_to_center = np.linalg.norm(np.array(target_world) - self.center_world_m)
        
        if distance_to_center > self.config.MIN_CORRECTION_DISTANCE_M:  # Minimum correction distance
            print(f"Correction triggered: Radius {self.radius_m:.2f}m > {self.config.CORRECTION_THRESHOLD_M:.2f}m. "
                  f"Confident cluster at {best_coord} (prob: {max_prob:.6f}). Distance to center: {distance_to_center:.2f}m.")
            return True, target_world
        
        print(f"Correction not triggered: Distance to confident cluster ({distance_to_center:.2f}m) too small.")
        return False, None

    def apply_correction(self, correction_delta_m: np.ndarray):
        """Apply correction movement to the system."""
        # Move circle center
        self.center_world_m += correction_delta_m
        
        # After applying correction, reset radius to initial value
        self.radius_m = self.config.INITIAL_RADIUS_M
        self._update_circle_database() # Re-initialize database based on new center and initial radius
        self._normalize_probabilities()
        
        print(f"Correction applied. New center: {self.center_world_m}, New radius: {self.radius_m}")

    def _normalize_probabilities(self):
        """Normalize all probabilities to sum to 1."""
        if not self.patch_probabilities:
            return
        
        total_prob = sum(self.patch_probabilities.values())
        if total_prob > 0:
            for coord in self.patch_probabilities:
                self.patch_probabilities[coord] /= total_prob
        else:
            # If total_prob is zero, all probabilities are zero. Re-initialize uniformly.
            print("Warning: All probabilities are zero after normalization. Re-initializing uniformly.")
            if self.patch_probabilities: # Check if there are any patches in the dictionary
                uniform_prob = 1.0 / len(self.patch_probabilities) # Use all patches currently in dict
                for coord in self.patch_probabilities:
                    self.patch_probabilities[coord] = uniform_prob
            else:
                # If no patches at all, it's a critical state, re-initialize the entire circle database.
                print("Critical: No active patches after zeroing. Re-initializing circle.")
                # This is a fallback to re-establish a search space around the last known center
                current_vio_pos_m = self.center_world_m if hasattr(self, 'center_world_m') else np.array([0.0, 0.0])
                self.initialize_circle(tuple(current_vio_pos_m))

    def _world_to_grid(self, x_m: float, y_m: float) -> tuple:
        """Convert world coordinates to grid coordinates."""
        # Convert to pixel coordinates relative to map center
        # Q4: center of the image is (0, 0)
        # Map image has lat/lng center in filename. Assuming (0,0) world coordinates corresponds to map_w/2, map_h/2 pixels.
        x_px = x_m / self.m_per_pixel + self.map_w / 2.0
        y_px = y_m / self.m_per_pixel + self.map_h / 2.0
        
        # Convert to grid coordinates
        grid_col = int(x_px // self.patch_size_px)
        grid_row = int(y_px // self.patch_size_px)
        
        # Update the circle center for the next iteration based on the maximum probability patch
        max_prob_coord = max(self.patch_probabilities, key=self.patch_probabilities.get) if self.patch_probabilities else None
        if max_prob_coord:
            self.center_world_m = np.array(self._grid_to_world(max_prob_coord[0], max_prob_coord[1]))
            # Q4: The radius grows proportional to error. For now, reset to initial radius during correction
            # and grow when probability max is decreasing.
            # For now, always reset to INITIAL_RADIUS_M if a confident cluster is found,
            # otherwise let it grow up to MAX_RADIUS_M.
            
            # Determine if a confident cluster is found. If so, reset radius.
            # This logic should ideally be tied to `check_correction_trigger`
            # For now, let's keep radius consistent until correction is explicitly applied.
            # self.radius_m = self.config.INITIAL_RADIUS_M # This should only happen ONCE a correction is applied.
        
        # If no patches, or all probabilities are zero, don't update center, radius will grow.
        
        return (grid_row, grid_col)

    def _grid_to_world(self, grid_row: int, grid_col: int) -> tuple:
        """Convert grid coordinates to world coordinates (patch center)."""
        # Pixel coordinates of patch center
        x_px = (grid_col + 0.5) * self.patch_size_px
        y_px = (grid_row + 0.5) * self.patch_size_px
        
        # Convert to world coordinates relative to map center
        x_m = (x_px - self.map_w / 2.0) * self.m_per_pixel
        y_m = (y_px - self.map_h / 2.0) * self.m_per_pixel
        
        return (x_m, y_m)

    def get_state_info(self) -> dict:
        """Get current state information for visualization."""
        return {
            'center_world_m': tuple(self.center_world_m),
            'radius_m': self.radius_m,
            'num_patches': len(self.patch_probabilities),
            'max_probability': max(self.patch_probabilities.values()) if self.patch_probabilities else 0,
            'patch_probabilities': dict(self.patch_probabilities),
            'last_camera_view': self.last_camera_view,
            'last_superpoint_vis_data': self.last_superpoint_vis_data,
            'last_top5_patches': self.last_top5_patches
        }

    def update_circle_center(self, true_position_m: tuple):
        """Update circle center to ground truth position."""
        self.center_world_m = np.array(true_position_m, dtype=np.float64)
        self._update_circle_database() 