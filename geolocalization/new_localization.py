import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import cv2
from tqdm import tqdm
import random
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from train_encoder import SiameseNet, get_eval_transforms

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
        
        # Load trained model
        self.model = self._load_model()
        self.eval_transforms = get_eval_transforms()
        
        # Initialize confidence circle and probability map
        self.center_world_m = np.array([0.0, 0.0], dtype=np.float64)  # Will be set by drone
        self.radius_m = config.INITIAL_RADIUS_M
        self.patch_probabilities = {}  # {(grid_row, grid_col): probability}
        self.patch_embeddings = {}     # {(grid_row, grid_col): embedding}
        
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
        
        # Remove embeddings for patches no longer in circle
        current_coords = set(self.patch_probabilities.keys())
        new_coords = set(circle_patches)
        
        # Remove old patches and SET PROBABILITIES TO ZERO for patches that exit circle
        for coord in current_coords - new_coords:
            self.patch_probabilities.pop(coord, None)  # Remove probability
            self.patch_embeddings.pop(coord, None)     # Remove embedding
        
        # Add new patches and compute embeddings
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
        """Compute embeddings for a set of patch coordinates."""
        if not patch_coords:
            return
            
        patches_batch = []
        coords_batch = []
        
        for grid_row, grid_col in patch_coords:
            # Extract patch from map
            x_px = grid_col * self.patch_size_px
            y_px = grid_row * self.patch_size_px
            
            # Ensure bounds
            x_end = min(x_px + self.patch_size_px, self.map_w)
            y_end = min(y_px + self.patch_size_px, self.map_h)
            
            patch_image = self.map_image.crop((x_px, y_px, x_end, y_end))
            
            # Resize to model input size
            patch_image = patch_image.resize((self.config.CROP_SIZE_PX, self.config.CROP_SIZE_PX), 
                                           Image.Resampling.LANCZOS)
            
            patches_batch.append(self.eval_transforms(patch_image))
            coords_batch.append((grid_row, grid_col))
        
        # Compute embeddings in batch
        with torch.no_grad():
            batch_tensor = torch.stack(patches_batch).to(self.device)
            embeddings = self.model.get_embedding(batch_tensor).cpu().numpy()
            
            for i, coord in enumerate(coords_batch):
                self.patch_embeddings[coord] = embeddings[i].astype('float32')

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
                avg_neighbor_prob = np.mean(neighbor_probs)
                new_prob = avg_neighbor_prob * 0.1  # New patches get lower probability
            else:
                new_prob = 1e-6  # Very small if no neighbors
            
            self.patch_probabilities[coord] = new_prob

    def update_motion_prediction(self, vio_delta_m: np.ndarray, epsilon_m: float):
        """
        Update state using VIO measurements and motion prediction.
        Implements 1D convolutions as described in the paper.
        NOTE: Circle center is NOT updated with VIO - it should be centered on ground truth
        """
        # DON'T update circle center with VIO - circle stays centered on ground truth
        # self.center_world_m += vio_delta_m  # REMOVED - this was the bug
        
        # Increase radius by epsilon (Q3 correction)
        self.radius_m += epsilon_m
        self.radius_m = min(self.radius_m, self.config.MAX_RADIUS_M)
        
        # Update circle database
        self._update_circle_database()
        
        # Apply VIO prediction using 1D convolutions
        if self.patch_probabilities:
            self._apply_vio_prediction_convolution(vio_delta_m, epsilon_m)
        
        # Normalize probabilities
        self._normalize_probabilities()

    def _apply_vio_prediction_convolution(self, vio_delta_m: np.ndarray, epsilon_m: float):
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
        
        # Apply translation using OpenCV warpAffine
        if abs(motion_x_grid) > 0.1 or abs(motion_y_grid) > 0.1:
            shift_matrix = np.float32([[1, 0, motion_x_grid], [0, 1, motion_y_grid]])
            prob_grid = cv2.warpAffine(prob_grid, shift_matrix, (grid_w, grid_h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Apply Gaussian blur for motion uncertainty
        motion_uncertainty = (epsilon_m + self.config.VIO_X_VARIANCE) / self.patch_size_m
        if motion_uncertainty > 0.1:
            prob_grid = gaussian_filter1d(prob_grid, sigma=motion_uncertainty, axis=1)
            prob_grid = gaussian_filter1d(prob_grid, sigma=motion_uncertainty, axis=0)
        
        # Update probabilities dictionary
        self.patch_probabilities.clear()
        for row in range(grid_h):
            for col in range(grid_w):
                if prob_grid[row, col] > 1e-9:
                    coord = (row + min_row, col + min_col)
                    # Only keep if still in circle
                    if coord in self.patch_embeddings:
                        self.patch_probabilities[coord] = prob_grid[row, col]

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
        min_distance = float('inf')
        
        for coord in self.patch_probabilities:
            if coord in self.patch_embeddings:
                patch_embedding = self.patch_embeddings[coord]
                # Calculate L2 distance
                distance = np.linalg.norm(query_embedding - patch_embedding)
                distances[coord] = distance
                min_distance = min(min_distance, distance)
        
        print(f"Distance range: {min_distance:.4f} to {max(distances.values()) if distances else 0:.4f}")
        
        # Update probabilities based on similarity (lower distance = higher likelihood)
        if distances:
            # Use temperature parameter to control sharpness of probability distribution
            temperature = 0.5  # Lower = sharper distribution
            
            likelihoods = {}
            for coord in self.patch_probabilities:
                if coord in distances:
                    distance = distances[coord]
                    # Convert to likelihood using exponential with temperature
                    # Subtract min_distance for numerical stability
                    likelihood = np.exp(-(distance - min_distance) / temperature)
                    likelihoods[coord] = likelihood
                    # Update probability (multiply prior by likelihood)
                    self.patch_probabilities[coord] *= likelihood
            
            print(f"Likelihood range: {min(likelihoods.values()):.6f} to {max(likelihoods.values()):.6f}")
        
        # Normalize probabilities
        self._normalize_probabilities()
        
        # Show how probabilities changed
        final_probs = dict(self.patch_probabilities)
        prob_changes = [(coord, initial_probs[coord], final_probs[coord]) 
                       for coord in initial_probs if coord in final_probs]
        
        # Show a few examples
        if prob_changes:
            print("Probability changes (coord, before, after):")
            for i, (coord, before, after) in enumerate(prob_changes[:3]):
                print(f"  {coord}: {before:.6f} -> {after:.6f} (×{after/before:.2f})")
            print(f"  ... and {len(prob_changes)-3} more")
        
        print(f"Final prob sum: {sum(self.patch_probabilities.values()):.6f}")
        print("Measurement update completed")
        
        # --- Store top-5 retrievals for visualization ---
        # Sort by probability (descending)
        sorted_coords = sorted(self.patch_probabilities, key=lambda c: self.patch_probabilities[c], reverse=True)
        self.last_top5_patches = []
        for coord in sorted_coords[:5]:
            # Get patch image
            grid_row, grid_col = coord
            x_px = grid_col * self.patch_size_px
            y_px = grid_row * self.patch_size_px
            x_end = min(x_px + self.patch_size_px, self.map_w)
            y_end = min(y_px + self.patch_size_px, self.map_h)
            patch_image = self.map_image.crop((x_px, y_px, x_end, y_end))
            patch_image = patch_image.resize((self.config.CROP_SIZE_PX, self.config.CROP_SIZE_PX), Image.Resampling.LANCZOS)
            self.last_top5_patches.append({
                'coord': coord,
                'prob': self.patch_probabilities[coord],
                'image': patch_image
            })

    def check_correction_trigger(self) -> tuple:
        """
        Check if correction should be triggered and return correction target.
        Returns: (should_correct, target_world_pos_m)
        """
        if self.radius_m <= self.config.CORRECTION_THRESHOLD_M:
            return False, None
        
        # Find the most confident cluster
        max_prob = 0
        best_coord = None
        
        for coord, prob in self.patch_probabilities.items():
            if prob > max_prob:
                max_prob = prob
                best_coord = coord
        
        if best_coord is None:
            return False, None
        
        # Convert to world coordinates
        target_world = self._grid_to_world(best_coord[0], best_coord[1])
        
        # Check if correction distance is significant
        distance_to_center = np.linalg.norm(np.array(target_world) - self.center_world_m)
        
        if distance_to_center > 10.0:  # Minimum correction distance
            return True, target_world
        
        return False, None

    def apply_correction(self, correction_delta_m: np.ndarray):
        """Apply correction movement to the system."""
        # Move circle center
        self.center_world_m += correction_delta_m
        
        # Shrink radius
        self.radius_m *= 0.9  # Shrink by 10%
        self.radius_m = max(self.radius_m, 50.0)  # Minimum radius
        
        # Update database
        self._update_circle_database()

    def _normalize_probabilities(self):
        """Normalize all probabilities to sum to 1."""
        if not self.patch_probabilities:
            return
        
        total_prob = sum(self.patch_probabilities.values())
        if total_prob > 0:
            for coord in self.patch_probabilities:
                self.patch_probabilities[coord] /= total_prob

    def _world_to_grid(self, x_m: float, y_m: float) -> tuple:
        """Convert world coordinates to grid coordinates."""
        # Convert to pixel coordinates relative to map center
        x_px = x_m / self.m_per_pixel + self.map_w // 2
        y_px = y_m / self.m_per_pixel + self.map_h // 2
        
        # Convert to grid coordinates
        grid_col = int(x_px // self.patch_size_px)
        grid_row = int(y_px // self.patch_size_px)
        
        return (grid_row, grid_col)

    def _grid_to_world(self, grid_row: int, grid_col: int) -> tuple:
        """Convert grid coordinates to world coordinates (patch center)."""
        # Pixel coordinates of patch center
        x_px = (grid_col + 0.5) * self.patch_size_px
        y_px = (grid_row + 0.5) * self.patch_size_px
        
        # Convert to world coordinates relative to map center
        x_m = (x_px - self.map_w // 2) * self.m_per_pixel
        y_m = (y_px - self.map_h // 2) * self.m_per_pixel
        
        return (x_m, y_m)

    def get_state_info(self) -> dict:
        """Get current state information for visualization."""
        return {
            'center_world_m': tuple(self.center_world_m),
            'radius_m': self.radius_m,
            'num_patches': len(self.patch_probabilities),
            'max_probability': max(self.patch_probabilities.values()) if self.patch_probabilities else 0,
            'patch_probabilities': dict(self.patch_probabilities)
        }

    def update_circle_center(self, true_position_m: tuple):
        """Update circle center to ground truth position."""
        self.center_world_m = np.array(true_position_m, dtype=np.float64)
        self._update_circle_database() 