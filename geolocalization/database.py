import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from tqdm import tqdm
import faiss
import psutil
import os

# Add project root to path to allow imports from other directories
sys.path.append(str(Path(__file__).parent.parent))
from train_encoder import SiameseNet, get_eval_transforms
from geolocalization import config

class EmbeddingDatabase:
    """
    Handles the creation and querying of a database of pre-computed embeddings
    for a large satellite map divided into 100m x 100m patches.
    """
    def __init__(self, device: str = None):
        self.device = device or config.DEVICE
        self.map_image_path = config.MAP_IMAGE_PATH
        self.model_weights_path = config.MODEL_WEIGHTS_PATH
        self.m_per_pixel = config.M_PER_PIXEL
        self.patch_size_m = config.GRID_PATCH_SIZE_M
        self.patch_size_px = int(self.patch_size_m / self.m_per_pixel)
        
        # Load the map image
        if not Path(self.map_image_path).exists():
            raise FileNotFoundError(f"Map image not found at {self.map_image_path}")
        self.map_image = Image.open(self.map_image_path).convert('RGB')
        self.map_w, self.map_h = self.map_image.size
        
        # Calculate grid dimensions
        self.grid_w = self.map_w // self.patch_size_px
        self.grid_h = self.map_h // self.patch_size_px
        
        print(f"Map size: {self.map_w}x{self.map_h} pixels")
        print(f"Patch size: {self.patch_size_px}x{self.patch_size_px} pixels ({self.patch_size_m}m x {self.patch_size_m}m)")
        print(f"Grid size: {self.grid_w}x{self.grid_h} patches")
        
        # Load the trained model
        self.model = self._load_model()
        self.eval_transforms = get_eval_transforms()
        
        # Database storage
        self.embeddings = []
        self.grid_coordinates = []  # (row, col) coordinates in the grid
        self.world_coordinates = []  # (x_m, y_m) coordinates in world space
        self.faiss_index = None

    def _load_model(self) -> nn.Module:
        """Load the trained SiameseNet model."""
        model = SiameseNet('efficientnet_b0').to(self.device)
        
        # Try to load the final model first, then checkpoints
        if Path(self.model_weights_path).exists():
            print(f"Loading model weights from {self.model_weights_path}")
            model.load_state_dict(torch.load(self.model_weights_path, map_location=self.device, weights_only=True))
        else:
            # Try to find the latest checkpoint
            checkpoint_dir = Path("training_results/efficientnet_b0/checkpoints")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                    print(f"Loading model weights from checkpoint: {latest_checkpoint}")
                    checkpoint = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    print("Warning: No trained model found. Using untrained model.")
            else:
                print("Warning: No trained model found. Using untrained model.")
        
        model.eval()
        return model

    @torch.no_grad()
    def build_database(self):
        """
        Divides the map into 100m x 100m patches and computes embeddings.
        Creates a comprehensive database covering the entire map.
        """
        process = psutil.Process(os.getpid())
        print(f"Initial memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        print(f"Building comprehensive embedding database...")
        
        # Create all patch positions (every 100m)
        patch_positions = []
        for row in range(self.grid_h):
            for col in range(self.grid_w):
                # Pixel coordinates of top-left corner
                x_px = col * self.patch_size_px
                y_px = row * self.patch_size_px
                
                # World coordinates of patch center (relative to map center)
                center_x_m = (x_px + self.patch_size_px // 2 - self.map_w // 2) * self.m_per_pixel
                center_y_m = (y_px + self.patch_size_px // 2 - self.map_h // 2) * self.m_per_pixel
                
                patch_positions.append((row, col, x_px, y_px, center_x_m, center_y_m))
        
        print(f"Processing {len(patch_positions)} patches total.")

        # Process in batches to manage memory
        batch_size = 256
        for i in tqdm(range(0, len(patch_positions), batch_size), desc="Building embedding database"):
            batch_positions = patch_positions[i:i+batch_size]
            patches_batch = []
            
            for row, col, x_px, y_px, center_x_m, center_y_m in batch_positions:
                # Extract patch and ensure it's the right size
                x_end = min(x_px + self.patch_size_px, self.map_w)
                y_end = min(y_px + self.patch_size_px, self.map_h)
                
                patch_image = self.map_image.crop((x_px, y_px, x_end, y_end))
                
                # Pad if necessary (for edge patches)
                if patch_image.size != (self.patch_size_px, self.patch_size_px):
                    padded_patch = Image.new('RGB', (self.patch_size_px, self.patch_size_px), color=(0, 0, 0))
                    padded_patch.paste(patch_image, (0, 0))
                    patch_image = padded_patch
                
                patches_batch.append(self.eval_transforms(patch_image))
                self.grid_coordinates.append((row, col))
                self.world_coordinates.append((center_x_m, center_y_m))
            
            if patches_batch:
                batch_tensor = torch.stack(patches_batch).to(self.device)
                embedding_batch = self.model.get_embedding(batch_tensor).cpu().numpy()
                self.embeddings.extend(embedding_batch)

        print(f"Memory usage after processing: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        self.embeddings = np.array(self.embeddings).astype('float32')
        self.grid_coordinates = np.array(self.grid_coordinates)
        self.world_coordinates = np.array(self.world_coordinates)
        
        # Build FAISS index
        embedding_dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(self.embeddings)
        
        print(f"Database built with {self.faiss_index.ntotal} embeddings.")
        print(f"Final memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Computes an embedding for a single image."""
        # Ensure image is the right size
        if image.size != (self.patch_size_px, self.patch_size_px):
            image = image.resize((self.patch_size_px, self.patch_size_px), Image.Resampling.LANCZOS)
        
        img_tensor = self.eval_transforms(image).unsqueeze(0).to(self.device)
        embedding = self.model.get_embedding(img_tensor).squeeze().cpu().numpy()
        return embedding.astype('float32')

    def get_closest_embeddings(self, query_embedding: np.ndarray, k: int) -> tuple:
        """Find the k-nearest neighbors using FAISS."""
        if self.faiss_index is None:
            return np.array([]), np.array([]), np.array([])
            
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        distances = distances.flatten()
        indices = indices.flatten()
        
        # Filter out invalid indices
        valid_mask = indices != -1
        distances = distances[valid_mask]
        indices = indices[valid_mask]

        return distances, indices, self.embeddings[indices]

    def get_embeddings_in_circle(self, center_world_m: tuple, radius_m: float) -> tuple:
        """
        Returns indices and embeddings for patches within a circular region.
        center_world_m: (x, y) coordinates in meters relative to map center
        """
        # Check if world coordinates exist
        if isinstance(self.world_coordinates, list):
            if len(self.world_coordinates) == 0:
                return [], np.array([])
            patch_centers = np.array(self.world_coordinates)
        else:
            if self.world_coordinates.size == 0:
                return [], np.array([])
            patch_centers = self.world_coordinates
            
        center_x_m, center_y_m = center_world_m
        
        # Calculate distances from center to all patch centers
        distances = np.sqrt((patch_centers[:, 0] - center_x_m)**2 + 
                          (patch_centers[:, 1] - center_y_m)**2)
        
        # Find patches within radius
        indices_in_circle = np.where(distances <= radius_m)[0]
        
        if len(indices_in_circle) == 0:
            return [], np.array([])
        
        # Handle case where embeddings aren't built yet (during testing)
        if len(self.embeddings) == 0:
            return indices_in_circle, np.array([])
            
        return indices_in_circle, self.embeddings[indices_in_circle]
        
    def pixel_to_world(self, x_px: int, y_px: int) -> tuple:
        """Convert pixel coordinates to world coordinates (meters from map center)."""
        x_m = (x_px - self.map_w // 2) * self.m_per_pixel
        y_m = (y_px - self.map_h // 2) * self.m_per_pixel
        return x_m, y_m

    def world_to_pixel(self, x_m: float, y_m: float) -> tuple:
        """Convert world coordinates to pixel coordinates."""
        x_px = int(x_m / self.m_per_pixel + self.map_w // 2)
        y_px = int(y_m / self.m_per_pixel + self.map_h // 2)
        return x_px, y_px

    def get_grid_coordinates(self, index: int) -> tuple:
        """Returns the (row, col) grid coordinates for a given embedding index."""
        return self.grid_coordinates[index]

    def get_world_coordinates(self, index: int) -> tuple:
        """Returns the (x_m, y_m) world coordinates for a given embedding index."""
        return self.world_coordinates[index]

    def get_patch_center_world(self, patch_id) -> tuple:
        """Returns the world coordinates for a patch given its ID (grid coordinates or index)."""
        if isinstance(patch_id, tuple):
            # patch_id is (row, col) grid coordinates
            row, col = patch_id
            # Convert grid coordinates to world coordinates
            center_x_px = (col + 0.5) * self.patch_size_px
            center_y_px = (row + 0.5) * self.patch_size_px
            center_x_m = (center_x_px - self.map_w // 2) * self.m_per_pixel
            center_y_m = (center_y_px - self.map_h // 2) * self.m_per_pixel
            return (center_x_m, center_y_m)
        elif isinstance(patch_id, int):
            # patch_id is an index into the database
            if 0 <= patch_id < len(self.world_coordinates):
                return self.world_coordinates[patch_id]
        
        return None

if __name__ == '__main__':
    # Test the database creation
    db = EmbeddingDatabase()
    db.build_database()
    print("Database creation test completed successfully!") 