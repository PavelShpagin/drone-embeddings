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
from train_encoder import get_eval_transforms

class EmbeddingDatabase:
    """
    Handles the creation and querying of a database of pre-computed embeddings
    for a large satellite map.
    """
    def __init__(self, model: nn.Module, device: str, map_image: Image.Image, patch_size_px: int, m_per_pixel: float):
        self.model = model
        self.device = device
        self.map_image = map_image
        self.patch_size_px = patch_size_px
        self.m_per_pixel = m_per_pixel
        
        self.map_w, self.map_h = self.map_image.size
        self.grid_w = self.map_w // self.patch_size_px
        self.grid_h = self.map_h // self.patch_size_px
        
        self.eval_transforms = get_eval_transforms()
        self.embeddings = []
        self.grid_coordinates = []
        self.faiss_index = None

    @torch.no_grad()
    def build_database(self):
        """
        Divides the map into a coarse grid of patches and computes an embedding for each.
        Processes the map in batches to avoid storing all patches in memory at once.
        """
        process = psutil.Process(os.getpid())
        print(f"Initial memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        # Use a larger stride to create a coarser, non-exhaustive grid of embeddings
        stride = self.patch_size_px * 2 
        
        grid_positions = []
        for y in range(0, self.map_h - self.patch_size_px + 1, stride):
            for x in range(0, self.map_w - self.patch_size_px + 1, stride):
                grid_positions.append((y, x))
        
        print(f"Generating a coarse grid with {len(grid_positions)} embeddings.")

        batch_size = 512
        for i in tqdm(range(0, len(grid_positions), batch_size), desc="Building coarse embedding database"):
            positions_batch = grid_positions[i:i+batch_size]
            patches_batch = []
            
            for y, x in positions_batch:
                patch_image = self.map_image.crop((x, y, x + self.patch_size_px, y + self.patch_size_px))
                patches_batch.append(self.eval_transforms(patch_image))
                # Grid coordinates are now direct pixel coordinates of the top-left corner
                self.grid_coordinates.append((y // self.patch_size_px, x // self.patch_size_px))
            
            if patches_batch:
                batch_tensor = torch.stack(patches_batch).to(self.device)
                embedding_batch = self.model.get_embedding(batch_tensor).cpu().numpy()
                self.embeddings.extend(embedding_batch)

        print(f"Memory usage after processing all batches: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        self.embeddings = np.array(self.embeddings).astype('float32')
        self.grid_coordinates = np.array(self.grid_coordinates)
        
        # Build the FAISS index for efficient similarity search
        embedding_dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)
        self.faiss_index.add(self.embeddings)
        print(f"FAISS index built with {self.faiss_index.ntotal} embeddings.")
        print(f"Final memory usage after building DB: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Computes an embedding for a single image."""
        img_tensor = self.eval_transforms(image).unsqueeze(0).to(self.device)
        embedding = self.model.get_embedding(img_tensor).squeeze().cpu().numpy()
        return embedding.astype('float32')

    def get_closest_embeddings(self, query_embedding: np.ndarray, k: int) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Uses FAISS to find the k-nearest neighbors to the query embedding.
        Returns their distances, indices, and the embeddings themselves.
        """
        if self.faiss_index is None:
            return np.array([]), np.array([]), np.array([])
            
        # Reshape for FAISS search
        query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # The output from faiss is 2D, so we flatten it
        distances = distances.flatten()
        indices = indices.flatten()
        
        # Filter out invalid indices if any (-1)
        valid_mask = indices != -1
        distances = distances[valid_mask]
        indices = indices[valid_mask]

        return distances, indices, self.embeddings[indices]

    def get_embeddings_in_radius(self, center_m: tuple, radius_m: float) -> (list, np.ndarray):
        """
        Returns the indices and embeddings for all patches whose centers
        are within the given circle.
        """
        center_px = (center_m[0] / self.m_per_pixel, center_m[1] / self.m_per_pixel)
        radius_px = radius_m / self.m_per_pixel

        # Calculate patch center coordinates in pixels
        patch_centers_px = (self.grid_coordinates + 0.5) * self.patch_size_px
        
        # Calculate squared distance from the given center to all patch centers
        dist_sq = np.sum((patch_centers_px - np.array([center_px[1], center_px[0]]))**2, axis=1)
        
        # Find indices of patches within the radius
        indices_in_circle = np.where(dist_sq <= radius_px**2)[0]
        
        if len(indices_in_circle) == 0:
            return [], np.array([])
            
        return indices_in_circle, self.embeddings[indices_in_circle]
        
    def get_grid_coordinates(self, index: int) -> tuple:
        """Returns the (gy, gx) grid coordinates for a given embedding index."""
        return self.grid_coordinates[index]

if __name__ == '__main__':
    # Example usage:
    # This requires the data and trained model to be in the expected locations
    MAP_IMAGE_PATH = "inference/46.6234, 32.7851.jpg"
    MODEL_WEIGHTS_PATH = "training_results/efficientnet_b0/final_model.pth"
    M_PER_PIXEL = 0.487
    PATCH_SIZE_M = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(MAP_IMAGE_PATH).exists() or not Path(MODEL_WEIGHTS_PATH).exists():
        print("Error: Ensure map image and model weights are available at the specified paths.")
    else:
        db = EmbeddingDatabase(MAP_IMAGE_PATH, MODEL_WEIGHTS_PATH, M_PER_PIXEL, PATCH_SIZE_M, DEVICE)
        # You can now use db.get_embedding(x, y) or db.get_all_embeddings_in_circle(...)
        print("Example: Getting embeddings for a circle at the center of the map.")
        center_x = db.map_w // 2
        center_y = db.map_h // 2
        radius_px = 250
        circle_embs = db.get_all_embeddings_in_circle(center_x, center_y, radius_px)
        print(f"Found {len(circle_embs)} embeddings in the circle.") 