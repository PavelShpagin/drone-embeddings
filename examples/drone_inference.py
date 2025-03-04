import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import distance
import time
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv, dotenv_values
import torch
from PIL import Image
from torchvision import transforms
from src.google_maps import get_static_map, calculate_google_zoom
from src.azure_maps import get_azure_maps_image, calculate_azure_zoom
from src.simulation.drone import DroneFlight
from src.models.siamese_net import SiameseNet
import pandas as pd

# =============================================================================
# Assume the following functions are defined/imported from your project:
# get_static_map, calculate_google_zoom, DroneFlight, SiameseNet
# For demonstration, we'll assume that these functions exist.
# =============================================================================

# --- Extended GeoLocalizer with top-K lookup ---
class GeoLocalizer:
    def __init__(self, model_path, config_path, secrets_path='.env'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load environment variables
        load_dotenv(secrets_path)
        self.secrets = dotenv_values(secrets_path)
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model (assumed SiameseNet)
        model_config = self.config.get('model', {})
        backbone_name = model_config.get('backbone', 'shufflenet_v2_x1_0')
        pretrained = model_config.get('pretrained', True)
        embedding_dim = model_config.get('embedding_dim', 128)
        
        print(f"Initializing model with {backbone_name}, embedding_dim: {embedding_dim}")
        self.model = SiameseNet(backbone_name=backbone_name, 
                                pretrained=pretrained,
                                embedding_dim=embedding_dim)
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading model from 'model_state_dict' key")
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading model directly from checkpoint")
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Set up image transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Database containers
        self.db_embeddings = None
        self.db_coords = None
        self.db_images = None

    def extract_embedding(self, img_tensor):
        """Extract embedding using the model's forward_one method."""
        with torch.no_grad():
            if hasattr(self.model, 'forward_one'):
                embedding = self.model.forward_one(img_tensor)
                return embedding.cpu().numpy()
            else:
                try:
                    if hasattr(self.model, 'get_embedding'):
                        return self.model.get_embedding(img_tensor).cpu().numpy()
                    elif hasattr(self.model, 'embedding'):
                        return self.model.embedding(img_tensor).cpu().numpy()
                    elif hasattr(self.model, 'encode'):
                        return self.model.encode(img_tensor).cpu().numpy()
                    else:
                        return self.model(img_tensor).cpu().numpy()
                except Exception as e:
                    print(f"Error extracting embedding: {e}")
                    return np.zeros((1, 512))
    
    def build_database(self, lat_center, lng_center, grid_size=9, 
                       step_meters=50, height_range=30, base_height=100):
        """Build a database of embeddings for a grid of coordinates."""
        print(f"Building database around ({lat_center}, {lng_center})...")
        
        lat_deg, lng_deg = self._meters_to_degrees(step_meters, lat_center)
        lat_range = np.linspace(lat_center - lat_deg * grid_size/2, 
                                lat_center + lat_deg * grid_size/2, grid_size)
        lng_range = np.linspace(lng_center - lng_deg * grid_size/2, 
                                lng_center + lng_deg * grid_size/2, grid_size)
        height_values = np.array([base_height - height_range, base_height, base_height + height_range])
        
        coordinates = []
        for height in height_values:
            for lng in lng_range:
                for lat in lat_range:
                    coordinates.append((lat, lng, height))
        
        embeddings = []
        valid_coords = []
        images = []
        
        print(f"Fetching {len(coordinates)} images for database...")
        for i, (lat, lng, height) in enumerate(coordinates):
            print(f"Fetching database image {i+1}/{len(coordinates)}...")
            # Alternate between providers if needed (here we call get_static_map)
            img = get_static_map(lat, lng, calculate_google_zoom(height, lat))
            if img is not None:
                img_tensor = self.transform(img.convert('RGB')).unsqueeze(0).to(self.device)
                try:
                    embedding = self.extract_embedding(img_tensor)
                    embeddings.append(embedding[0])
                    valid_coords.append((lat, lng, height))
                    images.append(img)
                except Exception as e:
                    print(f"Failed to get embedding for point {i}: {e}")
            time.sleep(0.5)
        
        self.db_embeddings = np.array(embeddings)
        self.db_coords = valid_coords
        self.db_images = images
        
        print(f"Database built with {len(valid_coords)} points")
        return valid_coords
    
    def find_closest_point(self, query_img):
        """Return the single best match from the database."""
        if self.db_embeddings is None:
            raise ValueError("Database not built. Call build_database first.")
        
        query_tensor = self.transform(query_img.convert('RGB')).unsqueeze(0).to(self.device)
        try:
            query_embedding = self.extract_embedding(query_tensor)[0]
        except Exception as e:
            print(f"Error extracting query embedding: {e}")
            idx = np.random.randint(0, len(self.db_coords))
            return self.db_coords[idx], 0.0, idx
        
        distances = distance.cdist([query_embedding], self.db_embeddings, 'cosine')[0]
        best_idx = np.argmin(distances)
        similarity = 1 - distances[best_idx]
        return self.db_coords[best_idx], similarity, best_idx

    def find_top_k_points(self, query_img, k=5):
        """Find the top K closest points in the database to the query image"""
        if self.db_embeddings is None:
            raise ValueError("Database not built. Call build_database first.")
        
        # Process query image
        query_tensor = self.transform(query_img.convert('RGB')).unsqueeze(0).to(self.device)
        
        # Get query embedding
        try:
            # Get embedding from model
            print(f"Query tensor shape: {query_tensor.shape}")
            query_embedding = self.extract_embedding(query_tensor)
            
            # Handle different return types
            if isinstance(query_embedding, np.ndarray):
                if query_embedding.ndim > 1:
                    query_embedding = query_embedding[0]  # Get first element if array of arrays
            elif isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy()
                if query_embedding.ndim > 1:
                    query_embedding = query_embedding[0]
                
            print(f"Query embedding shape: {query_embedding.shape}")
            
        except Exception as e:
            print(f"Error extracting query embedding: {e}")
            # Return random points if we can't get an embedding
            import random
            indices = random.sample(range(len(self.db_coords)), min(k, len(self.db_coords)))
            return [self.db_coords[i] for i in indices], [0.0] * len(indices), indices
        
        # Compute distances to all database points
        print(f"Query embedding: {query_embedding.shape}, DB embeddings: {self.db_embeddings.shape}")
        distances = distance.cdist([query_embedding], self.db_embeddings, 'cosine')[0]
        
        # Find top k points
        top_indices = np.argsort(distances)[:k]
        top_coords = [self.db_coords[i] for i in top_indices]
        similarities = [1 - distances[i] for i in top_indices]  # Convert cosine distance to similarity
        
        print(f"Found {len(top_indices)} matches with similarities: {similarities}")
        
        # Make sure we're returning all three values
        return top_coords, similarities, top_indices

    def _meters_to_degrees(self, meters, latitude):
        """Convert meters to degrees."""
        lat_deg = meters / 111111
        lng_deg = meters / (111111 * np.cos(np.radians(latitude)))
        return lat_deg, lng_deg

# =============================================================================
# Dual-Trajectory Simulation and Correction Methods
# =============================================================================

def simulate_dual_trajectories(
    start_lat, start_lng, start_altitude, velocity, 
    model_path, config_path,
    drift_std=0.1, drift_velocity=np.array([0.1, -0.05, 0.02]),
    duration=50, sample_interval=1):
    """
    Simulate both true and drifted trajectories, apply correction methods
    
    Args:
        start_lat, start_lng, start_altitude: Starting position
        velocity: Base velocity vector for both drones
        model_path, config_path: Paths for model and config
        drift_std: Standard deviation of the Gaussian noise added to drifted trajectory
        drift_velocity: Additional velocity vector added to drifted trajectory
        duration: Duration of simulation in seconds
        sample_interval: Interval between sampling points
    
    Returns:
        DataFrame with trajectory data and statistics
    """
    # Create output directories
    os.makedirs('output/drone_inference', exist_ok=True)
    
    # Initialize GeoLocalizer
    localizer = GeoLocalizer(model_path, config_path)
    
    # Build database around start point
    localizer.build_database(
        lat_center=start_lat,
        lng_center=start_lng,
        grid_size=2,  # Expanded grid
        step_meters=100,
        base_height=start_altitude
    )
    
    # Initialize true drone flight
    true_drone = DroneFlight(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=start_altitude,
        velocity=velocity,
        noise_std=0.01  # Small noise for true trajectory
    )
    
    # Initialize drifted drone flight
    # Start from same position but with drift
    drifted_drone = DroneFlight(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=start_altitude,
        velocity=velocity + drift_velocity,
        noise_std=drift_std  # Larger noise for drifted trajectory
    )
    
    # Initialize correction methods
    method1_drone_pos = [(start_lat, start_lng, start_altitude)]  # top 1 constant shift
    method2_drone_pos = [(start_lat, start_lng, start_altitude)]  # top K mean shift
    method3_drone_pos = [(start_lat, start_lng, start_altitude)]  # top K weighted by std
    method4_drone_pos = [(start_lat, start_lng, start_altitude)]  # top K gaussian product
    
    # Initialize data collectors
    true_positions = []
    true_coords = []
    drifted_positions = []
    drifted_coords = []
    
    # Top K for methods (hyperparameter)
    K = 5
    
    # Shift strength factor (hyperparameter)
    alpha = 0.3  # How strongly to shift towards the prediction (0 to 1)
    
    # Data collection for analysis
    data = []
    
    # Run simulation
    for t in range(duration):
        # Step drones forward
        true_state = true_drone.step()
        drifted_state = drifted_drone.step()
        
        # Extract individual x, y, z components to make it easier to convert to numpy array later
        true_pos = true_state.position
        true_positions.append([true_pos[0], true_pos[1], true_pos[2]])
        true_coords.append((true_state.lat, true_state.lng, true_state.altitude))
        
        drifted_pos = drifted_state.position
        drifted_positions.append([drifted_pos[0], drifted_pos[1], drifted_pos[2]])
        drifted_coords.append((drifted_state.lat, drifted_state.lng, drifted_state.altitude))
        
        # Process every sample_interval steps
        if t % sample_interval == 0:
            print(f"\nTime step: {t}/{duration}")
            
            # Get true image at current position
            if t % 2 == 0:
                true_img = get_static_map(
                    true_state.lat, true_state.lng,
                    calculate_google_zoom(true_state.altitude, true_state.lat),
                    scale=2
                )
            else:
                true_img = get_azure_maps_image(
                    true_state.lat, true_state.lng,
                    calculate_azure_zoom(true_state.altitude, true_state.lat),
                    size=256, scale=2
                )
            
            if true_img is None:
                print(f"Warning: Could not get image at {t}, skipping correction")
                continue
            
            # Find top K matches
            top_coords, similarities, indices = localizer.find_top_k_points(true_img, k=K)
            
            # Save current drone position in lat, lng, alt format
            current_drifted = np.array([drifted_state.lat, drifted_state.lng, drifted_state.altitude])
            current_true = np.array([true_state.lat, true_state.lng, true_state.altitude])
            
            # Convert top coordinates to numpy array
            top_coords_array = np.array(top_coords)
            
            # Method 1: Simple shift towards top 1 prediction (constant shift)
            top_1_coord = np.array(top_coords[0])
            method1_shift = alpha * (top_1_coord - current_drifted)
            method1_new_pos = current_drifted + method1_shift
            method1_drone_pos.append(tuple(method1_new_pos))
    
    # Convert lists to numpy arrays for proper indexing
    true_positions = np.array(true_positions)
    drifted_positions = np.array(drifted_positions)
    
    # Convert method positions to numpy arrays - skipping the initial position which was repeated
    method1_positions = lat_lng_alt_to_xyz(method1_drone_pos[1:], start_lat, start_lng)
    method2_positions = lat_lng_alt_to_xyz(method2_drone_pos[1:], start_lat, start_lng)
    method3_positions = lat_lng_alt_to_xyz(method3_drone_pos[1:], start_lat, start_lng)
    method4_positions = lat_lng_alt_to_xyz(method4_drone_pos[1:], start_lat, start_lng)
    
    # Now the arrays can be properly indexed with [:, 0], [:, 1], [:, 2]
    
    # Visualization functions would work correctly here with numpy arrays
    
    # Return the data frame for analysis
    return pd.DataFrame(data)

def lat_lng_alt_to_xyz(coords_list, ref_lat, ref_lng):
    """
    Convert list of (lat, lng, alt) tuples to numpy array of (x, y, z) relative to reference
    
    Args:
        coords_list: List of (lat, lng, alt) tuples
        ref_lat: Reference latitude in degrees
        ref_lng: Reference longitude in degrees
    
    Returns:
        numpy array of shape (n, 3) with [x, y, z] coordinates in meters
    """
    result = []
    for lat, lng, alt in coords_list:
        # Convert to meters from reference point
        # 1 degree of latitude is approximately 111,111 meters
        x = (lat - ref_lat) * 111111
        # 1 degree of longitude varies with latitude
        y = (lng - ref_lng) * 111111 * np.cos(np.radians(ref_lat))
        # Altitude is already in meters
        z = alt
        result.append([x, y, z])
    
    # Return as numpy array for easier indexing
    return np.array(result) if result else np.zeros((0, 3))

# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    # Configuration parameters (example values)
    start_lat = 46.4825  # e.g., Odesa City latitude
    start_lng = 30.7233
    start_altitude = 100  # meters
    velocity = np.array([0.5, 0.5, 0.1])  # meters per second
    
    model_path = "models/siamese_net_best.pth"  # Update to your model path
    config_path = "config/train_config.yaml"
    
    # Run dual-trajectory simulation with corrections over 50 frames
    simulation_results = simulate_dual_trajectories(
        start_lat=start_lat,
        start_lng=start_lng,
        start_altitude=start_altitude,
        velocity=velocity,
        model_path=model_path,
        config_path=config_path,
        duration=50
    )
