import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from src.models.siamese_net import SiameseNet
from dotenv import load_dotenv, dotenv_values
import yaml
import os
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
import math
import time
from src.google_maps import calculate_google_zoom, get_static_map
from src.azure_maps import calculate_azure_zoom, get_azure_maps_image
from scipy import stats
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage

def meters_to_degrees(meters, latitude):
    """Convert meters to degrees of lat/lon"""
    # Earth's radius in meters
    earth_radius = 6371000
    
    # 1 degree of latitude is approximately 111km
    lat_deg = meters / 111111
    
    # 1 degree of longitude varies with latitude
    lon_deg = meters / (111111 * np.cos(np.radians(latitude)))
    
    return lat_deg, lon_deg

def create_grid_coordinates(center_lat, center_lon, base_height=100, grid_size=6, step_meters=50, height_step=50):
    """Create a grid of coordinates around the center point with varying heights
    
    Args:
        center_lat (float): Center latitude
        center_lon (float): Center longitude
        base_height (int): Base altitude in meters
        grid_size (int): Number of points in lat/lon directions
        step_meters (float): Distance between points in meters
        height_step (int): Step size for height in meters
    """
    # Convert step distance from meters to degrees
    step_lat, step_lon = meters_to_degrees(step_meters, center_lat)
    
    lat_range = np.linspace(center_lat - step_lat * grid_size/2, 
                           center_lat + step_lat * grid_size/2, 
                           grid_size)
    lon_range = np.linspace(center_lon - step_lon * grid_size/2, 
                           center_lon + step_lon * grid_size/2, 
                           grid_size)
    
    # Create height range in meters
    height_samples = 3  # Number of height samples
    height_values = np.array([base_height - height_step, base_height, base_height + height_step])
    
    coordinates = []
    for height in height_values:
        for lon in lon_range:
            for lat in lat_range:
                coordinates.append((lat, lon, height))
    
    return coordinates, lat_range, lon_range, height_values

def compute_similarities(model, query_image, database_images, device):
    """Compute similarities between query image and database images"""
    # Ensure inputs are tensors
    if not isinstance(query_image, torch.Tensor):
        query_image = transforms.ToTensor()(query_image)
    
    if not all(isinstance(img, torch.Tensor) for img in database_images):
        database_images = [transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img 
                         for img in database_images]
    
    model.eval()
    with torch.no_grad():
        # Get query embedding
        query = query_image.unsqueeze(0).to(device)
        query_embed = model.forward_one(query)
        
        # Get database embeddings
        database_tensor = torch.stack(database_images).to(device)
        database_embeds = model.forward_one(database_tensor)
        
        # Compute similarities (cosine similarity)
        similarities = torch.nn.functional.cosine_similarity(
            query_embed.unsqueeze(1),
            database_embeds.unsqueeze(0)
        ).squeeze().cpu().numpy()
    
    return similarities

def plot_similarity_map(similarities, lat_range, lon_range, height_values, query_coords, save_path=None):
    """Plot 3D similarity map with scatter points"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Debug the shapes
    print(f"similarities shape: {similarities.shape}")
    print(f"height_values shape: {height_values.shape}, len: {len(height_values)}")
    print(f"lon_range shape: {lon_range.shape}, len: {len(lon_range)}")
    print(f"lat_range shape: {lat_range.shape}, len: {len(lat_range)}")
    
    # Check if reshaping is possible
    expected_size = len(height_values) * len(lon_range) * len(lat_range)
    actual_size = similarities.size
    print(f"Expected size: {expected_size}, Actual size: {actual_size}")
    
    if expected_size != actual_size:
        print("WARNING: Size mismatch, adjusting for plotting...")
        # Use a subset or pad the array as needed
        if actual_size > expected_size:
            similarities = similarities[:expected_size]
        else:
            # Pad with zeros
            similarities = np.pad(similarities, (0, expected_size - actual_size))
    
    # Reshape with the correct order
    sim_3d = similarities.reshape(len(height_values), len(lon_range), len(lat_range))
    
    # Create coordinate meshgrid
    points = []
    values = []
    colors = []
    
    for h_idx, height in enumerate(height_values):
        for lon_idx, lon in enumerate(lon_range):
            for lat_idx, lat in enumerate(lat_range):
                points.append((lat, lon, height))
                sim_value = sim_3d[h_idx, lon_idx, lat_idx]
                values.append(sim_value)
                colors.append(sim_value)
    
    # Convert to numpy arrays
    points = np.array(points)
    values = np.array(values)
    colors = np.array(colors)
    
    # Calculate average similarity
    avg_similarity = np.mean(similarities)
    
    # Plot points
    scatter = ax.scatter(
        points[:, 0],  # Latitude
        points[:, 1],  # Longitude
        points[:, 2],  # Height
        c=colors,      # Color based on similarity
        cmap='hot',
        alpha=0.8,
        s=50
    )
    
    # Plot query point as a star
    ax.scatter(
        query_coords[0],  # Latitude
        query_coords[1],  # Longitude
        query_coords[2],  # Height
        color='blue',
        marker='*',
        s=300,
        label='Query Location'
    )
    
    # Customize plot
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Altitude (meters)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Similarity Score')
    
    plt.title(f'3D Similarity Map (Average: {avg_similarity:.3f})')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_top_matches(query_image, database_images, similarities, coordinates, 
                    n_top=5, save_path=None):
    """Plot query image and top N matching images with their scores"""
    top_indices = np.argsort(similarities)[-n_top:][::-1]
    
    fig, axes = plt.subplots(2, n_top, figsize=(15, 6))
    
    # Plot query image
    query_np = query_image.permute(1, 2, 0).cpu().numpy()
    # Denormalize the image for better visualization
    query_np = query_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    query_np = np.clip(query_np, 0, 1)
    
    for i in range(n_top):
        axes[0, i].imshow(query_np)
        axes[0, i].axis('off')
        if i == n_top // 2:
            axes[0, i].set_title('Query Image')
    
    # Plot top matches
    for i, idx in enumerate(top_indices):
        if idx < len(database_images):  # Ensure index is valid
            img = database_images[idx].permute(1, 2, 0).cpu().numpy()
            # Denormalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[1, i].imshow(img)
            axes[1, i].axis('off')
            if idx < len(coordinates):  # Ensure index is valid
                coords = coordinates[idx]
                score = similarities[idx]
                axes[1, i].set_title(f'Score: {score:.3f}\nLat: {coords[0]:.4f}\nLon: {coords[1]:.4f}\nAlt: {coords[2]:.0f}m')
            else:
                axes[1, i].set_title(f'Score: {similarities[idx]:.3f}')
        else:
            axes[1, i].set_title("Invalid Index")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_gaussian_fit(similarities, coordinates, query_coords, save_path=None):
    """
    Fit a Gaussian distribution to the top similarity points and visualize it
    
    Args:
        similarities (np.ndarray): Array of similarity scores
        coordinates (list): List of (lat, lon, height) tuples
        query_coords (tuple): Query coordinates (lat, lon, height)
        save_path (str): Path to save the plot
    """
    # Check array lengths and print diagnostics
    print(f"Debug - similarities shape: {similarities.shape if hasattr(similarities, 'shape') else len(similarities)}")
    print(f"Debug - coordinates length: {len(coordinates)}")
    
    # Ensure arrays have matching lengths
    if len(similarities) != len(coordinates):
        print("Warning: Similarities and coordinates arrays have different lengths.")
        print(f"Trimming to minimum length: {min(len(similarities), len(coordinates))}")
        min_len = min(len(similarities), len(coordinates))
        similarities = similarities[:min_len]
        coordinates = coordinates[:min_len]
    
    # Calculate number of top points (use at most 20, but not more than available)
    num_top = min(20, len(similarities))
    
    # Get top indices by similarity
    top_indices = np.argsort(similarities)[-num_top:]
    top_similarities = similarities[top_indices]
    
    # Extract coordinates of top points
    top_coords = np.array([coordinates[i] for i in top_indices])
    
    # Check if we have enough points for covariance calculation
    if len(top_coords) < 3:
        print(f"Error: Need at least 3 points for 3D Gaussian, but only have {len(top_coords)}.")
        # Create a simple scatter plot instead
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the available points
        scatter_all = ax.scatter(
            [c[0] for c in coordinates],
            [c[1] for c in coordinates],
            [c[2] for c in coordinates],
            c=similarities,
            cmap='viridis',
            s=30,
            alpha=0.5,
        )
        
        # Add query point
        ax.scatter(
            query_coords[0],
            query_coords[1],
            query_coords[2],
            color='red',
            marker='*',
            s=200,
            label='Query Location'
        )
        
        ax.set_title(f"Similarity Plot (not enough points for Gaussian fit)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        
        return None, None
    
    # Fit multivariate Gaussian using MLE
    mean = np.mean(top_coords, axis=0)
    cov = np.cov(top_coords, rowvar=False)
    
    # Check if covariance matrix is valid (not singular)
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix is singular or near-singular.")
        print("Adding regularization to make it invertible.")
        # Add small regularization term to diagonal
        cov = cov + np.eye(3) * 1e-6
    
    print("Gaussian Distribution Parameters:")
    print(f"Mean: {mean}")
    print(f"Covariance Matrix:\n{cov}")
    
    # Create a figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for detailed Gaussian visualization
    all_coords = np.array(coordinates)
    padding_factor = 0.2  # 20% padding around data range
    
    # Calculate bounds with padding
    min_lat = np.min(all_coords[:, 0])
    max_lat = np.max(all_coords[:, 0])
    min_lon = np.min(all_coords[:, 1])
    max_lon = np.max(all_coords[:, 1])
    min_height = np.min(all_coords[:, 2])
    max_height = np.max(all_coords[:, 2])
    
    # Add padding
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon
    height_range = max_height - min_height
    
    min_lat -= lat_range * padding_factor
    max_lat += lat_range * padding_factor
    min_lon -= lon_range * padding_factor
    max_lon += lon_range * padding_factor
    min_height -= height_range * padding_factor
    max_height += height_range * padding_factor
    
    # Create grid for volumetric rendering
    grid_size = 20  # Number of points in each dimension
    x_grid = np.linspace(min_lat, max_lat, grid_size)
    y_grid = np.linspace(min_lon, max_lon, grid_size)
    z_grid = np.linspace(min_height, max_height, grid_size)
    
    # Create 3D meshgrid
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    # Create a multivariate normal distribution
    rv = stats.multivariate_normal(mean, cov)
    
    # Stack grid points and reshape for PDF evaluation
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    density = rv.pdf(grid_points).reshape(xx.shape)
    
    # Normalize density for better visualization
    density_normalized = density / np.max(density)
    
    # Plot the Gaussian using volumetric rendering
    # First, create multiple isosurfaces at different probability levels
    levels = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(levels)))
    
    # Add isosurfaces from highest probability to lowest (for better transparency)
    for i, (level, color) in enumerate(zip(reversed(levels), reversed(colors))):
        # Find points above this density level
        mask = density_normalized > level
        
        if np.any(mask):
            # Get coordinates and corresponding densities
            points_x = xx[mask]
            points_y = yy[mask]
            points_z = zz[mask]
            points_density = density_normalized[mask]
            
            # Plot only a subset of points for efficiency (randomly sample)
            max_points = 2000  # Maximum points to plot
            if len(points_x) > max_points:
                idx = np.random.choice(len(points_x), max_points, replace=False)
                points_x = points_x[idx]
                points_y = points_y[idx]
                points_z = points_z[idx]
                points_density = points_density[idx]
            
            # Use alpha that scales with density and level
            alpha_scale = 0.1 + (level * 0.7)  # Higher levels are more opaque
            
            # Add volumetric scatter
            ax.scatter(
                points_x, points_y, points_z,
                c=points_density,
                cmap='plasma',
                alpha=alpha_scale,
                s=10 + (level * 30),  # Higher probability points are larger
                edgecolors='none'
            )
    
    # Plot the points with similarities
    scatter_all = ax.scatter(
        [c[0] for c in coordinates],
        [c[1] for c in coordinates],
        [c[2] for c in coordinates],
        c=similarities,
        cmap='viridis',
        s=30,
        alpha=0.7,
        marker='o',
        label='All Points'
    )
    
    # Plot the top points
    scatter_top = ax.scatter(
        top_coords[:, 0],
        top_coords[:, 1],
        top_coords[:, 2],
        c=top_similarities,
        cmap='hot',
        s=80,
        alpha=1.0,
        marker='D',  # Diamond marker for top points
        edgecolors='black',
        label='Top Matches'
    )
    
    # Plot query point
    ax.scatter(
        query_coords[0],
        query_coords[1],
        query_coords[2],
        color='blue',
        marker='*',
        s=200,
        label='Query Location',
        edgecolors='white'
    )
    
    # Add mean point of distribution
    ax.scatter(
        mean[0],
        mean[1],
        mean[2],
        color='lime',
        marker='x',
        s=150,
        label='Gaussian Mean',
        linewidth=3
    )
    
    # Plot eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    for i in range(3):
        # Scale eigenvector by eigenvalue
        scaled_eigenvector = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
        
        # Plot arrow from mean to mean + eigenvector
        ax.quiver(
            mean[0], mean[1], mean[2],
            scaled_eigenvector[0], scaled_eigenvector[1], scaled_eigenvector[2],
            color=['red', 'green', 'blue'][i],
            arrow_length_ratio=0.1,
            label=f'Eigenvector {i+1}',
            linewidth=2
        )
    
    # Add colorbar for similarities
    cbar = plt.colorbar(scatter_all, ax=ax, pad=0.1, shrink=0.7, aspect=20)
    cbar.set_label('Similarity Score')
    
    # Add separate colorbar for density (on the right side)
    cax2 = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(
        plt.cm.ScalarMappable(cmap='plasma'),
        cax=cax2
    )
    cbar2.set_label('Gaussian Density')
    
    # Label axes
    ax.set_xlabel('Latitude', fontsize=12)
    ax.set_ylabel('Longitude', fontsize=12)
    ax.set_zlabel('Height (meters)', fontsize=12)
    
    # Set axes limits
    ax.set_xlim(min_lat, max_lat)
    ax.set_ylim(min_lon, max_lon)
    ax.set_zlim(min_height, max_height)
    
    # Add title
    ax.set_title('Gaussian Probability Density of Top Similarity Points', fontsize=14)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Set figure background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Add statistical information as text
    info_text = (
        f"Gaussian Parameters:\n"
        f"Mean Lat: {mean[0]:.6f}\n"
        f"Mean Lon: {mean[1]:.6f}\n"
        f"Mean Height: {mean[2]:.1f}m\n"
        f"Eigenvalues: {eigenvalues[0]:.6f}, {eigenvalues[1]:.6f}, {eigenvalues[2]:.6f}\n"
        f"Det(Cov): {np.linalg.det(cov):.9f}"
    )
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, 
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Compute Mahalanobis distance from query to mean
    query_array = np.array(query_coords)
    mahalanobis_dist = np.sqrt(np.dot(np.dot((query_array - mean).T, np.linalg.inv(cov)), (query_array - mean)))
    
    # Add Mahalanobis distance info
    distance_text = (
        f"Query to Mean:\n"
        f"Euclidean: {np.linalg.norm(query_array - mean):.4f}\n"
        f"Mahalanobis: {mahalanobis_dist:.4f} σ"
    )
    plt.figtext(0.02, 0.15, distance_text, fontsize=9,
               bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean, cov

def main():
    # Load config and environment variables
    with open('config/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    load_dotenv()
    secrets = dotenv_values(".env")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and load checkpoint
    model = SiameseNet(
        backbone_name=config['model']['backbone'],
        pretrained=False,
        gem_p=config['model']['gem_p'],
        embedding_dim=config['model']['embedding_dim']
    ).to(device)
    
    # Modified checkpoint loading
    checkpoint = torch.load('models/siamese_net_best.pth', 
                          map_location=device,
                          weights_only=False)
    
    # Load only the state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Define query coordinates 
    query_coords = (37.7749, -122.4194)  # Example: San Francisco
    query_height = 100  # 100 meters above ground
    
    # Create grid with height variations - using smaller grid to reduce API calls
    grid_size = 5  # Reduced from 10 to 5
    grid_coords, lat_range, lon_range, height_values = create_grid_coordinates(
        *query_coords, 
        base_height=query_height,
        grid_size=grid_size,
        step_meters=10,  # 10 meters between grid points
        height_step=10   # 10 meters step for heights
    )
    
    # Set query coordinates with height
    query_coords_with_height = (query_coords[0], query_coords[1], query_height)
    
    # Standard image transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Fetching query image at {query_coords_with_height}...")
    # Get query image from Azure
    query_image = get_azure_maps_image(
        query_coords_with_height[0], 
        query_coords_with_height[1],
        calculate_azure_zoom(query_coords_with_height[2], query_coords_with_height[0], 256)
    )
    
    if query_image is None:
        print("Failed to fetch query image from Azure. Trying Google Maps...")
        query_image = get_static_map(
            query_coords_with_height[0],
            query_coords_with_height[1],    
            calculate_google_zoom(query_coords_with_height[2], query_coords_with_height[0], 256)
        )
        
    if query_image is None:
        print("Failed to fetch query image. Creating a random image for testing.")
        # Create a random image instead of failing
        random_image = np.random.rand(256, 256, 3)
        query_image = Image.fromarray((random_image * 255).astype('uint8'))
    
    # Apply transforms to query image
    query_image = transform(query_image.convert('RGB'))
    
    # Create a placeholder image for failed fetches
    placeholder_img = torch.zeros(3, 256, 256)
    
    # Fetch database images
    print(f"Fetching {len(grid_coords)} database images...")
    database_images = []
    sampled_coords = []

    for i, (lat, lon, height) in enumerate(grid_coords):
        # Fetch every point (or use sampling if needed)
        print(f"Fetching image {i+1}/{len(grid_coords)}...")
        
        # Alternate between Google and Azure to reduce rate limiting
        if i % 2 == 0:
            img = get_static_map(lat, lon, calculate_google_zoom(height, lat, 256))
        else:
            img = get_azure_maps_image(lat, lon, calculate_azure_zoom(height, lat, 256))
        
        if img is not None:
            database_images.append(transform(img.convert('RGB')))
            sampled_coords.append((lat, lon, height))
        else:
            # Add a placeholder
            database_images.append(placeholder_img.clone())
            sampled_coords.append((lat, lon, height))
        
        # Sleep to avoid rate limiting
        time.sleep(0.5)

    # Replace grid_coords with sampled_coords to ensure alignment
    grid_coords = sampled_coords

    # Compute similarities
    similarities = compute_similarities(model, query_image, database_images, device)
    
    # Print statistics
    print(f"Similarity Statistics:")
    print(f"Average: {np.mean(similarities):.3f}")
    print(f"Min: {np.min(similarities):.3f}")
    print(f"Max: {np.max(similarities):.3f}")
    print(f"Std: {np.std(similarities):.3f}")
    
    # Fit and plot Gaussian distribution
    gaussian_mean, gaussian_cov = plot_gaussian_fit(
        similarities, 
        grid_coords,  # These now match the indices in similarities
        query_coords_with_height,
        save_path='test_results/gaussian_fit.png'
    )
    
    # Calculate distance between query point and Gaussian mean
    query_point = np.array(query_coords_with_height)
    dist_to_mean = np.linalg.norm(query_point - gaussian_mean)
    print(f"Distance from query point to Gaussian mean: {dist_to_mean:.2f} units")
    
    # Calculate Mahalanobis distance from query point to distribution
    mahalanobis_dist = stats.mahalanobis(query_point, gaussian_mean, np.linalg.inv(gaussian_cov))
    print(f"Mahalanobis distance from query point to distribution: {mahalanobis_dist:.2f}")
    
    # Create visualizations
    os.makedirs('test_results', exist_ok=True)
    plot_similarity_map(similarities, lat_range, lon_range, height_values, 
                       query_coords_with_height, 
                       save_path='test_results/similarity_map_3d.png')
    plot_top_matches(query_image, database_images, similarities, grid_coords,
                    save_path='test_results/top_matches.png')

if __name__ == "__main__":
    main() 