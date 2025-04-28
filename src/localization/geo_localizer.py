import numpy as np
import time
import yaml
from dotenv import load_dotenv, dotenv_values
import torch
from PIL import Image
from torchvision import transforms
import pyproj
from scipy.spatial import distance

# Import necessary components
from src.google_maps import get_static_map, calculate_google_zoom
from src.azure_maps import get_azure_maps_image, calculate_azure_zoom
from src.models.siamese_net import SiameseNet
from src.elevation import get_google_elevation  # Import elevation function


class GeoLocalizer:
    def __init__(self, model_path, config_path, secrets_path=".env"):
        # Load config
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            raise

        # Load environment variables
        load_dotenv(secrets_path)
        self.secrets = dotenv_values(secrets_path)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GeoLocalizer using device: {self.device}")

        # Initialize model
        model_config = self.config.get("model", {})
        backbone_name = model_config.get("backbone", "shufflenet_v2_x1_0")
        pretrained = model_config.get("pretrained", True)
        embedding_dim = model_config.get("embedding_dim", 128)

        self.model = SiameseNet(
            backbone_name=backbone_name,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
        )
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # Image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Coordinate Transformers
        self._lla_crs = pyproj.CRS("EPSG:4326")  # WGS84 geographic
        self._ecef_crs = pyproj.CRS("EPSG:4978")  # ECEF
        self._transformer_to_ecef = pyproj.Transformer.from_crs(
            self._lla_crs, self._ecef_crs, always_xy=True
        )
        self._transformer_to_lla = pyproj.Transformer.from_crs(
            self._ecef_crs, self._lla_crs, always_xy=True
        )

        # Database containers (ECEF only)
        self.db_embeddings = None
        self.db_coords_ecef = None
        self.db_images = None  # Keep for potential debugging/visualization

    def _meters_to_degrees(self, meters, latitude):
        """Approximate conversion of meters to degrees."""
        lat_deg = meters / 111111
        lng_deg = meters / (111111 * np.cos(np.radians(latitude)))
        return lat_deg, lng_deg

    def extract_embedding(self, img_tensor):
        """Extract embedding using the model."""
        with torch.no_grad():
            # Simplified: Assume model has forward_one or just call model
            if hasattr(self.model, "forward_one"):
                embedding = self.model.forward_one(img_tensor)
            else:
                embedding = self.model(img_tensor)  # Fallback
            return embedding.cpu().numpy()

    def build_database(
        self,
        lat_center,
        lng_center,
        grid_size=9,
        step_meters=50,
        height_range=30,  # Local height variation around base_height
        base_height=50,  # Local height above ground for the center altitude layer
    ):
        """Builds a database of image embeddings and ECEF coordinates."""
        print(
            f"Building database around ({lat_center:.4f}, {lng_center:.4f}), base height {base_height}m..."
        )

        lat_step_deg, lng_step_deg = self._meters_to_degrees(step_meters, lat_center)
        lat_range = np.linspace(
            lat_center - lat_step_deg * (grid_size // 2),
            lat_center + lat_step_deg * (grid_size // 2),
            grid_size,
        )
        lng_range = np.linspace(
            lng_center - lng_step_deg * (grid_size // 2),
            lng_center + lng_step_deg * (grid_size // 2),
            grid_size,
        )
        # Local heights relative to ground elevation
        height_offsets = np.array([-height_range, 0, height_range])

        target_coordinates_lla = []
        for lng in lng_range:
            for lat in lat_range:
                # Get ground elevation for this specific grid point
                ground_elevation = get_google_elevation(lat, lng)
                if ground_elevation is None:
                    print(
                        f"\nWarning: Failed to get elevation for DB point ({lat:.4f}, {lng:.4f}). Skipping."
                    )
                    continue

                for offset in height_offsets:
                    # Calculate absolute altitude (ellipsoidal)
                    altitude = ground_elevation + base_height + offset
                    target_coordinates_lla.append((lat, lng, altitude))

        if not target_coordinates_lla:
            print(
                "Error: No valid coordinates generated for the database (check elevation API?)."
            )
            return  # Cannot proceed

        embeddings_list = []
        ecef_coords_list = []
        images_list = []  # Keep for potential debugging

        print(f"Fetching {len(target_coordinates_lla)} images for database...")
        for i, (lat, lng, altitude) in enumerate(target_coordinates_lla):
            print(
                f"\rProcessing database point {i+1}/{len(target_coordinates_lla)}...",
                end="",
                flush=True,
            )

            # Use altitude directly for zoom calculation
            zoom_level = calculate_google_zoom(altitude, lat)  # Or Azure equivalent
            img = get_static_map(lat, lng, zoom_level)  # Or Azure equivalent

            if img is not None:
                img_tensor = (
                    self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
                )
                try:
                    embedding = self.extract_embedding(img_tensor)
                    if (
                        embedding is not None
                        and embedding.ndim == 2
                        and embedding.shape[0] > 0
                    ):
                        # Convert final LLA to ECEF
                        x, y, z = self._transformer_to_ecef.transform(
                            lng, lat, altitude
                        )
                        ecef_coord = np.array([x, y, z])

                        embeddings_list.append(embedding[0])
                        ecef_coords_list.append(ecef_coord)
                        images_list.append(img)  # Store image
                    else:
                        print(
                            f"\nWarning: Got invalid embedding for ({lat:.4f}, {lng:.4f}, {altitude:.1f}). Skipping."
                        )
                except Exception as e:
                    print(
                        f"\nError processing point ({lat:.4f}, {lng:.4f}, {altitude:.1f}): {e}"
                    )
            else:
                print(
                    f"\nWarning: Failed to fetch image for ({lat:.4f}, {lng:.4f}, {altitude:.1f}). Skipping."
                )

            time.sleep(0.1)  # Be nice to APIs

        print(f"\nDatabase built with {len(ecef_coords_list)} points.")

        if not ecef_coords_list:
            print("Warning: Database is empty after processing.")
            self.db_embeddings = np.empty((0, embedding_dim))  # Ensure correct shape
            self.db_coords_ecef = np.empty((0, 3))
            self.db_images = []
        else:
            self.db_embeddings = np.array(embeddings_list)
            self.db_coords_ecef = np.array(ecef_coords_list)
            self.db_images = images_list

    def find_top_k_points(self, query_img, k=5):
        """
        Finds the top K closest points in the database to the query image.

        Args:
            query_img: PIL Image.
            k: Number of top points to return.

        Returns:
            tuple: (top_k_ecef_coords, similarities, indices)
                   - top_k_ecef_coords (list): List of ECEF coordinate arrays [x, y, z].
                   - similarities (list): List of cosine similarities (1 - cosine distance).
                   - indices (list): List of integer indices of the matches in the database.
            Returns (None, None, None) if the database is empty or an error occurs.
        """
        if (
            self.db_embeddings is None
            or self.db_coords_ecef is None
            or len(self.db_embeddings) == 0
        ):
            print("Error: Database not built or is empty.")
            return None, None, None

        query_tensor = (
            self.transform(query_img.convert("RGB")).unsqueeze(0).to(self.device)
        )
        try:
            query_embedding = self.extract_embedding(query_tensor)
            if query_embedding is None or query_embedding.shape[0] == 0:
                raise ValueError("Empty query embedding")
            query_embedding = query_embedding[0]  # Assuming batch size 1
        except Exception as e:
            print(f"Error extracting query embedding: {e}")
            return None, None, None  # Indicate failure

        # Compute cosine distances (lower is better)
        distances = distance.cdist([query_embedding], self.db_embeddings, "cosine")[0]

        num_available = len(self.db_coords_ecef)
        actual_k = min(k, num_available)
        if actual_k < k:
            print(
                f"Warning: Requested k={k}, but database only has {num_available}. Returning {actual_k}."
            )

        # Get indices of the *smallest* distances
        top_indices = np.argsort(distances)[:actual_k]

        # Get corresponding ECEF coordinates and calculate similarities
        top_coords_ecef = [self.db_coords_ecef[i] for i in top_indices]
        similarities = [1 - distances[i] for i in top_indices]  # Higher is better

        # print(f"Found {len(top_indices)} matches. Top similarities: {[f'{s:.3f}' for s in similarities]}")

        return (
            top_coords_ecef,
            similarities,
            top_indices.tolist(),
        )  # Return list of indices
