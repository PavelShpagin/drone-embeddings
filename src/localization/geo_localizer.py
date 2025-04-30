import numpy as np
import time
import yaml
from dotenv import load_dotenv, dotenv_values
import torch
from PIL import Image
from torchvision import transforms
import pyproj
from scipy.spatial import distance
import concurrent.futures

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
        """Extract embedding using the model for a batch of images."""
        with torch.no_grad():
            # Ensure tensor is on the correct device
            if img_tensor.device != self.device:
                img_tensor = img_tensor.to(self.device)

            # Simplified: Assume model has forward_one or just call model
            if hasattr(self.model, "forward_one"):
                # If forward_one exists but doesn't handle batches, this needs adjustment.
                # Assuming direct model call handles batches correctly.
                # Check SiameseNet implementation if forward_one is intended for single images.
                try:
                    embedding = self.model(img_tensor)
                except TypeError:
                    print(
                        "Warning: Model direct call failed for batch, attempting loop with forward_one (may be slow)."
                    )
                    # Fallback: Process batch items individually (less efficient)
                    embeddings_list = [
                        self.model.forward_one(item.unsqueeze(0)) for item in img_tensor
                    ]
                    embedding = torch.cat(embeddings_list, dim=0)
            else:
                embedding = self.model(
                    img_tensor
                )  # Standard model call assumes batch handling
            return embedding.cpu().numpy()

    def _fetch_and_process_point(
        self, lat, lng, altitude, max_retries=3, initial_delay=0.2
    ):
        """Fetches image, transforms it to tensor, and converts coords for a single point."""
        current_delay = initial_delay
        for attempt in range(max_retries):
            try:
                # --- API Call with Rate Limit ---
                zoom_level = calculate_google_zoom(altitude, lat)  # Or Azure equivalent
                img = get_static_map(lat, lng, zoom_level)  # Or Azure equivalent
                time.sleep(0.1)

                if img is not None:
                    # --- Process Image (Transform only) ---
                    img_tensor = self.transform(img.convert("RGB")).unsqueeze(0)
                    # --- Convert Coords ---
                    x, y, z = self._transformer_to_ecef.transform(lng, lat, altitude)
                    ecef_coord = np.array([x, y, z])
                    # Return successful result (tensor, coord, original_img)
                    # Remove the extra batch dim from tensor before returning
                    return img_tensor.squeeze(0), ecef_coord, img
                else:
                    # Image fetch failed
                    print(
                        f"\nWarning: Failed fetch attempt {attempt+1}/{max_retries} for ({lat:.4f}, {lng:.4f}, {altitude:.1f}).",
                        flush=True,
                    )

            except Exception as e:
                print(
                    f"\nError processing point ({lat:.4f}, {lng:.4f}, {altitude:.1f}) on attempt {attempt+1}: {e}",
                    flush=True,
                )

            # --- Retry Logic ---
            if attempt < max_retries - 1:
                print(f"Retrying in {current_delay:.1f}s...", flush=True)
                time.sleep(current_delay)
                current_delay *= 2
            else:
                print(
                    f"Max retries reached for ({lat:.4f}, {lng:.4f}, {altitude:.1f}). Skipping.",
                    flush=True,
                )
                return None, None, None

        return None, None, None

    def build_database(
        self,
        lat_center,
        lng_center,
        grid_size=9,
        step_meters=50,
        height_range=30,
        base_height=50,
        max_workers=8,  # Number of parallel threads for fetching
        batch_size=32,  # Batch size for model inference
    ):
        """Builds a database of image embeddings and ECEF coordinates in parallel, using batch inference."""
        print(
            f"Building database around ({lat_center:.4f}, {lng_center:.4f}), base height {base_height}m..."
        )
        start_time = time.time()

        # --- 1. Generate Target Coordinates (Elevation still sequential) ---
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
        height_offsets = np.array([-height_range, 0, height_range])

        target_coordinates_lla = []
        print("Calculating grid coordinates and fetching ground elevations...")
        coord_candidates = []
        for lng in lng_range:
            for lat in lat_range:
                coord_candidates.append({"lat": lat, "lng": lng})

        for candidate in coord_candidates:
            lat, lng = candidate["lat"], candidate["lng"]
            ground_elevation = get_google_elevation(lat, lng)
            if ground_elevation is None:
                print(
                    f"\nWarning: Failed to get elevation for DB point ({lat:.4f}, {lng:.4f}). Skipping heights for this point.",
                    flush=True,
                )
                continue
            for offset in height_offsets:
                altitude = ground_elevation + base_height + offset
                target_coordinates_lla.append((lat, lng, altitude))
            time.sleep(0.05)

        if not target_coordinates_lla:
            print("Error: No valid coordinates generated for the database.")
            # Assign empty arrays with correct dimensions
            embedding_dim = (
                self.model.embedding_dim
                if hasattr(self.model, "embedding_dim")
                else self.config.get("model", {}).get("embedding_dim", 128)
            )
            self.db_embeddings = np.empty((0, embedding_dim))
            self.db_coords_ecef = np.empty((0, 3))
            self.db_images = []
            return

        # --- 2. Fetch and Transform Images in Parallel ---
        print(
            f"Fetching and transforming {len(target_coordinates_lla)} images using up to {max_workers} workers..."
        )
        fetch_results = []  # Store tuples of (tensor, ecef_coord, img)
        processed_count = 0
        total_points = len(target_coordinates_lla)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_coord = {
                executor.submit(self._fetch_and_process_point, lat, lng, alt): (
                    lat,
                    lng,
                    alt,
                )
                for lat, lng, alt in target_coordinates_lla
            }

            for future in concurrent.futures.as_completed(future_to_coord):
                coord = future_to_coord[future]
                try:
                    img_tensor, ecef_coord, img = future.result()
                    if (
                        img_tensor is not None
                    ):  # Check if fetch/transform was successful
                        fetch_results.append((img_tensor, ecef_coord, img))
                except Exception as exc:
                    lat, lng, alt = coord
                    print(
                        f"\nError processing ({lat:.4f}, {lng:.4f}, {alt:.1f}) during fetch/transform: {exc}",
                        flush=True,
                    )
                processed_count += 1
                print(
                    f"\rFetched/Transformed {processed_count}/{total_points} images...",
                    end="",
                    flush=True,
                )

        print(f"\nFinished fetching. Got {len(fetch_results)} successful results.")

        if not fetch_results:
            print(
                "Warning: No images were successfully fetched and transformed. Database will be empty."
            )
            embedding_dim = (
                self.model.embedding_dim
                if hasattr(self.model, "embedding_dim")
                else self.config.get("model", {}).get("embedding_dim", 128)
            )
            self.db_embeddings = np.empty((0, embedding_dim))
            self.db_coords_ecef = np.empty((0, 3))
            self.db_images = []
            return

        # --- 3. Batch Inference ---
        print(f"Running batch inference (batch size: {batch_size})...")
        inference_start_time = time.time()
        all_embeddings = []
        all_coords = []
        all_images = []  # Keep corresponding images if needed

        # Sort results by coordinate perhaps? Or maintain order? Let's maintain order for now.
        # Unzip the fetched results
        img_tensors, ecef_coords, imgs = zip(*fetch_results)

        num_batches = (len(img_tensors) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_start_idx = i * batch_size
            batch_end_idx = min((i + 1) * batch_size, len(img_tensors))
            tensor_batch_list = img_tensors[batch_start_idx:batch_end_idx]

            if not tensor_batch_list:
                continue

            # Stack tensors into a batch
            tensor_batch = torch.stack(tensor_batch_list)

            # Run inference
            embedding_batch = self.extract_embedding(
                tensor_batch
            )  # Expects batch, returns numpy array

            # Append results
            all_embeddings.extend(embedding_batch)
            all_coords.extend(ecef_coords[batch_start_idx:batch_end_idx])
            all_images.extend(imgs[batch_start_idx:batch_end_idx])

            print(f"\rProcessed batch {i+1}/{num_batches}...", end="", flush=True)

        inference_end_time = time.time()
        print(
            f"\nInference finished in {inference_end_time - inference_start_time:.2f} seconds."
        )

        # --- 4. Assemble Database ---
        print(f"Assembling final database...")
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds.")

        if not all_coords:
            print("Warning: Database is empty after processing.")
            embedding_dim = (
                self.model.embedding_dim
                if hasattr(self.model, "embedding_dim")
                else self.config.get("model", {}).get("embedding_dim", 128)
            )
            self.db_embeddings = np.empty((0, embedding_dim))
            self.db_coords_ecef = np.empty((0, 3))
            self.db_images = []
        else:
            self.db_embeddings = np.array(all_embeddings)
            self.db_coords_ecef = np.array(all_coords)
            self.db_images = all_images  # Store the collected images
            print(f"Final database size: {len(self.db_coords_ecef)} points.")
            print(f"Embeddings shape: {self.db_embeddings.shape}")
            print(f"Coordinates shape: {self.db_coords_ecef.shape}")

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
