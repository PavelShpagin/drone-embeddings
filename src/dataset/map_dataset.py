import torch
from torch.utils.data import Dataset
import requests
from io import BytesIO
from PIL import Image
import random
import math
from dotenv import load_dotenv, dotenv_values
from src.google_maps import calculate_google_zoom, get_static_map
from src.azure_maps import calculate_azure_zoom, get_azure_maps_image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from random import Random
from src.utils.transforms import get_train_transforms
import os
import uuid
import json
import time


class MapDataset(Dataset):
    def __init__(self, locations, google_api_key, azure_api_key, transform=None):
        self.locations = locations
        self.google_api_key = google_api_key
        self.azure_api_key = azure_api_key
        self.transform = transform

        if self.transform is None:
            # Ensure transform is provided if tensors are expected
            raise ValueError(
                "A transform function (e.g., ToTensor + Normalize) must be provided."
            )

        # Define seasonal date ranges (month, day)
        self.seasons = {
            "winter": [
                ("2023-12-21", "2024-02-20"),
                ("2022-12-21", "2023-02-20"),
                ("2021-12-21", "2022-02-20"),
            ],
            "spring": [
                ("2024-03-21", "2024-05-20"),
                ("2023-03-21", "2023-05-20"),
                ("2022-03-21", "2022-05-20"),
            ],
            "summer": [
                ("2023-06-21", "2023-08-20"),
                ("2022-06-21", "2022-08-20"),
                ("2021-06-21", "2021-08-20"),
            ],
            "autumn": [
                ("2023-09-21", "2023-11-20"),
                ("2022-09-21", "2022-11-20"),
                ("2021-09-21", "2021-11-20"),
            ],
        }

    def __len__(self):
        return len(self.locations)

    def _get_random_date(self):
        # Randomly select a season
        season = random.choice(list(self.seasons.keys()))
        # Randomly select a year range within the season
        date_range = random.choice(self.seasons[season])

        # Convert dates to datetime objects
        start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(date_range[1], "%Y-%m-%d")

        # Calculate random date within range
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        # Ensure days_between_dates is not negative if start/end are same
        if days_between_dates < 0:
            days_between_dates = 0
        # random.randrange(0) raises error, handle edge case
        random_number_of_days = random.randrange(days_between_dates + 1)
        random_date = start_date + timedelta(days=random_number_of_days)

        return random_date.strftime("%Y-%m-%d")

    def __getitem__(self, idx):
        """
        Fetches ONE query, ONE positive, and ONE negative sample for the given base index.
        Returns both the raw PIL images and the transformed tensors.
        Includes retry logic and error handling for image fetching.
        """
        MAX_FETCH_ATTEMPTS = 5  # Retry limit
        attempts = 0

        base_location = self.locations[idx % len(self.locations)]
        pseudo_random = Random(idx)  # Use index for deterministic randomness base
        # Ensure base height is positive
        base_height = max(1, base_location[2])  # Use 1m as minimum? Adjust if needed.
        # Add variability to height
        original_height = base_height + pseudo_random.uniform(
            -min(50, base_height - 1), 300
        )
        original_height = max(1, original_height)  # Ensure height stays positive

        while attempts < MAX_FETCH_ATTEMPTS:
            attempts += 1
            print(f"Dataset: Index {idx}, Attempt {attempts}/{MAX_FETCH_ATTEMPTS}...")

            # --- Determine locations and dates for this attempt ---
            query_date = self._get_random_date()
            positive_date = self._get_random_date()
            negative_date = self._get_random_date()

            current_height = original_height  # Use the attempt's height
            current_base = (base_location[0], base_location[1], current_height)

            # Generate nearby locations for positive/negative samples
            positive_location = self._get_nearby_location(current_base, 5, 20, 1, 5)
            negative_location = self._get_nearby_location(
                current_base, 300, 1000, 30, 100
            )

            # --- Fetch Raw PIL Images with Error Handling ---
            raw_query_image = None
            raw_positive_image = None
            raw_negative_image = None
            fetch_success = {"query": False, "positive": False, "negative": False}

            try:
                raw_query_image = get_static_map(
                    lat=current_base[0],
                    lng=current_base[1],
                    zoom=calculate_google_zoom(current_height, current_base[0]),
                    size="256x256",
                    scale=1,
                    date=query_date,
                )
                if raw_query_image is not None:
                    fetch_success["query"] = True
            except Exception as e:
                print(
                    f"Index {idx}, Attempt {attempts}: Error fetching Google query: {e}"
                )

            try:
                # Use positive_location for Azure call
                raw_positive_image = get_azure_maps_image(
                    latitude=positive_location[0],
                    longitude=positive_location[1],
                    zoom=calculate_azure_zoom(current_height, positive_location[0]),
                    size=256,
                    layer="satellite",
                    date=positive_date,
                )
                if raw_positive_image is not None:
                    fetch_success["positive"] = True
            except Exception as e:
                print(
                    f"Index {idx}, Attempt {attempts}: Error fetching Azure positive: {e}"
                )

            try:
                # Use negative_location for Google call
                raw_negative_image = get_static_map(
                    lat=negative_location[0],
                    lng=negative_location[1],
                    zoom=calculate_google_zoom(current_height, negative_location[0]),
                    size="256x256",
                    scale=1,
                    date=negative_date,
                )
                if raw_negative_image is not None:
                    fetch_success["negative"] = True
            except Exception as e:
                print(
                    f"Index {idx}, Attempt {attempts}: Error fetching Google negative: {e}"
                )

            # Log fetch status
            print(
                f"Index {idx}, Attempt {attempts}: Fetch Status Q:{fetch_success['query']}, P:{fetch_success['positive']}, N:{fetch_success['negative']}"
            )
            all_fetched = all(fetch_success.values())

            if all_fetched:
                # --- Process Successful Fetch ---
                try:
                    # Convert to RGB *before* transform
                    raw_query_image = raw_query_image.convert("RGB")
                    raw_positive_image = raw_positive_image.convert("RGB")
                    raw_negative_image = raw_negative_image.convert("RGB")

                    # Apply transformations to copies
                    query_t = self.transform(raw_query_image.copy())
                    positive_t = self.transform(raw_positive_image.copy())
                    negative_t = self.transform(raw_negative_image.copy())

                    # --- Return the 6 items ---
                    print(
                        f"Index {idx}, Attempt {attempts}: Success. Returning 6 items."
                    )
                    return (
                        raw_query_image,
                        raw_positive_image,
                        raw_negative_image,
                        query_t,
                        positive_t,
                        negative_t,
                    )
                except Exception as e:
                    # Catch errors during conversion or transform
                    print(
                        f"Index {idx}, Attempt {attempts}: Error processing/transforming images: {e}"
                    )
                    # Fall through to retry (loop continues)

            # --- Handle Fetch Failure ---
            print(f"Index {idx}, Attempt {attempts}: Incomplete image set, retrying...")
            # Optional: Increase height slightly for next attempt? Could help if zoom is issue.
            original_height *= 1.05
            original_height = max(1, original_height)  # Keep height positive
            # Optional: Add delay
            time.sleep(0.5)

        # --- Exhausted Retries ---
        print(
            f"FATAL: Failed to fetch complete image set for index {idx} after {MAX_FETCH_ATTEMPTS} attempts."
        )
        # Raise specific error for dataloader to catch
        raise RuntimeError(f"Failed fetching images for index {idx}")

    def _get_nearby_location(
        self,
        base_location,
        min_distance,
        max_distance,
        min_height_change,
        max_height_change,
    ):
        # Randomly generate a nearby location within the specified distance and height range
        lat, lon, height = base_location
        # Ensure distances are positive
        min_distance = max(0, min_distance)
        max_distance = max(min_distance, max_distance)

        distance = random.uniform(min_distance, max_distance)
        angle = random.uniform(0, 2 * math.pi)  # Random angle

        # Approximate conversion: 1 degree latitude ~= 111 km
        delta_lat = (distance * math.cos(angle)) / 111111
        # Approximate conversion: 1 degree longitude ~= 111 km * cos(latitude)
        delta_lon = (distance * math.sin(angle)) / (
            111111 * abs(math.cos(math.radians(lat)))
        )

        delta_height = random.uniform(min_height_change, max_height_change)
        return lat + delta_lat, lon + delta_lon, height + delta_height

    def _fetch_google_image(self, location, retry_count=3, height_multiplier=10):
        lat, lon, height = location
        original_height = height

        for attempt in range(retry_count):
            current_height = original_height * (height_multiplier**attempt)
            google_zoom = calculate_google_zoom(current_height)
            image = get_static_map(
                lat=lat, lng=lon, zoom=google_zoom, size="256x256", scale=1
            )
            if image is not None:
                return image
            print(
                f"Failed to fetch Google image at height {current_height}m, retrying at higher altitude..."
            )

        print(f"Failed all attempts to fetch Google image for location: {lat}, {lon}")
        return Image.new("RGB", (256, 256), color="gray")

    def _fetch_azure_image(self, location, retry_count=3, height_multiplier=10):
        lat, lon, height = location
        original_height = height

        for attempt in range(retry_count):
            current_height = original_height * (height_multiplier**attempt)
            azure_zoom = calculate_azure_zoom(current_height)
            image = get_azure_maps_image(
                latitude=lat,
                longitude=lon,
                zoom=azure_zoom,
                size=256,
                layer="satellite",
            )
            if image is not None:
                return image
            print(
                f"Failed to fetch Azure image at height {current_height}m, retrying at higher altitude..."
            )

        print(f"Failed all attempts to fetch Azure image for location: {lat}, {lon}")
        return Image.new("RGB", (256, 256), color="gray")


# --- Helper function for saving (can be outside or inside the class) ---
def _save_raw_image_helper(pil_image, output_dir, project_root):
    """Saves a PIL image with a UUID name and returns the relative path."""
    if pil_image is None:
        # This check might be redundant if None images are filtered before calling
        return None
    try:
        img_uuid = uuid.uuid4()
        img_format = "png"  # Assuming PNG
        img_filename = f"{img_uuid}.{img_format}"
        img_path_absolute = os.path.join(output_dir, img_filename)

        # Ensure directory exists (might be called many times, but exist_ok handles it)
        os.makedirs(output_dir, exist_ok=True)
        pil_image.save(img_path_absolute, format="PNG")

        relative_path = os.path.relpath(img_path_absolute, project_root).replace(
            "\\", "/"
        )
        return relative_path
    except Exception as e:
        print(f"Error saving raw image: {e}")
        return None


class MapBatchSampler:
    def __init__(self, dataset, batch_size=1, pos_samples=5, neg_samples=5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.num_batches = len(dataset)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for idx in indices:
            yield [idx] * (1 + self.pos_samples + self.neg_samples)

    def __len__(self):
        return self.num_batches


class MapDataLoader:
    def __init__(
        self,
        dataset,
        pos_samples=5,
        neg_samples=5,
        save=False,
        imagery_output_dir=None,
        jsonl_output_file=None,
        project_root=None,
    ):
        self.dataset = dataset
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.sampler = MapBatchSampler(
            dataset, 1, pos_samples, neg_samples
        )  # Assumes batch size 1 implicitly

        self.save = save
        self.imagery_output_dir = imagery_output_dir
        self.jsonl_output_file = jsonl_output_file
        self.project_root = project_root

        if self.save:
            if not all(
                [self.imagery_output_dir, self.jsonl_output_file, self.project_root]
            ):
                raise ValueError(
                    "If save=True, imagery_output_dir, jsonl_output_file, and project_root must be provided."
                )
            # Ensure output directory for JSONL exists upfront
            os.makedirs(os.path.dirname(self.jsonl_output_file), exist_ok=True)
            # Ensure imagery directory exists upfront
            os.makedirs(self.imagery_output_dir, exist_ok=True)

    def __iter__(self):
        for batch_indices in self.sampler:
            # --- Data Collection ---
            # Accumulators for the batch (assuming batch size is handled by sampler logic, likely 1)
            raw_queries_pil = []
            raw_positives_pil = []
            raw_negatives_pil = []
            query_tensors = []
            positive_tensors = []
            negative_tensors = []

            # Assuming sampler yields list for one logical sample (query + its pos/neg)
            base_idx = batch_indices[0]  # The core index for this iteration

            # Fetch query data (raw + tensor)
            # Need to handle potential fetch errors within __getitem__ now
            try:
                q_pil, _, _, q_t, _, _ = self.dataset[
                    base_idx
                ]  # Get query raw PIL and tensor
                raw_queries_pil.append(q_pil)
                query_tensors.append(q_t)
            except Exception as e:
                print(
                    f"Error fetching query data for index {base_idx}: {e}. Skipping sample."
                )
                continue  # Skip this iteration if query fails

            # Fetch positive samples (raw + tensor)
            current_pos_pil = []
            current_pos_t = []
            for _ in range(self.pos_samples):
                try:
                    # __getitem__ needs to reliably generate a distinct positive sample each time it's called for the same index
                    _, p_pil, _, _, p_t, _ = self.dataset[
                        base_idx
                    ]  # Get positive raw PIL and tensor
                    current_pos_pil.append(p_pil)
                    current_pos_t.append(p_t)
                except Exception as e:
                    print(
                        f"Error fetching positive data for index {base_idx}: {e}. Skipping positive sample."
                    )
            raw_positives_pil.append(
                current_pos_pil
            )  # List of lists (or adjust if batch size > 1)
            positive_tensors.extend(current_pos_t)  # Flatten list for stacking

            # Fetch negative samples (raw + tensor)
            current_neg_pil = []
            current_neg_t = []
            for _ in range(self.neg_samples):
                try:
                    # __getitem__ needs to reliably generate a distinct negative sample each time it's called for the same index
                    _, _, n_pil, _, _, n_t = self.dataset[
                        base_idx
                    ]  # Get negative raw PIL and tensor
                    current_neg_pil.append(n_pil)
                    current_neg_t.append(n_t)
                except Exception as e:
                    print(
                        f"Error fetching negative data for index {base_idx}: {e}. Skipping negative sample."
                    )
            raw_negatives_pil.append(current_neg_pil)  # List of lists
            negative_tensors.extend(current_neg_t)  # Flatten list for stacking

            # --- Saving Logic (if enabled) ---
            if self.save:
                # Assuming batch size of 1 from the sampler for simplicity here
                q_path = _save_raw_image_helper(
                    raw_queries_pil[0], self.imagery_output_dir, self.project_root
                )

                # Check if query image saved successfully
                if q_path:
                    p_paths = [
                        _save_raw_image_helper(
                            p_img, self.imagery_output_dir, self.project_root
                        )
                        for p_img in raw_positives_pil[0]
                    ]
                    n_paths = [
                        _save_raw_image_helper(
                            n_img, self.imagery_output_dir, self.project_root
                        )
                        for n_img in raw_negatives_pil[0]
                    ]

                    # Filter out None paths (failed saves)
                    p_paths = [p for p in p_paths if p is not None]
                    n_paths = [n for n in n_paths if n is not None]

                    record = {
                        "query_path": q_path,
                        "positive_paths": p_paths,
                        "negative_paths": n_paths,
                    }

                    try:
                        # Open/append/close for each record (safer, less efficient)
                        with open(self.jsonl_output_file, "a") as f_jsonl:
                            f_jsonl.write(json.dumps(record) + "\n")
                    except IOError as e:
                        print(f"Error writing to {self.jsonl_output_file}: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred during JSONL writing: {e}")
                else:
                    print(
                        f"Skipping record for index {base_idx} because query image failed to save."
                    )

            # --- Batching Tensors ---
            # Check if we collected enough tensors (especially if fetching failed)
            if (
                not query_tensors
                or len(positive_tensors) != self.pos_samples
                or len(negative_tensors) != self.neg_samples
            ):
                print(
                    f"Warning: Insufficient tensors collected for index {base_idx} (Q:{len(query_tensors)}, P:{len(positive_tensors)}, N:{len(negative_tensors)}). Skipping batch."
                )
                continue

            # Stack tensors to create the batch
            query_batch = torch.stack(query_tensors)
            positive_batch = torch.stack(positive_tensors)
            negative_batch = torch.stack(negative_tensors)

            # --- Yield Tensors Only ---
            yield query_batch, positive_batch, negative_batch

    def __len__(self):
        return len(self.sampler)


if __name__ == "__main__":
    base_height = 100
    # Example usage
    locations = [
        (50.4162, 30.8906, base_height),  # Agricultural fields east of Kyiv
        (48.9483, 29.7241, base_height),  # Farmland in Vinnytsia Oblast
        (49.3721, 31.0945, base_height),  # Agricultural area in Cherkasy Oblast
        (48.5673, 33.4218, base_height),  # Farmland in Dnipropetrovsk Oblast
        (46.6234, 32.7851, base_height),  # Agricultural fields in Kherson Oblast
        (50.7156, 29.2367, base_height),  # Agricultural area in Zhytomyr Oblast
        (51.4523, 32.8945, base_height),  # Farmland in Chernihiv Oblast
        (48.2367, 35.7823, base_height),  # Agricultural fields in Zaporizhzhia Oblast
        (47.8945, 30.2367, base_height),  # Farmland in Mykolaiv Oblast
    ]

    load_dotenv()

    # Load secrets from .env file as a dictionary
    secrets = dotenv_values(".env")
    GOOGLE_API_KEY = secrets.get("GOOGLE_MAPS_API_KEY")
    AZURE_API_KEY = secrets.get("AZURE_MAPS_API_KEY")
    dataset = MapDataset(
        locations, GOOGLE_API_KEY, AZURE_API_KEY, transform=get_train_transforms()
    )
    dataloader = MapDataLoader(dataset, pos_samples=5, neg_samples=5)

    # Example of iterating through the dataloader
    for idx, (query_batch, positive_batch, negative_batch) in enumerate(dataloader):
        if idx < 2:  # Display first 2 batches
            fig, axes = plt.subplots(3, 6, figsize=(20, 10))

            # Convert tensor to numpy and transpose if necessary
            def prepare_for_display(img):
                if isinstance(img, torch.Tensor):
                    return img.permute(
                        1, 2, 0
                    ).numpy()  # Change from (C,H,W) to (H,W,C)
                return img

            # Display query image
            axes[0, 0].imshow(prepare_for_display(query_batch[0]))
            axes[0, 0].set_title(f"Batch {idx+1}\nQuery (Google)")
            axes[0, 0].axis("off")
            for i in range(1, 6):
                axes[0, i].axis("off")

            # Display positive images
            for i, pos_img in enumerate(positive_batch):
                axes[1, i].imshow(prepare_for_display(pos_img))
                axes[1, i].set_title(f"Positive {i+1}")
                axes[1, i].axis("off")
            axes[1, 5].axis("off")

            # Display negative images
            for i, neg_img in enumerate(negative_batch):
                axes[2, i].imshow(prepare_for_display(neg_img))
                axes[2, i].set_title(f"Negative {i+1}")
                axes[2, i].axis("off")
            axes[2, 5].axis("off")

            plt.tight_layout()
            plt.show()
