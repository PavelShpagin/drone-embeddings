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

class MapDataset(Dataset):
    def __init__(self, locations, google_api_key, azure_api_key, transform=None):
        self.locations = locations
        self.google_api_key = google_api_key
        self.azure_api_key = azure_api_key
        self.transform = transform
        
        # Define seasonal date ranges (month, day)
        self.seasons = {
            'winter': [
                ('2023-12-21', '2024-02-20'),
                ('2022-12-21', '2023-02-20'),
                ('2021-12-21', '2022-02-20'),
            ],
            'spring': [
                ('2024-03-21', '2024-05-20'),
                ('2023-03-21', '2023-05-20'),
                ('2022-03-21', '2022-05-20'),
            ],
            'summer': [
                ('2023-06-21', '2023-08-20'),
                ('2022-06-21', '2022-08-20'),
                ('2021-06-21', '2021-08-20'),
            ],
            'autumn': [
                ('2023-09-21', '2023-11-20'),
                ('2022-09-21', '2022-11-20'),
                ('2021-09-21', '2021-11-20'),
            ]
        }

    def __len__(self):
        return len(self.locations)

    def _get_random_date(self):
        # Randomly select a season
        season = random.choice(list(self.seasons.keys()))
        # Randomly select a year range within the season
        date_range = random.choice(self.seasons[season])
        
        # Convert dates to datetime objects
        start_date = datetime.strptime(date_range[0], '%Y-%m-%d')
        end_date = datetime.strptime(date_range[1], '%Y-%m-%d')
        
        # Calculate random date within range
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + timedelta(days=random_number_of_days)
        
        return random_date.strftime('%Y-%m-%d')

    def __getitem__(self, idx):

        base_location = self.locations[idx % len(self.locations)]
        pseudo_random = Random(idx)
        original_height = base_location[2] + pseudo_random.uniform(-50, 300)
        
        while True:
            query_date = self._get_random_date()
            positive_date = self._get_random_date()
            negative_date = self._get_random_date()

            

            current_height = original_height
            current_base = (base_location[0], base_location[1], current_height)
            
            # Generate nearby locations with current height
            positive_location = self._get_nearby_location(current_base, 5, 20, 1, 5)
            negative_location = self._get_nearby_location(current_base, 300, 1000, 30, 100)

            # Try fetching all three images at current height with dates
            query_image = get_static_map(
                lat=current_base[0], 
                lng=current_base[1], 
                zoom=calculate_google_zoom(current_height, current_base[0]), 
                size="256x256",
                scale=1,
                date=query_date
            )
            
            positive_image = get_azure_maps_image(
                latitude=positive_location[0],
                longitude=positive_location[1],
                zoom=calculate_azure_zoom(current_height, positive_location[0]),
                size=256,
                layer='satellite',
                date=positive_date
            )
            
            negative_image = get_static_map(
                lat=negative_location[0],
                lng=negative_location[1],
                zoom=calculate_google_zoom(current_height, negative_location[0]),
                size="256x256",
                scale=1,
                date=negative_date
            )

            if all(img is not None for img in [query_image, positive_image, negative_image]):
                # Convert images to RGB mode before applying transforms
                query_image = query_image.convert('RGB')
                positive_image = positive_image.convert('RGB')
                negative_image = negative_image.convert('RGB')

                if self.transform:
                    query_image = self.transform(query_image)
                    positive_image = self.transform(positive_image)
                    negative_image = self.transform(negative_image)
                return query_image, positive_image, negative_image

    def _get_nearby_location(self, base_location, min_distance, max_distance, min_height_change, max_height_change):
        # Randomly generate a nearby location within the specified distance and height range
        lat, lon, height = base_location
        delta_lat = random.uniform(min_distance, max_distance) / 111320  # Convert meters to degrees
        delta_lon = random.uniform(min_distance, max_distance) / (111320 * abs(math.cos(math.radians(lat))))
        delta_height = random.uniform(min_height_change, max_height_change)
        return lat + delta_lat, lon + delta_lon, height + delta_height

    def _fetch_google_image(self, location, retry_count=3, height_multiplier=10):
        lat, lon, height = location
        original_height = height

        for attempt in range(retry_count):
            current_height = original_height * (height_multiplier ** attempt)
            google_zoom = calculate_google_zoom(current_height)
            image = get_static_map(
                lat=lat, 
                lng=lon, 
                zoom=google_zoom, 
                size="256x256",
                scale=1
            )
            if image is not None:
                return image
            print(f"Failed to fetch Google image at height {current_height}m, retrying at higher altitude...")

        print(f"Failed all attempts to fetch Google image for location: {lat}, {lon}")
        return Image.new('RGB', (256, 256), color='gray')

    def _fetch_azure_image(self, location, retry_count=3, height_multiplier=10):
        lat, lon, height = location
        original_height = height

        for attempt in range(retry_count):
            current_height = original_height * (height_multiplier ** attempt)
            azure_zoom = calculate_azure_zoom(current_height)
            image = get_azure_maps_image(
                latitude=lat,
                longitude=lon,
                zoom=azure_zoom,
                size=256,
                layer='satellite'
            )
            if image is not None:
                return image
            print(f"Failed to fetch Azure image at height {current_height}m, retrying at higher altitude...")

        print(f"Failed all attempts to fetch Azure image for location: {lat}, {lon}")
        return Image.new('RGB', (256, 256), color='gray')

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
    def __init__(self, dataset, pos_samples=5, neg_samples=5):
        self.dataset = dataset
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.sampler = MapBatchSampler(dataset, 1, pos_samples, neg_samples)

    def __iter__(self):
        for batch_indices in self.sampler:
            query_images = []
            positive_images = []
            negative_images = []

            base_idx = batch_indices[0]  # All indices are the same
            base_location = self.dataset.locations[base_idx]

            # Get query image
            query_img, _, _ = self.dataset[base_idx]
            query_images.append(query_img)

            # Get multiple positive samples
            for _ in range(self.pos_samples):
                _, pos_img, _ = self.dataset[base_idx]
                positive_images.append(pos_img)

            # Get multiple negative samples
            for _ in range(self.neg_samples):
                _, _, neg_img = self.dataset[base_idx]
                negative_images.append(neg_img)

            # Stack images into tensors if they're not PIL images
            if isinstance(query_images[0], torch.Tensor):
                query_images = torch.stack(query_images)
                positive_images = torch.stack(positive_images)
                negative_images = torch.stack(negative_images)

            yield query_images, positive_images, negative_images

    def __len__(self):
        return len(self.sampler)

if __name__ == "__main__":
    base_height = 200
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
    dataset = MapDataset(locations, GOOGLE_API_KEY, AZURE_API_KEY, transform=get_train_transforms())
    dataloader = MapDataLoader(dataset, pos_samples=5, neg_samples=5)

    # Example of iterating through the dataloader
    for idx, (query_batch, positive_batch, negative_batch) in enumerate(dataloader):
        if idx < 2:  # Display first 2 batches
            fig, axes = plt.subplots(3, 6, figsize=(20, 10))
            
            # Convert tensor to numpy and transpose if necessary
            def prepare_for_display(img):
                if isinstance(img, torch.Tensor):
                    return img.permute(1, 2, 0).numpy()  # Change from (C,H,W) to (H,W,C)
                return img

            # Display query image
            axes[0,0].imshow(prepare_for_display(query_batch[0]))
            axes[0,0].set_title(f'Batch {idx+1}\nQuery (Google)')
            axes[0,0].axis('off')
            for i in range(1, 6):
                axes[0,i].axis('off')
            
            # Display positive images
            for i, pos_img in enumerate(positive_batch):
                axes[1,i].imshow(prepare_for_display(pos_img))
                axes[1,i].set_title(f'Positive {i+1}')
                axes[1,i].axis('off')
            axes[1,5].axis('off')
            
            # Display negative images
            for i, neg_img in enumerate(negative_batch):
                axes[2,i].imshow(prepare_for_display(neg_img))
                axes[2,i].set_title(f'Negative {i+1}')
                axes[2,i].axis('off')
            axes[2,5].axis('off')
            
            plt.tight_layout()
            plt.show()