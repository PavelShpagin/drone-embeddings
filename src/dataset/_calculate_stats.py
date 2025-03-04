import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv, dotenv_values
from src.dataset.map_dataset import MapDataset, MapDataLoader

def calculate_dataset_stats(dataloader, num_images=10000):
    # Initialize arrays to store channel sums and squared sums for query and positive images
    query_channel_sum = np.zeros(3)
    query_channel_sum_squared = np.zeros(3)
    query_pixel_count = 0

    positive_channel_sum = np.zeros(3)
    positive_channel_sum_squared = np.zeros(3)
    positive_pixel_count = 0

    # Iterate through images
    for i in tqdm(range(num_images)):
        # Get the query and positive images (index 0 and 1 from the tuple)
        query_image, positive_images, _ = next(iter(dataloader))
        
        # Convert query image to numpy array and normalize to [0, 1]
        query_img_array = np.array(query_image).astype(np.float32) / 255.0
        # Convert positive image to numpy array and normalize to [0, 1]
        positive_img_array = np.array(positive_images[0]).astype(np.float32) / 255.0
        
        print(np.array(query_image).shape, np.array(positive_images[0]).shape)
        # Update sums for query image
        query_channel_sum += query_img_array.sum(axis=(0, 1))
        query_channel_sum_squared += (query_img_array ** 2).sum(axis=(0, 1))
        query_pixel_count += query_img_array.shape[0] * query_img_array.shape[1]

        # Update sums for positive image
        positive_channel_sum += positive_img_array.sum(axis=(0, 1))
        positive_channel_sum_squared += (positive_img_array ** 2).sum(axis=(0, 1))
        positive_pixel_count += positive_img_array.shape[0] * positive_img_array.shape[1]

        # Print current processed image index
        print(f"Processed image {i + 1}/{num_images}")

    # Calculate mean and std for query images
    query_mean = query_channel_sum / query_pixel_count
    query_std = np.sqrt((query_channel_sum_squared / query_pixel_count) - (query_mean ** 2))

    # Calculate mean and std for positive images
    positive_mean = positive_channel_sum / positive_pixel_count
    positive_std = np.sqrt((positive_channel_sum_squared / positive_pixel_count) - (positive_mean ** 2))

    return query_mean, query_std, positive_mean, positive_std

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    secrets = dotenv_values(".env")
    
    # Initialize dataset
    base_height = 200
    locations = [   
    (50.4162, 30.8906, base_height),  # Agricultural fields east of Kyiv
    (48.9483, 29.7241, base_height),  # Farmland in Vinnytsia Oblast
    (49.3721, 31.0945, base_height),  # Agricultural area in Cherkasy Oblast
    (48.5673, 33.4218, base_height),  # Farmland in Dnipropetrovsk Oblast
    (46.6234, 32.7851, base_height),  # Agricultural fields in Kherson Oblast
    (49.8234, 25.3612, base_height),  # Farmland west of Ternopil
    (50.7156, 29.2367, base_height),  # Agricultural area in Zhytomyr Oblast
    (51.4523, 32.8945, base_height),  # Farmland in Chernihiv Oblast
    (48.2367, 35.7823, base_height),  # Agricultural fields in Zaporizhzhia Oblast
    (47.8945, 30.2367, base_height),  # Farmland in Mykolaiv Oblast
]

    dataset = MapDataset(
        locations=locations,
        google_api_key=secrets.get("GOOGLE_MAPS_API_KEY"),
        azure_api_key=secrets.get("AZURE_MAPS_API_KEY"),
        transform=None
    )
    dataloader = MapDataLoader(dataset, pos_samples=1, neg_samples=1)

    # Calculate statistics
    mean, std = calculate_dataset_stats(dataloader, num_images=10000)
    
    print("\nDataset Statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")