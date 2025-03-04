import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests
from typing import Optional
import numpy as np
import math
from datetime import datetime

from src.config import AZURE_MAPS_API_KEY

def get_azure_maps_image(
    latitude: float,
    longitude: float,
    zoom: int = 12,
    size: int = 256,
    layer: str = "satellite",
    scale: int = 2,
    date: Optional[str] = None
) -> Optional[Image.Image]:
    """
    Retrieve a static map image from Azure Maps.
    
    Args:
        latitude (float): Center latitude
        longitude (float): Center longitude
        zoom (int): Zoom level (0-20)
        size (int): Image size in pixels (default 256)
        layer (str): Map layer type ('satellite', 'basic', etc.)
        scale (int): Resolution scale factor (1 or 2, default 2 for higher quality)
    
    Returns:
        Optional[Image.Image]: PIL Image object if successful, None otherwise
    """
    base_url = "https://atlas.microsoft.com/map/static"

    subscription_key = AZURE_MAPS_API_KEY
    
    headers = {
        'x-ms-client-id': 'd75127e2-c8d1-48b2-b401-3552da9fe791',  # Replace with your actual client ID
        'Subscription-Key': subscription_key
    }
    # Request double size and scale down for better quality
    request_size = size * scale
    
    time_stamp = ""
    if date:
        time_stamp = datetime.strptime(date, '%Y-%m-%d').strftime("%Y-%m-%dT%H:%M:%SZ")
    
    params = {
        'api-version': '2024-04-01',
        'center': f"{longitude},{latitude}",
        'zoom': zoom,
        'width': request_size,
        'height': request_size,
        'tilesetId': 'microsoft.imagery',
        'language': 'en-US',
        'timeStamp': time_stamp
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check if we received actual image data
        image = Image.open(BytesIO(response.content))
        
        # Check if the image is the default gray error image
        # Convert to grayscale and check if all pixels are similar
        gray_image = image.convert('L')
        pixels = np.array(gray_image)
        if np.std(pixels) < 8:  # If image has very little variation, it's likely an error image
            print("Received error image from Azure Maps - invalid parameters or no imagery available")
            return None
            
        if scale > 1:
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        print(f"Error fetching Azure Maps image: {e}")
        return None


def calculate_azure_zoom(altitude: float, lat: float, image_size: int = 256) -> int:
    """
    Calculate appropriate zoom level for Azure Maps based on altitude.
    
    Azure Maps uses Web Mercator projection (EPSG:3857).
    At zoom level 0, the entire world is 512x512 pixels.
    Each zoom level doubles the number of pixels.
    
    Args:
        altitude (float): Altitude in meters
        image_size (int): Size of the image in pixels
    
    Returns:
        int: Calculated zoom level (0-20)
    """
    EARTH_CIRCUMFERENCE = 40075016.686  # Earth's circumference at equator in meters
    BASE_TILE_SIZE = 512/2  # Azure Maps base tile size at zoom 0
    
    ground_resolution_0 = EARTH_CIRCUMFERENCE / BASE_TILE_SIZE
    desired_ground_resolution = (altitude * 2 * math.cos(math.radians(lat))) / image_size
    zoom = round(np.log2(ground_resolution_0 / desired_ground_resolution))
    
    return max(0, min(zoom, 20)) 