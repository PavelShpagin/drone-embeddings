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
        'x-ms-client-id': '7a43d81a-e128-4cc0-9769-433ec717aa42',  # Replace with your actual client ID
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
        'zoom': int(zoom),
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

        zoom_diff = zoom - int(zoom)
        scale_factor = 2 ** zoom_diff

        if scale_factor != 1.0:
            # Get current dimensions
            width, height = image.size
            
            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize the image
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Calculate crop box to get center portion
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            right = left + width
            bottom = top + height
            
            # Crop to original size
            image = image.crop((left, top, right, bottom))
            
        return image

    except Exception as e:
        print(f"Error fetching Azure Maps image: {e}")
        return None


def calculate_azure_zoom(altitude: float, lat: float, image_size: int = 256) -> int:
    """
    Calculate appropriate zoom level for Azure Maps based on altitude 
    to achieve a consistent ground resolution across different map providers.

    Args:
        altitude (float): Altitude above ground in meters.
        lat (float): Latitude in degrees.
        image_size (int): The desired edge size of the map image in pixels (default 256).

    Returns:
        int: Calculated zoom level (0-20).
    """
    if altitude <= 0:
        return 20 # Max zoom if altitude is zero or negative

    EARTH_CIRCUMFERENCE = 40075016.686  # meters
    AZURE_BASE_MAP_WIDTH_PX = 512  # Azure Maps uses 512px map width at zoom 0

    # Target ground resolution (meters per pixel) - should be same as calculated for Google
    # Simplified relation: scale proportional to altitude. Factor 2 kept from original.
    target_resolution = (altitude * 2) / image_size

    # Calculate required zoom level using the Mercator projection resolution formula:
    # Resolution = (Circumference * cos(lat)) / (BaseMapWidth * 2^zoom)
    # Solving for zoom: zoom = log2( (Circumference * cos(lat)) / (BaseMapWidth * TargetResolution) )

    # Prevent division by zero or log of non-positive number if target_resolution is invalid
    if target_resolution <= 0:
        return 20

    cos_lat = math.cos(math.radians(lat))
    if cos_lat <= 0: # Avoid issues near the poles
         return 0 # Min zoom if at pole
         
    try:
        zoom_float = np.log2( (EARTH_CIRCUMFERENCE * cos_lat) / (AZURE_BASE_MAP_WIDTH_PX * target_resolution) )
    except ValueError:
        # Handle potential math domain errors if the argument to log2 is non-positive
        zoom_float = 20 # Default to max zoom in case of calculation error

    # Clamp zoom level to Azure's typical satellite imagery range
    return max(0, min(zoom_float, 20)) 