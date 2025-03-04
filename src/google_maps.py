import requests
from PIL import Image
from io import BytesIO
from .config import API_KEY
import numpy as np
import math
from datetime import datetime

def get_static_map(lat, lng, zoom=19, size="1024x1024", scale=1, date=None):
    """Get static map image from Google Maps API"""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        'center': f"{lat},{lng}",
        'zoom': zoom,
        'size': size,
        'scale': scale,
        'maptype': 'satellite',
        'format': 'png',
        'key': API_KEY
    }
    
    # Add date parameter if provided (only works with Google Maps Dynamic API)
    if date:
        params["timestamp"] = int(datetime.strptime(date, '%Y-%m-%d').timestamp())
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error fetching Google Maps image: {e}")
        return None 

def calculate_google_zoom(altitude: float, lat: float, image_size: int = 256) -> int:
    """
    Calculate appropriate zoom level for Google Maps based on altitude.
    
    Google Maps uses Web Mercator projection (EPSG:3857).
    At zoom level 0, the entire world is 256x256 pixels.
    Each zoom level doubles the number of pixels.
    
    Args:
        altitude (float): Altitude in meters
        image_size (int): Size of the image in pixels (default Google Maps size)
    
    Returns:
        int: Calculated zoom level (0-21)
    """
    EARTH_CIRCUMFERENCE = 40075016.686  # Earth's circumference at equator in meters
    BASE_TILE_SIZE = 256  # Google Maps base tile size at zoom 0
    
    ground_resolution_0 = EARTH_CIRCUMFERENCE / BASE_TILE_SIZE
    desired_ground_resolution = (altitude * 2 * math.cos(math.radians(lat))) / image_size
    zoom = round(np.log2(ground_resolution_0 / desired_ground_resolution))
    
    return max(0, min(zoom, 21))  # Google Maps supports up to zoom level 21 