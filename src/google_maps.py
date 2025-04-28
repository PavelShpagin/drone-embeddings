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
        'zoom': int(zoom),
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
        # Check if we received actual image data
        if response.content:
            # Get the image
            image = Image.open(BytesIO(response.content))
            
            # Calculate the scale factor based on the difference between actual zoom and rounded zoom
            # This handles fractional zoom levels by scaling the image appropriately
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
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error fetching Google Maps image: {e}")
        return None 

def calculate_google_zoom(altitude: float, lat: float, image_size: int = 256) -> int:
    """
    Calculate appropriate zoom level for Google Maps based on altitude 
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
    GOOGLE_BASE_MAP_WIDTH_PX = 512  # Google Maps uses 256px map width at zoom 0

    # Target ground resolution (meters per pixel)
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
        zoom_float = np.log2( (EARTH_CIRCUMFERENCE * cos_lat) / (GOOGLE_BASE_MAP_WIDTH_PX * target_resolution) )
    except ValueError:
        # Handle potential math domain errors if the argument to log2 is non-positive
        zoom_float = 20 # Default to max zoom in case of calculation error

    # Clamp zoom level to Google's typical satellite imagery range
    return max(0, min(zoom_float, 20)) 