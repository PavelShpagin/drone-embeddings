import os
from dotenv import dotenv_values
import requests

# Load secrets from .env file as a dictionary
secrets = dotenv_values(".env")
API_KEY = secrets.get("GOOGLE_MAPS_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in .env file")

def get_static_map(lat, lng, zoom=20, size="2048x2048", scale=2):
    """
    Get a high-quality static map image from directly above the coordinates
    
    Args:
        lat (float): Latitude
        lng (float): Longitude
        zoom (int): Zoom level (1-22). 20 is very close to ground
        size (str): Image dimensions in pixels (widthxheight)
        scale (int): Image scale (1 or 2). 2 provides higher resolution
    """
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
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        filename = f"snapshot_{lat}_{lng}_hq.png"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Saved high-quality image as {filename}")
        return filename
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    # New York Times Square
    get_static_map(40.7580, -73.9855)
    
    # Tokyo Tower
    get_static_map(35.6586, 139.7454)

    # Closer view (zoom=21)
    get_static_map(35.6586, 139.7454, zoom=21)

    # Larger image
    get_static_map(35.6586, 139.7454, size="1024x1024") 