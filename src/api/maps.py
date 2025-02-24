import requests
from ..config import API_KEY

from PIL import Image
from io import BytesIO

def get_static_map(lat, lng, zoom=19, size="1024x1024", scale=1):
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
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
    except Exception as e:
        print(f"Error: {e}")
        return None