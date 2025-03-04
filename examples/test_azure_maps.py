from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import time
import math

def get_azure_maps_image(subscription_key, lat, lng, zoom=16, size=1024):
    """
    Get satellite imagery from Azure Maps Static Image API
    Returns PIL Image object or None
    """
    try:
        base_url = "https://atlas.microsoft.com/map/static"
        
        # Add the subscription-key in the header instead of params
        headers = {
            'x-ms-client-id': 'd75127e2-c8d1-48b2-b401-3552da9fe791',  # Your client ID
            'Subscription-Key': 'BWF9aLzBYG5kIICl9gFAFld8VctJgkmv4Vz2A8mc3YoJeG4ZMB4XJQQJ99BCACYeBjFhybFzAAAgAZMP1lhm'  # Your primary key
        }
        
        params = {
            'api-version': '2024-04-01',
            'center': f"{lng},{lat}",
            'zoom': zoom,
            'width': size,
            'height': size,
            'tilesetId': 'microsoft.imagery',
            'language': 'en-US'
        }
        
        print(f"Requesting Azure Maps image for coordinates: {lat}, {lng}")
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 200:
            print("Successfully received image")
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to get image: {response.status_code}")
            print(f"Response content: {response.content}")
            return None
            
    except Exception as e:
        print(f"Error getting Azure Maps image: {e}")
        return None

def lng_to_tile_x(lng, zoom):
    """Convert longitude to tile x coordinate"""
    n = 2.0 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    return x

def lat_to_tile_y(lat, zoom):
    """Convert latitude to tile y coordinate"""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return y

def test_azure_maps_api():
    # Load Azure Maps key
    KEY_PATH = Path("secrets/azure-maps-key.txt")
    
    if not KEY_PATH.exists():
        raise FileNotFoundError(
            f"Azure Maps key file not found at {KEY_PATH}. "
            "Please place your Azure Maps subscription key in this location."
        )
    
    subscription_key = KEY_PATH.read_text().strip()
    
    # Test locations with appropriate zoom levels
    locations = [
        {
            "name": "Odesa_Port",
            "lat": 46.4925,
            "lng": 30.7487,
            "zooms": [15, 16, 17, 18]  # Zoom levels per documentation
        },
        {
            "name": "Odesa_City",
            "lat": 46.4825,
            "lng": 30.7233,
            "zooms": [15, 16, 17, 18]
        },
        {
            "name": "Odesa_Beach",
            "lat": 46.4603,
            "lng": 30.7658,
            "zooms": [15, 16, 17, 18]
        }
    ]
    
    # Image sizes to test (within documented limits)
    sizes = [512, 1024, 1500]  # Max height/width is 1500 per docs
    
    for location in locations:
        print(f"\nTesting location: {location['name']}")
        
        for zoom in location['zooms']:
            print(f"\nTesting at zoom level: {zoom}")
            
            for size in sizes:
                print(f"Testing with image size: {size}x{size}")
                
                image = get_azure_maps_image(
                    subscription_key,
                    location['lat'],
                    location['lng'],
                    zoom=zoom,
                    size=size
                )
                
                if image:
                    # Create output directory if it doesn't exist
                    output_dir = Path("output/azure_maps")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save the image with zoom and size in filename
                    filename = output_dir / f"{location['name'].lower()}_zoom{zoom}_size{size}.png"
                    image.save(filename)
                    print(f"Image saved successfully to: {filename}")
                    
                    # Print image details
                    print(f"Image size: {image.size}")
                    print(f"Image mode: {image.mode}")
                else:
                    print("Failed to get image")
                
                # Add a small delay between requests to respect rate limits
                time.sleep(1)

if __name__ == "__main__":
    test_azure_maps_api()