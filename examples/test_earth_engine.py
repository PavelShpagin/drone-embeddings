from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
import urllib

def calculate_zoom(altitude, lat):
    """
    Calculate appropriate zoom level based on altitude using Google Earth's formula
    altitude: in meters
    lat: latitude in degrees
    """
    return -np.log2(altitude * 2 / (1024 * 156543.03392 * np.cos(lat * np.pi / 180)))

def get_earth_engine_image(lat, lng, altitude, session, dataset_info):
    """
    Get Earth Engine satellite image for given coordinates and altitude
    Returns PIL Image object or None
    """
    project = 'projects/earthengine-public'
    asset_id = dataset_info['asset_id']
    name = f'{project}/assets/{asset_id}'
    
    print(f"\nTrying dataset: {dataset_info['name']}")
    url = f'https://earthengine.googleapis.com/v1alpha/{name}:listImages'
    
    params = {
        'startTime': '2023-01-01T00:00:00.000Z',
        'endTime': '2023-12-31T00:00:00.000Z',
        'region': '{"type":"Point", "coordinates":' + str([lng, lat]) + '}',
        'filter': dataset_info.get('filter', 'CLOUD_COVER < 20')
    }
    
    try:
        response = session.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed to list images: {response.status_code}")
            print(f"Response content: {response.content}")
            return None
            
        content = json.loads(response.content)
        images = content.get('images', [])
        
        if not images:
            print("No suitable images found")
            return None
            
        # Use the first available image
        image_id = images[0]['id']
        name = f'{project}/assets/{image_id}'
        print(f"Using image: {image_id}")
        
        # Get image pixels
        url = f'https://earthengine.googleapis.com/v1alpha/{name}:getPixels'
        body = {
            'fileFormat': 'PNG',
            'bandIds': dataset_info['bands'],
            'grid': {
                'dimensions': {'width': 1024, 'height': 1024}
            },
            'visualizationOptions': {
                'ranges': dataset_info['ranges']
            }
        }
        
        print("Requesting image...")
        response = session.post(url, json=body)
        
        if response.status_code == 200:
            print("Successfully received image")
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to get image: {response.status_code}")
            print(f"Response content: {response.content}")
            return None
            
    except Exception as e:
        print(f"Error getting Earth Engine image: {e}")
        return None

def test_earth_engine_api():
    # Service account setup
    KEY_PATH = Path("secrets/earth-engine-key.json")
    
    if not KEY_PATH.exists():
        raise FileNotFoundError(
            f"Service account key file not found at {KEY_PATH}. "
            "Please place your Earth Engine service account key in this location."
        )
    
    print("Initializing Earth Engine session...")
    credentials = service_account.Credentials.from_service_account_file(
        KEY_PATH,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    session = AuthorizedSession(credentials)
    
    # Updated datasets - removed Landsat 9
    datasets = [
        {
            "name": "Sentinel-2 Level 2A",
            "asset_id": "COPERNICUS/S2_SR",
            "bands": ['B4', 'B3', 'B2'],
            "ranges": [{'min': 0, 'max': 3000}],
            "filter": "CLOUDY_PIXEL_PERCENTAGE < 20"
        },
        {
            "name": "Sentinel-1 SAR GRD",
            "asset_id": "COPERNICUS/S1_GRD",
            "bands": ['VV', 'VH', 'VV'],
            "ranges": [{'min': -25, 'max': 0}],
            "filter": ""  # SAR works through clouds
        },
        {
            "name": "Sentinel-2 Level 1C",  # Added as backup
            "asset_id": "COPERNICUS/S2",
            "bands": ['B4', 'B3', 'B2'],
            "ranges": [{'min': 0, 'max': 3000}],
            "filter": "CLOUDY_PIXEL_PERCENTAGE < 20"
        }
    ]
    
    # Test locations in Ukraine with different altitudes
    locations = [
        {
            "name": "Odesa_Port",
            "lat": 46.4925,
            "lng": 30.7487,
            "altitudes": [100, 200, 500]  # meters above ground
        },
        {
            "name": "Odesa_City",
            "lat": 46.4825,
            "lng": 30.7233,
            "altitudes": [100, 200, 500]
        },
        {
            "name": "Odesa_Beach",
            "lat": 46.4603,
            "lng": 30.7658,
            "altitudes": [100, 200, 500]
        }
    ]
    
    for location in locations:
        print(f"\nTesting location: {location['name']}")
        
        for altitude in location['altitudes']:
            print(f"\nTesting at altitude: {altitude}m")
            
            for dataset in datasets:
                image = get_earth_engine_image(
                    location['lat'], 
                    location['lng'], 
                    altitude, 
                    session,
                    dataset
                )
                
                if image:
                    # Create output directory if it doesn't exist
                    output_dir = Path("output")
                    output_dir.mkdir(exist_ok=True)
                    
                    # Save the image with altitude in filename
                    filename = output_dir / f"{location['name'].lower()}_{altitude}m_{dataset['name'].lower().replace(' ', '_')}.png"
                    image.save(filename)
                    print(f"Image saved successfully to: {filename}")
                    
                    # Print image details
                    print(f"Image size: {image.size}")
                    print(f"Image mode: {image.mode}")
                else:
                    print(f"Failed to get image for {dataset['name']}")

if __name__ == "__main__":
    test_earth_engine_api() 