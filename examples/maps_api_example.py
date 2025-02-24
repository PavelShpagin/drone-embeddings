import os
from dotenv import dotenv_values, load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

load_dotenv()

# Load secrets from .env file as a dictionary
secrets = dotenv_values(".env")
API_KEY = secrets.get("GOOGLE_MAPS_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in .env file")

def create_session_with_retry():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def get_static_map(lat, lng, zoom=50, size="1024x1024", scale=1):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        'center': f"{lat},{lng}",
        'zoom': zoom,
        'size': size,
        'format': 'png',
        'key': API_KEY
    }
    
    session = create_session_with_retry()
    
    try:
        response = session.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        if response.status_code == 200:
            filename = f"snapshot_{lat}_{lng}_hq.png"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Saved high-quality image as {filename}")
            return filename
        
    except requests.exceptions.ConnectionError as e:
        print("Connection error. Please check your internet connection.")
        print(f"Error details: {e}")
    except requests.exceptions.Timeout:
        print("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    
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