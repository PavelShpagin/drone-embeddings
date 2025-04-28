import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def get_google_elevation(lat: float, lng: float) -> float | None:
    """
    Fetches the ground elevation for a given latitude and longitude using the Google Maps Elevation API.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.

    Returns:
        The elevation in meters above mean sea level, or None if the request fails
        or the API key is missing.
    """
    if not GOOGLE_MAPS_API_KEY:
        print("Error: GOOGLE_MAPS_API_KEY not found in environment variables.")
        return None

    base_url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {
        "locations": f"{lat},{lng}",
        "key": GOOGLE_MAPS_API_KEY
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        if data["status"] == "OK" and data["results"]:
            elevation = data["results"][0]["elevation"]
            return float(elevation)
        else:
            print(f"Error fetching elevation: {data.get('status', 'Unknown status')}")
            if 'error_message' in data:
                print(f"API Error Message: {data['error_message']}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error during elevation API request: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing elevation API response: {e}")
        return None

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Example: Mount Everest approx location
    test_lat = 27.9881
    test_lng = 86.9250
    elevation = get_google_elevation(test_lat, test_lng)
    if elevation is not None:
        print(f"Elevation at ({test_lat}, {test_lng}): {elevation:.2f} meters")
    else:
        print("Failed to retrieve elevation.")

    # Example: Dead Sea approx location (negative elevation)
    test_lat_ds = 31.5590
    test_lng_ds = 35.4732
    elevation_ds = get_google_elevation(test_lat_ds, test_lng_ds)
    if elevation_ds is not None:
        print(f"Elevation at ({test_lat_ds}, {test_lng_ds}): {elevation_ds:.2f} meters")
    else:
        print("Failed to retrieve elevation.")
