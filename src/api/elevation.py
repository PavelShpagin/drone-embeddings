import requests
from ..config import API_KEY

def get_elevation(lat, lng):
    """Get elevation from Google Maps Elevation API"""
    base_url = "https://maps.googleapis.com/maps/api/elevation/json"
    
    params = {
        'locations': f"{lat},{lng}",
        'key': API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if data['status'] == 'OK':
            return data['results'][0]['elevation']
        else:
            print(f"Error: {data['status']}")
            return None
            
    except Exception as e:
        print(f"Error fetching elevation: {e}")
        return None