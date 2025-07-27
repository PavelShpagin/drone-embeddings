import requests
import json
from IPython.display import Image, display
from src.config import PLANET_API_KEY  # Your API key

# --- STEP 1: Search for PSScene imagery near Kyiv ---
search_url = 'https://api.planet.com/data/v1/quick-search'

# Define the search payload:
# - We search for PSScene items within a point at [lon, lat] for Kyiv.
# - DateRange and cloud cover filters narrow results.
payload = {
    "item_types": ["PSScene"],
    "filter": {
        "type": "AndFilter",
        "config": [
            {
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": {
                    "type": "Point",
                    "coordinates": [30.5234, 50.4501]  # [longitude, latitude] for Kyiv
                }
            },
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": "2025-02-20T00:00:00.000Z",
                    "lte": "2025-02-24T23:59:59.999Z"
                }
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": 0.1}  # scenes with less than 10% cloud cover
            }
        ]
    }
}

# Send the POST request (with Basic Auth: API key as username, empty password)
search_response = requests.post(search_url, json=payload, auth=(PLANET_API_KEY, ''))
if search_response.status_code == 200:
    search_results = search_response.json()
    # --- STEP 2: Loop through each returned scene ---
    for feature in search_results.get("features", []):
        feature_id = feature.get("id")
        asset_types = feature.get("assets", [])
        # Default thumbnail URL provided in the _links section
        original_thumb_url = feature.get("_links", {}).get("thumbnail")
        
        print("Feature ID:", feature_id)
        print("Available asset types:", asset_types)
        print("Original Thumbnail URL:", original_thumb_url)
        
        if original_thumb_url:
            # --- STEP 3: Modify thumbnail URL to request a "zoomed in" image ---
            # We want a small ground footprint of roughly 150 m.
            # For a 512-pixel wide image, set scale = ~0.3 m/px (512 x 0.3 ≈ 153.6 m)
            modified_thumb_url = f"{original_thumb_url}?width=512&scale=0.3"
            print("Modified Thumbnail URL:", modified_thumb_url)
            
            # --- STEP 4: Fetch the modified thumbnail using authentication ---
            thumb_response = requests.get(modified_thumb_url, auth=(PLANET_API_KEY, ''))
            if thumb_response.status_code == 200:
                # Display the image (in a Jupyter environment)
                with open(f"thumbnail_{feature_id}.png", 'wb') as f:
                    f.write(thumb_response.content)
                print(f"Thumbnail saved as thumbnail_{feature_id}.png")
            else:
                print("Error fetching thumbnail:", thumb_response.status_code, thumb_response.text)
else:
    print(f"Error: {search_response.status_code} - {search_response.text}")
