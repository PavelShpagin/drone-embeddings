import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv, dotenv_values

def display_maps(locations, api_key):
    for i, location in enumerate(locations):
        lat, lon = location
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=15&size=400x400&maptype=satellite&key={api_key}"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"Location {i+1}: ({lat}, {lon})")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    locations = [
        (50.4162, 30.8906),  # Agricultural fields east of Kyiv
        (48.9483, 29.7241),  # Farmland in Vinnytsia Oblast
        (49.3721, 31.0945),  # Agricultural area in Cherkasy Oblast
        (48.5673, 33.4218),  # Farmland in Dnipropetrovsk Oblast
        (46.6234, 32.7851),  # Agricultural fields in Kherson Oblast
        (49.8234, 25.3612),  # Farmland west of Ternopil
        (50.7156, 29.2367),  # Agricultural area in Zhytomyr Oblast
        (51.4523, 32.8945),  # Farmland in Chernihiv Oblast
        (48.2367, 35.7823),  # Agricultural fields in Zaporizhzhia Oblast
        (47.8945, 30.2367),  # Farmland in Mykolaiv Oblast
    ]
    load_dotenv()

    # Load secrets from .env file as a dictionary
    secrets = dotenv_values(".env")
    API_KEY = secrets.get("GOOGLE_MAPS_API_KEY")
    display_maps(locations, API_KEY)

