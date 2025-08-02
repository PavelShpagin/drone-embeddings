#!/usr/bin/env python3
"""Minimal 4-Crop Test - Very Small Tiles"""

import ee
import requests
from PIL import Image
from pathlib import Path
import json
from io import BytesIO

# Initialize GEE
key_path = Path("../secrets/earth-engine-key.json")
with open(key_path, 'r') as f:
    info = json.load(f)
credentials = ee.ServiceAccountCredentials(info['client_email'], str(key_path))
ee.Initialize(credentials)
print("✅ GEE initialized")

# Coordinates
lat, lng = 50.4162, 30.8906

# Simple region (small coverage)
coverage = 2.0  # 2km only
region = ee.Geometry.Rectangle([
    lng - coverage/222, lat - coverage/222,
    lng + coverage/222, lat + coverage/222
])

# Simple image
collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(region)
             .filterDate('2024-06-01', '2024-08-31')
             .first()
             .select(['B4', 'B3', 'B2']))

image = collection.divide(10000).visualize(bands=['B4', 'B3', 'B2'], min=0, max=0.25)
print("✅ Image ready")

# Get bounds
bounds = region.bounds().getInfo()['coordinates'][0]
lon_min, lat_min = bounds[0]
lon_max, lat_max = bounds[2]

# 2x2 grid
lon_mid = (lon_min + lon_max) / 2
lat_mid = (lat_min + lat_max) / 2

regions = [
    ee.Geometry.Rectangle([lon_min, lat_mid, lon_mid, lat_max]),  # TL
    ee.Geometry.Rectangle([lon_mid, lat_mid, lon_max, lat_max]),  # TR
    ee.Geometry.Rectangle([lon_min, lat_min, lon_mid, lat_mid]),  # BL
    ee.Geometry.Rectangle([lon_mid, lat_min, lon_max, lat_mid])   # BR
]

print("📥 Downloading 4 small crops...")

# Download 4 small crops (512x512)
crops = []
for i, crop_region in enumerate(regions):
    print(f"   Crop {i+1}: ", end="")
    
    url = image.getThumbURL({
        'region': crop_region,
        'dimensions': '512x512',
        'format': 'png'
    })
    
    response = requests.get(url, timeout=15)
    crop = Image.open(BytesIO(response.content))
    crops.append(crop)
    print("✅")

print("🔧 Concatenating...")

# Create 1024x1024 final image
final = Image.new('RGB', (1024, 1024))
final.paste(crops[0], (0, 0))      # TL
final.paste(crops[1], (512, 0))    # TR
final.paste(crops[2], (0, 512))    # BL
final.paste(crops[3], (512, 512))  # BR

# Save
output_dir = Path("../data/gee_api")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{lat}, {lng}.minimal4crop.jpg"
final.save(output_path, 'JPEG', quality=95)

print(f"✅ DONE: {output_path}")
print(f"📏 Size: {final.size}")
print(f"💾 File: {output_path.stat().st_size / 1024:.0f} KB") 