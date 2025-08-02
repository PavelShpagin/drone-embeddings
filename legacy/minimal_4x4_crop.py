#!/usr/bin/env python3
"""Minimal 4x4 Grid Test - 16 Small Tiles"""

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

# Larger coverage for 4x4 grid
coverage = 4.0  # 4km coverage
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

# 4x4 grid calculations
lon_step = (lon_max - lon_min) / 4
lat_step = (lat_max - lat_min) / 4

print("📥 Downloading 16 crops (4x4 grid)...")

# Create 4x4 grid regions and download
crops = []
for row in range(4):
    for col in range(4):
        print(f"   Crop {row*4 + col + 1:2d}/16: ", end="")
        
        # Calculate region bounds for this tile
        tile_lon_min = lon_min + col * lon_step
        tile_lon_max = lon_min + (col + 1) * lon_step
        tile_lat_min = lat_min + row * lat_step
        tile_lat_max = lat_min + (row + 1) * lat_step
        
        crop_region = ee.Geometry.Rectangle([
            tile_lon_min, tile_lat_min, 
            tile_lon_max, tile_lat_max
        ])
        
        url = image.getThumbURL({
            'region': crop_region,
            'dimensions': '512x512',
            'format': 'png'
        })
        
        response = requests.get(url, timeout=15)
        crop = Image.open(BytesIO(response.content))
        crops.append(crop)
        print("✅")

print("🔧 Concatenating 4x4 grid...")

# Create 2048x2048 final image (4 x 512 = 2048)
final = Image.new('RGB', (2048, 2048))

# Paste all 16 crops in 4x4 arrangement
for row in range(4):
    for col in range(4):
        crop_idx = row * 4 + col
        x_pos = col * 512
        y_pos = row * 512
        final.paste(crops[crop_idx], (x_pos, y_pos))

# Save
output_dir = Path("../data/gee_api")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{lat}, {lng}.4x4grid.jpg"
final.save(output_path, 'JPEG', quality=95)

print(f"✅ DONE: {output_path}")
print(f"📏 Size: {final.size}")
print(f"💾 File: {output_path.stat().st_size / 1024:.0f} KB")
print(f"🗺️  Coverage: {coverage}km × {coverage}km") 