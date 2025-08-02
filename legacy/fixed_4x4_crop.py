#!/usr/bin/env python3
"""Fixed 4x4 Grid Test - Correct Coordinate Mapping"""

import ee
import requests
from PIL import Image
from pathlib import Path
import json
from io import BytesIO
import traceback
import time

try:
    print("🔧 Starting FIXED script...")
    
    # Initialize GEE
    print("🔑 Loading credentials...")
    key_path = Path("../secrets/earth-engine-key.json")
    with open(key_path, 'r') as f:
        info = json.load(f)
    
    print("🌍 Initializing GEE...")
    credentials = ee.ServiceAccountCredentials(info['client_email'], str(key_path))
    ee.Initialize(credentials)
    print("✅ GEE initialized")

    # Coordinates
    lat, lng = 49.3721, 31.0945
    print(f"📍 Coordinates: {lat}, {lng}")

    # Coverage
    coverage = 3.0  # 3km coverage
    print(f"🗺️  Coverage: {coverage}km")
    
    region = ee.Geometry.Rectangle([
        lng - coverage/222, lat - coverage/222,
        lng + coverage/222, lat + coverage/222
    ])

    print("🛰️  Getting satellite image...")
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(region)
                 .filterDate('2024-06-01', '2024-08-31')
                 .first()
                 .select(['B4', 'B3', 'B2']))

    image = collection.divide(10000).visualize(bands=['B4', 'B3', 'B2'], min=0, max=0.25)
    print("✅ Image ready")

    # Get bounds
    print("📐 Calculating bounds...")
    bounds = region.bounds().getInfo()['coordinates'][0]
    lon_min, lat_min = bounds[0]
    lon_max, lat_max = bounds[2]
    print(f"   Bounds: {lon_min:.6f}, {lat_min:.6f} to {lon_max:.6f}, {lat_max:.6f}")

    # 4x4 grid calculations
    lon_step = (lon_max - lon_min) / 4
    lat_step = (lat_max - lat_min) / 4
    print(f"   Step size: lon={lon_step:.6f}, lat={lat_step:.6f}")

    print("📥 Downloading 16 crops (4x4 grid)...")
    print("🔧 FIXED: Using correct coordinate mapping!")

    # Create 4x4 grid regions and download
    crops = []
    for row in range(4):
        for col in range(4):
            crop_num = row*4 + col + 1
            print(f"   Crop {crop_num:2d}/16 (row {row}, col {col}): ", end="", flush=True)
            
            try:
                # FIXED COORDINATE CALCULATION
                # Row 0 should be NORTH (highest lat), Row 3 should be SOUTH (lowest lat)
                tile_lon_min = lon_min + col * lon_step
                tile_lon_max = lon_min + (col + 1) * lon_step
                tile_lat_min = lat_max - (row + 1) * lat_step  # FIXED: Start from lat_max
                tile_lat_max = lat_max - row * lat_step        # FIXED: Start from lat_max
                
                print(f"lat[{tile_lat_min:.4f}-{tile_lat_max:.4f}] ", end="")
                
                crop_region = ee.Geometry.Rectangle([
                    tile_lon_min, tile_lat_min, 
                    tile_lon_max, tile_lat_max
                ])
                
                url = image.getThumbURL({
                    'region': crop_region,
                    'dimensions': '512x512',
                    'format': 'png'
                })
                
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                
                crop = Image.open(BytesIO(response.content))
                crops.append(crop)
                print("✅")
                
                # Small delay to be nice to the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                # Add a placeholder black image
                crops.append(Image.new('RGB', (512, 512), (0, 0, 0)))

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
            print(f"   Pasted crop {crop_idx+1} at ({x_pos}, {y_pos})")

    # Save
    print("💾 Saving FIXED image...")
    output_dir = Path("../data/gee_api")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{lat}, {lng}.FIXED.4x4grid.jpg"
    final.save(output_path, 'JPEG', quality=95)

    print(f"✅ DONE: {output_path}")
    print(f"📏 Size: {final.size}")
    print(f"💾 File: {output_path.stat().st_size / 1024:.0f} KB")
    print(f"🗺️  Coverage: {coverage}km × {coverage}km")
    print("🔧 Fixed coordinate mapping: Row 0=North, Row 3=South")

except Exception as e:
    print(f"💥 FATAL ERROR: {e}")
    print("🔍 Traceback:")
    traceback.print_exc() 