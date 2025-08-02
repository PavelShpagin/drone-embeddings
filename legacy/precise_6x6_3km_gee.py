#!/usr/bin/env python3
"""High-Precision 6x6 GEE Script - 3km Height Perspective"""

import ee
import requests
from PIL import Image
from pathlib import Path
import json
from io import BytesIO
import traceback
import time
import math

try:
    print("🛰️  Starting HIGH-PRECISION 6x6 GEE Script (3km height)")
    
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
    lat, lng = 50.4162, 30.8906  # Back to original coordinates
    print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")

    # HIGH-PRECISION COORDINATE CONVERSION
    # At 3km height, calculate field of view
    height_km = 3.0
    
    # Assume 60° field of view (typical aerial camera)
    # Ground coverage = 2 * height * tan(FOV/2)
    fov_degrees = 60
    fov_radians = math.radians(fov_degrees)
    total_coverage_km = 2 * height_km * math.tan(fov_radians / 2)
    
    print(f"🎯 Height: {height_km}km")
    print(f"📐 Field of view: {fov_degrees}°")
    print(f"🗺️  Ground coverage: {total_coverage_km:.3f}km × {total_coverage_km:.3f}km")

    # PRECISE COORDINATE CONVERSION
    # Latitude: 1 degree ≈ 111.32 km (constant)
    # Longitude: 1 degree ≈ 111.32 * cos(latitude) km
    km_per_degree_lat = 111.32
    km_per_degree_lng = 111.32 * math.cos(math.radians(lat))
    
    print(f"🧮 Conversion factors at lat {lat:.4f}°:")
    print(f"   Latitude:  1° = {km_per_degree_lat:.3f} km")
    print(f"   Longitude: 1° = {km_per_degree_lng:.3f} km")

    # Calculate precise offsets
    half_coverage = total_coverage_km / 2
    lat_offset = half_coverage / km_per_degree_lat
    lng_offset = half_coverage / km_per_degree_lng
    
    print(f"📏 Half coverage: {half_coverage:.3f} km")
    print(f"   Lat offset:  ±{lat_offset:.8f}°")
    print(f"   Lng offset:  ±{lng_offset:.8f}°")

    # Define precise region bounds
    lat_min = lat - lat_offset
    lat_max = lat + lat_offset
    lng_min = lng - lng_offset
    lng_max = lng + lng_offset
    
    print(f"🎯 Precise bounds:")
    print(f"   Lat: {lat_min:.8f} to {lat_max:.8f}")
    print(f"   Lng: {lng_min:.8f} to {lng_max:.8f}")

    region = ee.Geometry.Rectangle([lng_min, lat_min, lng_max, lat_max])

    print("🛰️  Getting satellite image...")
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(region)
                 .filterDate('2024-06-01', '2024-08-31')
                 .sort('CLOUDY_PIXEL_PERCENTAGE')
                 .first()
                 .select(['B4', 'B3', 'B2']))

    image = collection.divide(10000).visualize(bands=['B4', 'B3', 'B2'], min=0, max=0.25)
    print("✅ Image ready")

    # 6x6 grid calculations
    lat_step = (lat_max - lat_min) / 6
    lng_step = (lng_max - lng_min) / 6
    
    print(f"📐 6x6 Grid steps:")
    print(f"   Lat step: {lat_step:.8f}° ({lat_step * km_per_degree_lat * 1000:.1f}m)")
    print(f"   Lng step: {lng_step:.8f}° ({lng_step * km_per_degree_lng * 1000:.1f}m)")

    print("📥 Downloading 36 crops (6x6 grid)...")

    # Download 6x6 grid
    crops = []
    for row in range(6):
        for col in range(6):
            crop_num = row * 6 + col + 1
            print(f"   Crop {crop_num:2d}/36 (R{row}C{col}): ", end="", flush=True)
            
            try:
                # FIXED COORDINATE CALCULATION - High precision
                # Row 0 = North (highest lat), Row 5 = South (lowest lat)
                tile_lat_min = lat_max - (row + 1) * lat_step
                tile_lat_max = lat_max - row * lat_step
                tile_lng_min = lng_min + col * lng_step
                tile_lng_max = lng_min + (col + 1) * lng_step
                
                # Center coordinates for verification
                center_lat = (tile_lat_min + tile_lat_max) / 2
                center_lng = (tile_lng_min + tile_lng_max) / 2
                
                print(f"lat[{tile_lat_min:.5f}-{tile_lat_max:.5f}] ", end="")
                
                crop_region = ee.Geometry.Rectangle([
                    tile_lng_min, tile_lat_min, 
                    tile_lng_max, tile_lat_max
                ])
                
                url = image.getThumbURL({
                    'region': crop_region,
                    'dimensions': '512x512',
                    'format': 'png'
                })
                
                response = requests.get(url, timeout=25)
                response.raise_for_status()
                
                crop = Image.open(BytesIO(response.content))
                crops.append(crop)
                print("✅")
                
                # Small delay for API stability
                time.sleep(0.15)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                # Add placeholder
                crops.append(Image.new('RGB', (512, 512), (64, 64, 64)))

    print("🔧 Concatenating 6x6 grid...")

    # Create final image (6 x 512 = 3072)
    final_size = 512 * 6
    final = Image.new('RGB', (final_size, final_size))

    # Paste all 36 crops in 6x6 arrangement
    for row in range(6):
        for col in range(6):
            crop_idx = row * 6 + col
            x_pos = col * 512
            y_pos = row * 512
            final.paste(crops[crop_idx], (x_pos, y_pos))

    # Save
    print("💾 Saving PRECISE 6x6 image...")
    output_dir = Path("../data/gee_api")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{lat}, {lng}.PRECISE.6x6.3km.jpg"
    final.save(output_path, 'JPEG', quality=95)

    print(f"✅ DONE: {output_path}")
    print(f"📏 Final size: {final.size}")
    print(f"💾 File size: {output_path.stat().st_size / 1024:.0f} KB")
    print(f"🎯 Height perspective: {height_km}km")
    print(f"🗺️  Coverage: {total_coverage_km:.3f}km × {total_coverage_km:.3f}km")
    print(f"📐 Resolution: ~{(total_coverage_km * 1000) / final_size:.1f}m per pixel")
    
    # Precision verification
    actual_lat_span = lat_max - lat_min
    actual_lng_span = lng_max - lng_min
    actual_lat_km = actual_lat_span * km_per_degree_lat
    actual_lng_km = actual_lng_span * km_per_degree_lng
    
    print(f"🔍 Precision verification:")
    print(f"   Lat span: {actual_lat_span:.8f}° = {actual_lat_km:.3f}km")
    print(f"   Lng span: {actual_lng_span:.8f}° = {actual_lng_km:.3f}km")

except Exception as e:
    print(f"💥 FATAL ERROR: {e}")
    print("🔍 Traceback:")
    traceback.print_exc() 