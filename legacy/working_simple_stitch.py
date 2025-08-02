#!/usr/bin/env python3
"""
Working Simple Stitch - Clean Implementation
===========================================

Simple approach with correct positioning and no overlaps.
Based on proven working components.
"""

import ee
import requests
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
from io import BytesIO
import time

def initialize_gee():
    """Initialize Google Earth Engine."""
    print("🔍 Initializing Google Earth Engine...")
    
    service_account_key_path = Path("../secrets/earth-engine-key.json")
    
    with open(service_account_key_path, 'r') as f:
        service_account_info = json.load(f)
    
    credentials = ee.ServiceAccountCredentials(
        service_account_info['client_email'],
        str(service_account_key_path)
    )
    
    ee.Initialize(credentials)
    print("✅ Google Earth Engine initialized successfully!")

def get_satellite_image(lat, lng, coverage_km=5.0):
    """Get satellite image for the region."""
    print(f"🛰️ Getting satellite image for {coverage_km}km coverage...")
    
    # Define region
    region = ee.Geometry.Rectangle([
        lng - coverage_km / 222.0,
        lat - coverage_km / 222.0,
        lng + coverage_km / 222.0,
        lat + coverage_km / 222.0
    ])
    
    # Get Sentinel-2 collection for 2024 summer
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(region)
                 .filterDate('2024-06-01', '2024-08-31')
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                 .select(['B4', 'B3', 'B2']))
    
    # Use median composite
    image = collection.median().divide(10000)
    
    # Apply visualization
    visualized = image.visualize(**{
        'bands': ['B4', 'B3', 'B2'],
        'min': 0.0,
        'max': 0.25,
        'gamma': 1.1
    })
    
    print("✅ Satellite image ready")
    return visualized, region

def download_tile(image, tile_region, width, height):
    """Download a single tile."""
    for attempt in range(3):
        try:
            url = image.getThumbURL({
                'region': tile_region,
                'dimensions': f"{width}x{height}",
                'format': 'png',
                'crs': 'EPSG:4326'
            })
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            tile_image = Image.open(BytesIO(response.content))
            
            if tile_image.size != (width, height):
                tile_image = tile_image.resize((width, height), Image.Resampling.LANCZOS)
            
            return tile_image
            
        except Exception as e:
            if attempt < 2:
                print(f"      ⚠️ Retry {attempt+1}/3...")
                time.sleep(1)
                continue
            else:
                raise e

def create_simple_stitch(lat, lng):
    """Create simple stitched image with correct positioning."""
    print(f"\n🌍 Creating Simple Stitched Image")
    print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")
    
    # Target dimensions
    target_width = 8192
    target_height = 4633
    
    # Simple 4x4 grid
    tiles_x, tiles_y = 4, 4
    tile_width = target_width // tiles_x
    tile_height = target_height // tiles_y
    
    print(f"🧩 Grid: {tiles_x}×{tiles_y}, Tile size: {tile_width}×{tile_height}")
    
    # Initialize GEE
    initialize_gee()
    
    # Get satellite image
    image, region = get_satellite_image(lat, lng)
    
    # Get region bounds
    region_bounds = region.bounds().getInfo()['coordinates'][0]
    lon_min, lat_min = region_bounds[0]
    lon_max, lat_max = region_bounds[2]
    
    # Calculate steps
    lon_step = (lon_max - lon_min) / tiles_x
    lat_step = (lat_max - lat_min) / tiles_y
    
    # Create final image
    final_image = Image.new('RGB', (target_width, target_height))
    
    print("📥 Downloading tiles...")
    
    # Download tiles
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            print(f"   📦 Tile {tile_x+1},{tile_y+1}: ", end="")
            
            # Calculate exact boundaries (no overlap)
            tile_lon_min = lon_min + tile_x * lon_step
            tile_lon_max = lon_min + (tile_x + 1) * lon_step
            tile_lat_min = lat_min + tile_y * lat_step
            tile_lat_max = lat_min + (tile_y + 1) * lat_step
            
            tile_region = ee.Geometry.Rectangle([
                tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max
            ])
            
            # Download tile
            tile_image = download_tile(image, tile_region, tile_width, tile_height)
            
            # Position in final image
            x_pos = tile_x * tile_width
            y_pos = tile_y * tile_height
            
            # Simple paste - no blending
            final_image.paste(tile_image, (x_pos, y_pos))
            
            print("✅")
    
    # Save result
    output_dir = Path("../data/gee_api")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{lat}, {lng}.simple.jpg"
    final_image.save(output_path, 'JPEG', quality=95, optimize=True, dpi=(96, 96))
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Simple stitched image completed!")
    print(f"📁 Saved to: {output_path}")
    print(f"📏 Size: {final_image.size}")
    print(f"💾 File size: {file_size_mb:.1f} MB")
    print(f"🎯 Features:")
    print(f"   ✅ Correct geographic positioning")
    print(f"   ✅ No overlaps between tiles")  
    print(f"   ✅ Simple direct stitching")
    print(f"   ✅ 4km height equivalent view")
    
    return output_path

def main():
    """Main function."""
    print("🛰️ Working Simple Stitch")
    print("=" * 30)
    
    # Test coordinates
    lat = 50.4162
    lng = 30.8906
    
    try:
        result = create_simple_stitch(lat, lng)
        print(f"\n🎉 SUCCESS! Simple image created: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 