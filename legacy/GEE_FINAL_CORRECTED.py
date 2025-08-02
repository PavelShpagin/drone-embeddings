#!/usr/bin/env python3
"""
GEE FINAL CORRECTED - Fixed Vertical Alignment
---------------------------------------------
Fixes the vertical alignment issue from Test 4 by correcting
the coordinate mapping and tile positioning logic.
"""

import ee
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import json
import math
import traceback
from retry import retry

# --- Configuration ---
HEIGHT_KM = 3.0
FOV_DEGREES = 60
GRID_DIMENSION = 6
TILE_PIXELS = 512
SAVE_DIRECTORY = "../data/gee_final_corrected"

# --- GEE Initialization ---
try:
    print("🌍 Initializing Google Earth Engine...")
    SECRETS_PATH = Path("../secrets/earth-engine-key.json")
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(f"GEE secrets not found at: {SECRETS_PATH}")
    
    with open(SECRETS_PATH, 'r') as f:
        info = json.load(f)
    
    credentials = ee.ServiceAccountCredentials(info['client_email'], str(SECRETS_PATH))
    ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
    print("✅ GEE Initialized (High-Volume Endpoint)")

except Exception as e:
    print(f"💥 FATAL ERROR during GEE initialization: {e}")
    exit(1)

@retry(tries=3, delay=5, backoff=2, jitter=(1, 3))
def get_gee_tile(image: ee.Image, region: ee.Geometry, tile_size: int) -> Image.Image:
    """Downloads a single tile from GEE with retry logic."""
    url = image.getThumbURL({
        'region': region.getInfo()['coordinates'],
        'dimensions': f'{tile_size}x{tile_size}',
        'format': 'png'
    })
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def create_robust_image(region: ee.Geometry) -> ee.Image:
    """Creates a robust image using the proven harmonized approach."""
    print("🔧 Creating harmonized composite...")
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate('2023-01-01', '2024-12-31')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .select(['B2', 'B3', 'B4', 'B8']))
    
    count = collection.size().getInfo()
    print(f"   Found {count} compatible images")
    
    if count > 0:
        composite = collection.median()
        print("   ✅ Created median composite")
        return composite.visualize(bands=['B4', 'B3', 'B2'], min=[300, 300, 200], max=[2500, 2500, 2500])
    else:
        raise Exception("No suitable images found")

def get_final_corrected_image(lat: float, lng: float) -> Path:
    """Generate the final corrected image with proper vertical alignment."""
    final_pixels = GRID_DIMENSION * TILE_PIXELS
    print("\n" + "="*70)
    print(f"🛰️  FINAL CORRECTED GEE Sample ({lat:.6f}, {lng:.6f})")
    print(f"🎯 Fixing vertical alignment issue from Test 4")
    print("="*70)

    # Calculate coverage
    total_coverage_m = 2 * (HEIGHT_KM * 1000) * math.tan(math.radians(FOV_DEGREES / 2))
    print(f"🗺️  Coverage: {total_coverage_m / 1000:.6f} km")

    # Use GEE's native buffer approach (from Test 4)
    center_point = ee.Geometry.Point([lng, lat])
    buffer_radius_m = total_coverage_m / 2
    buffered_region = center_point.buffer(buffer_radius_m, 1)
    bounding_box = buffered_region.bounds(1)
    
    print(f"🔵 Buffer radius: {buffer_radius_m:.1f}m")

    # Create image
    visualized_image = create_robust_image(bounding_box)

    # CORRECTED coordinate mapping - fix the vertical alignment issue
    print("📥 Downloading tiles with CORRECTED alignment...")
    
    # Calculate lat/lng deltas with proper precision
    lat_radians = math.radians(lat)
    meters_per_degree_lat = 111132.92 - 559.82 * math.cos(2 * lat_radians) + 1.175 * math.cos(4 * lat_radians)
    meters_per_degree_lng = 111412.84 * math.cos(lat_radians) - 93.5 * math.cos(3 * lat_radians)
    
    lat_delta = (total_coverage_m / 2) / meters_per_degree_lat
    lng_delta = (total_coverage_m / 2) / meters_per_degree_lng
    
    print(f"🌐 Deltas: lat={lat_delta:.8f}°, lng={lng_delta:.8f}°")
    
    # Define grid boundaries
    min_lat = lat - lat_delta
    max_lat = lat + lat_delta
    min_lng = lng - lng_delta
    max_lng = lng + lng_delta
    
    # CRITICAL FIX: Proper tile coordinate calculation
    tile_lat_step = (max_lat - min_lat) / GRID_DIMENSION
    tile_lng_step = (max_lng - min_lng) / GRID_DIMENSION
    
    print(f"📏 Grid bounds: lat[{min_lat:.6f}, {max_lat:.6f}], lng[{min_lng:.6f}, {max_lng:.6f}]")
    print(f"📏 Tile steps: lat={tile_lat_step:.8f}°, lng={tile_lng_step:.8f}°")
    
    tiles = []
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            tile_num = row * GRID_DIMENSION + col + 1
            print(f"   Tile {tile_num:2d}/{GRID_DIMENSION**2} (r{row},c{col})...", end="", flush=True)
            try:
                # CORRECTED: Proper North-South mapping
                # North = max_lat, South = min_lat
                # Row 0 = North (top), Row 5 = South (bottom)
                tile_max_lat = max_lat - row * tile_lat_step        # North edge
                tile_min_lat = max_lat - (row + 1) * tile_lat_step  # South edge
                
                # West-East mapping (unchanged)
                tile_min_lng = min_lng + col * tile_lng_step        # West edge  
                tile_max_lng = min_lng + (col + 1) * tile_lng_step  # East edge
                
                print(f"[{tile_min_lng:.6f},{tile_min_lat:.6f},{tile_max_lng:.6f},{tile_max_lat:.6f}]", end="")
                
                # Create tile region with corrected coordinates
                tile_region = ee.Geometry.Rectangle([
                    tile_min_lng, tile_min_lat, tile_max_lng, tile_max_lat
                ], proj='EPSG:4326', geodesic=False)
                
                tile_image = get_gee_tile(visualized_image, tile_region, TILE_PIXELS)
                tiles.append(tile_image)
                print(" ✅")
            except Exception as e:
                print(f" ❌ Error: {e}")
                placeholder = Image.new('RGB', (TILE_PIXELS, TILE_PIXELS), (128, 64, 64))
                tiles.append(placeholder)
    
    # Stitch with corrected positioning
    print("🔧 Stitching with corrected alignment...")
    final_image = Image.new('RGB', (final_pixels, final_pixels))
    
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            tile_index = row * GRID_DIMENSION + col
            paste_x = col * TILE_PIXELS
            paste_y = row * TILE_PIXELS  # Row 0 = top, Row 5 = bottom
            
            final_image.paste(tiles[tile_index], (paste_x, paste_y))
            print(f"   Pasted tile {tile_index} at ({paste_x}, {paste_y})")

    output_dir = Path(SAVE_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"final_corrected_{lat:.6f}_{lng:.6f}.jpg"
    final_image.save(output_path, 'JPEG', quality=95)
    
    print(f"\n✅ FINAL CORRECTED image saved: {output_path}")
    print(f"📏 Size: {final_image.size} | 💾 File: {output_path.stat().st_size / 1024:.0f} KB")
    
    return output_path

if __name__ == '__main__':
    try:
        test_lat, test_lng = 50.4162, 30.8906
        result_file = get_final_corrected_image(lat=test_lat, lng=test_lng)
        print(f"\n🎯 FINAL RESULT: {result_file}")
        print("\n🎉 This should now have perfect vertical alignment!")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        traceback.print_exc() 