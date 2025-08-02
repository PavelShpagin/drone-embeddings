#!/usr/bin/env python3
"""
GEE PRECISION TEST 2 - Web Mercator Projection
----------------------------------------------
Tests using Web Mercator (EPSG:3857) for better pixel alignment
and avoiding UTM zone edge effects that can cause rounding errors.
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
SAVE_DIRECTORY = "../data/gee_precision_test_2"

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

def wgs84_to_web_mercator(lat: float, lng: float) -> tuple:
    """Convert WGS84 coordinates to Web Mercator (EPSG:3857)."""
    # Web Mercator formulas
    x = lng * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y

def get_precision_test_2(lat: float, lng: float) -> Path:
    """Test Web Mercator projection for better pixel alignment."""
    final_pixels = GRID_DIMENSION * TILE_PIXELS
    print("\n" + "="*70)
    print(f"🔬 PRECISION TEST 2: Web Mercator Projection ({lat:.6f}, {lng:.6f})")
    print("="*70)

    # Calculate coverage in meters
    total_coverage_m = 2 * (HEIGHT_KM * 1000) * math.tan(math.radians(FOV_DEGREES / 2))
    print(f"🗺️  Coverage: {total_coverage_m / 1000:.6f} km")

    # Convert center point to Web Mercator
    center_x_mercator, center_y_mercator = wgs84_to_web_mercator(lat, lng)
    print(f"📍 Web Mercator Center: ({center_x_mercator:.3f}, {center_y_mercator:.3f})")

    # Define Web Mercator projection
    web_mercator_proj = ee.Projection('EPSG:3857')
    
    # Calculate precise grid boundaries in Web Mercator
    half_coverage_m = total_coverage_m / 2
    min_x_mercator = center_x_mercator - half_coverage_m
    max_x_mercator = center_x_mercator + half_coverage_m
    min_y_mercator = center_y_mercator - half_coverage_m
    max_y_mercator = center_y_mercator + half_coverage_m

    print(f"🔲 Web Mercator Bounds:")
    print(f"   X: {min_x_mercator:.3f} to {max_x_mercator:.3f}")
    print(f"   Y: {min_y_mercator:.3f} to {max_y_mercator:.3f}")

    # Create region in Web Mercator, then transform to WGS84 for filtering
    region_mercator = ee.Geometry.Rectangle([
        min_x_mercator, min_y_mercator, max_x_mercator, max_y_mercator
    ], proj=web_mercator_proj, geodesic=False)
    
    region_wgs84_for_filtering = region_mercator.transform('EPSG:4326', 1)

    # Create image
    visualized_image = create_robust_image(region_wgs84_for_filtering)

    # Download tiles with Web Mercator grid
    print(f"📥 Downloading {GRID_DIMENSION**2} tiles with Web Mercator grid...")
    x_step = (max_x_mercator - min_x_mercator) / GRID_DIMENSION
    y_step = (max_y_mercator - min_y_mercator) / GRID_DIMENSION
    tiles = []
    
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            tile_num = row * GRID_DIMENSION + col + 1
            print(f"   Tile {tile_num:2d}/{GRID_DIMENSION**2}...", end="", flush=True)
            try:
                # Calculate tile boundaries in Web Mercator
                tile_min_x = min_x_mercator + col * x_step
                tile_max_x = min_x_mercator + (col + 1) * x_step
                tile_min_y = max_y_mercator - (row + 1) * y_step  # Y axis flipped
                tile_max_y = max_y_mercator - row * y_step
                
                tile_region_mercator = ee.Geometry.Rectangle([
                    tile_min_x, tile_min_y, tile_max_x, tile_max_y
                ], proj=web_mercator_proj, geodesic=False)
                
                tile_image = get_gee_tile(visualized_image, tile_region_mercator, TILE_PIXELS)
                tiles.append(tile_image)
                print(" ✅")
            except Exception as e:
                print(f" ❌ Error: {e}")
                placeholder = Image.new('RGB', (TILE_PIXELS, TILE_PIXELS), (32, 64, 32))
                tiles.append(placeholder)
    
    # Stitch and save
    print("🔧 Stitching Web Mercator test image...")
    final_image = Image.new('RGB', (final_pixels, final_pixels))
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            final_image.paste(tiles[row * GRID_DIMENSION + col], (col * TILE_PIXELS, row * TILE_PIXELS))

    output_dir = Path(SAVE_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"web_mercator_test_{lat:.6f}_{lng:.6f}.jpg"
    final_image.save(output_path, 'JPEG', quality=95)
    
    print(f"\n✅ PRECISION TEST 2 Complete: {output_path}")
    print(f"📏 Size: {final_image.size} | 💾 File: {output_path.stat().st_size / 1024:.0f} KB")
    
    return output_path

if __name__ == '__main__':
    try:
        test_lat, test_lng = 50.4162, 30.8906
        result_file = get_precision_test_2(lat=test_lat, lng=test_lng)
        print(f"\n🎯 Result: {result_file}")
    except Exception as e:
        print(f"\n💥 Error in precision test 2: {e}")
        traceback.print_exc() 