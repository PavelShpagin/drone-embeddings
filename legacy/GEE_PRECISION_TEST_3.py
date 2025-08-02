#!/usr/bin/env python3
"""
GEE PRECISION TEST 3 - Integer Grid Positioning
-----------------------------------------------
Tests using exact integer grid positioning to eliminate floating point errors
by working with pixel-aligned boundaries and integer coordinates.
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
SAVE_DIRECTORY = "../data/gee_precision_test_3"

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

def get_utm_epsg_code(lat: float, lng: float) -> str:
    """Calculates the EPSG code for the UTM zone of a given lat/lng."""
    utm_zone = math.floor((lng + 180) / 6) + 1
    return f"EPSG:{32600 + utm_zone if lat >= 0 else 32700 + utm_zone}"

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

def get_precision_test_3(lat: float, lng: float) -> Path:
    """Test exact integer grid positioning."""
    final_pixels = GRID_DIMENSION * TILE_PIXELS
    print("\n" + "="*70)
    print(f"🔬 PRECISION TEST 3: Integer Grid Positioning ({lat:.6f}, {lng:.6f})")
    print("="*70)

    # Calculate coverage in meters - round to nearest 10m to ensure integer alignment
    total_coverage_m_raw = 2 * (HEIGHT_KM * 1000) * math.tan(math.radians(FOV_DEGREES / 2))
    total_coverage_m = round(total_coverage_m_raw / 10) * 10  # Round to nearest 10m
    print(f"🗺️  Raw coverage: {total_coverage_m_raw / 1000:.6f} km")
    print(f"🗺️  Rounded coverage: {total_coverage_m / 1000:.6f} km")

    # UTM projection
    utm_epsg = get_utm_epsg_code(lat, lng)
    utm_projection = ee.Projection(utm_epsg)
    
    # Get UTM coordinates
    center_point_wgs84 = ee.Geometry.Point([lng, lat])
    center_point_utm = center_point_wgs84.transform(utm_projection, 1)
    easting_float, northing_float = center_point_utm.coordinates().getInfo()
    
    # Round UTM coordinates to nearest meter for exact integer positioning
    easting_int = round(easting_float)
    northing_int = round(northing_float)
    print(f"📍 UTM Coordinates (integer):")
    print(f"   Raw Easting:  {easting_float:.6f} → {easting_int}")
    print(f"   Raw Northing: {northing_float:.6f} → {northing_int}")

    # Calculate exact integer grid boundaries
    half_coverage_int = total_coverage_m // 2  # Integer division for exact alignment
    min_easting_int = easting_int - half_coverage_int
    max_easting_int = easting_int + half_coverage_int
    min_northing_int = northing_int - half_coverage_int
    max_northing_int = northing_int + half_coverage_int

    print(f"🔲 Integer Grid Bounds:")
    print(f"   Easting:  {min_easting_int} to {max_easting_int} ({max_easting_int - min_easting_int}m)")
    print(f"   Northing: {min_northing_int} to {max_northing_int} ({max_northing_int - min_northing_int}m)")

    # Create region with exact integer boundaries
    region_utm = ee.Geometry.Rectangle([
        float(min_easting_int), float(min_northing_int),
        float(max_easting_int), float(max_northing_int)
    ], proj=utm_projection, geodesic=False)
    
    region_wgs84_for_filtering = region_utm.transform('EPSG:4326', 1)

    # Create image
    visualized_image = create_robust_image(region_wgs84_for_filtering)

    # Download tiles with exact integer grid
    print(f"📥 Downloading {GRID_DIMENSION**2} tiles with integer grid...")
    total_coverage_int = max_easting_int - min_easting_int
    tile_size_int = total_coverage_int // GRID_DIMENSION  # Exact integer tile size
    
    print(f"🔢 Tile size: {tile_size_int}m x {tile_size_int}m (exact integer)")
    
    tiles = []
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            tile_num = row * GRID_DIMENSION + col + 1
            print(f"   Tile {tile_num:2d}/{GRID_DIMENSION**2}...", end="", flush=True)
            try:
                # Calculate exact integer tile boundaries
                tile_min_e_int = min_easting_int + col * tile_size_int
                tile_max_e_int = min_easting_int + (col + 1) * tile_size_int
                tile_min_n_int = max_northing_int - (row + 1) * tile_size_int
                tile_max_n_int = max_northing_int - row * tile_size_int
                
                print(f"[{tile_min_e_int},{tile_min_n_int},{tile_max_e_int},{tile_max_n_int}]", end="")
                
                tile_region_utm = ee.Geometry.Rectangle([
                    float(tile_min_e_int), float(tile_min_n_int),
                    float(tile_max_e_int), float(tile_max_n_int)
                ], proj=utm_projection, geodesic=False)
                
                tile_image = get_gee_tile(visualized_image, tile_region_utm, TILE_PIXELS)
                tiles.append(tile_image)
                print(" ✅")
            except Exception as e:
                print(f" ❌ Error: {e}")
                placeholder = Image.new('RGB', (TILE_PIXELS, TILE_PIXELS), (32, 32, 64))
                tiles.append(placeholder)
    
    # Stitch and save
    print("🔧 Stitching integer grid test image...")
    final_image = Image.new('RGB', (final_pixels, final_pixels))
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            final_image.paste(tiles[row * GRID_DIMENSION + col], (col * TILE_PIXELS, row * TILE_PIXELS))

    output_dir = Path(SAVE_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"integer_grid_test_{lat:.6f}_{lng:.6f}.jpg"
    final_image.save(output_path, 'JPEG', quality=95)
    
    print(f"\n✅ PRECISION TEST 3 Complete: {output_path}")
    print(f"📏 Size: {final_image.size} | 💾 File: {output_path.stat().st_size / 1024:.0f} KB")
    
    return output_path

if __name__ == '__main__':
    try:
        test_lat, test_lng = 50.4162, 30.8906
        result_file = get_precision_test_3(lat=test_lat, lng=test_lng)
        print(f"\n🎯 Result: {result_file}")
    except Exception as e:
        print(f"\n💥 Error in precision test 3: {e}")
        traceback.print_exc() 