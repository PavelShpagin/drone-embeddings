#!/usr/bin/env python3
"""
GEE PRECISION TEST 1 - High-Precision Decimal Handling
-----------------------------------------------------
Tests explicit precision control using Python's decimal module
to eliminate floating point rounding errors in coordinate calculations.
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
from decimal import Decimal, getcontext

# Set high precision for decimal calculations
getcontext().prec = 50

# --- Configuration ---
HEIGHT_KM = 3.0
FOV_DEGREES = 60
GRID_DIMENSION = 6
TILE_PIXELS = 512
SAVE_DIRECTORY = "../data/gee_precision_test_1"

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

def get_utm_epsg_code_precise(lat_decimal: Decimal, lng_decimal: Decimal) -> str:
    """Calculate UTM zone using high-precision decimals."""
    utm_zone = int((lng_decimal + 180) // 6) + 1
    return f"EPSG:{32600 + utm_zone if lat_decimal >= 0 else 32700 + utm_zone}"

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

def get_precision_test_1(lat: float, lng: float) -> Path:
    """Test high-precision decimal coordinate handling."""
    final_pixels = GRID_DIMENSION * TILE_PIXELS
    print("\n" + "="*70)
    print(f"🔬 PRECISION TEST 1: High-Precision Decimals ({lat:.6f}, {lng:.6f})")
    print("="*70)

    # Convert to high-precision decimals
    lat_decimal = Decimal(str(lat))
    lng_decimal = Decimal(str(lng))
    height_decimal = Decimal(str(HEIGHT_KM * 1000))
    fov_decimal = Decimal(str(FOV_DEGREES))
    
    print(f"📐 Using {getcontext().prec}-digit precision arithmetic")
    print(f"   Lat: {lat_decimal}")
    print(f"   Lng: {lng_decimal}")

    # Calculate coverage with high precision
    fov_radians = fov_decimal * Decimal('3.1415926535897932384626433832795') / Decimal('180')
    total_coverage_m_decimal = 2 * height_decimal * fov_radians.tan()
    total_coverage_m = float(total_coverage_m_decimal)
    
    print(f"🗺️  High-precision coverage: {total_coverage_m / 1000:.10f} km")

    # UTM projection with precise calculations
    utm_epsg = get_utm_epsg_code_precise(lat_decimal, lng_decimal)
    utm_projection = ee.Projection(utm_epsg)
    
    # Convert to precise floats for GEE (which doesn't support Decimal)
    center_point_wgs84 = ee.Geometry.Point([float(lng_decimal), float(lat_decimal)])
    center_point_utm = center_point_wgs84.transform(utm_projection, 1)
    easting_float, northing_float = center_point_utm.coordinates().getInfo()
    
    # Convert back to high-precision decimals for grid calculation
    easting_decimal = Decimal(str(easting_float))
    northing_decimal = Decimal(str(northing_float))
    half_coverage_decimal = total_coverage_m_decimal / 2
    
    print(f"📍 UTM Coordinates (high-precision):")
    print(f"   Easting:  {easting_decimal}")
    print(f"   Northing: {northing_decimal}")

    # Calculate precise grid boundaries
    min_easting_decimal = easting_decimal - half_coverage_decimal
    max_easting_decimal = easting_decimal + half_coverage_decimal
    min_northing_decimal = northing_decimal - half_coverage_decimal
    max_northing_decimal = northing_decimal + half_coverage_decimal

    # Convert back to floats for GEE
    region_utm = ee.Geometry.Rectangle([
        float(min_easting_decimal), float(min_northing_decimal),
        float(max_easting_decimal), float(max_northing_decimal)
    ], proj=utm_projection, geodesic=False)
    
    region_wgs84_for_filtering = region_utm.transform('EPSG:4326', 1)

    # Create image
    visualized_image = create_robust_image(region_wgs84_for_filtering)

    # Download tiles with precise grid
    print(f"📥 Downloading {GRID_DIMENSION**2} tiles with precision grid...")
    easting_step_decimal = (max_easting_decimal - min_easting_decimal) / GRID_DIMENSION
    northing_step_decimal = (max_northing_decimal - min_northing_decimal) / GRID_DIMENSION
    tiles = []
    
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            tile_num = row * GRID_DIMENSION + col + 1
            print(f"   Tile {tile_num:2d}/{GRID_DIMENSION**2}...", end="", flush=True)
            try:
                # Calculate precise tile boundaries
                tile_min_e_decimal = min_easting_decimal + col * easting_step_decimal
                tile_max_e_decimal = min_easting_decimal + (col + 1) * easting_step_decimal
                tile_min_n_decimal = max_northing_decimal - (row + 1) * northing_step_decimal
                tile_max_n_decimal = max_northing_decimal - row * northing_step_decimal
                
                tile_region_utm = ee.Geometry.Rectangle([
                    float(tile_min_e_decimal), float(tile_min_n_decimal),
                    float(tile_max_e_decimal), float(tile_max_n_decimal)
                ], proj=utm_projection, geodesic=False)
                
                tile_image = get_gee_tile(visualized_image, tile_region_utm, TILE_PIXELS)
                tiles.append(tile_image)
                print(" ✅")
            except Exception as e:
                print(f" ❌ Error: {e}")
                placeholder = Image.new('RGB', (TILE_PIXELS, TILE_PIXELS), (64, 32, 32))
                tiles.append(placeholder)
    
    # Stitch and save
    print("🔧 Stitching precision test image...")
    final_image = Image.new('RGB', (final_pixels, final_pixels))
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            final_image.paste(tiles[row * GRID_DIMENSION + col], (col * TILE_PIXELS, row * TILE_PIXELS))

    output_dir = Path(SAVE_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"precision_test_1_{lat:.6f}_{lng:.6f}.jpg"
    final_image.save(output_path, 'JPEG', quality=95)
    
    print(f"\n✅ PRECISION TEST 1 Complete: {output_path}")
    print(f"📏 Size: {final_image.size} | 💾 File: {output_path.stat().st_size / 1024:.0f} KB")
    
    return output_path

if __name__ == '__main__':
    try:
        test_lat, test_lng = 50.4162, 30.8906
        result_file = get_precision_test_1(lat=test_lat, lng=test_lng)
        print(f"\n🎯 Result: {result_file}")
    except Exception as e:
        print(f"\n💥 Error in precision test 1: {e}")
        traceback.print_exc() 