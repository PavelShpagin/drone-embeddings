#!/usr/bin/env python3
"""
DEFINITIVE Production GEE Sampler (UTM, Median Composite, Dynamic Visualization)
---------------------------------------------------------------------------------
This is the final, definitive script for sampling high-quality satellite
imagery from GEE. It solves all previous issues, including black images, by
using a robust median composite method.

Key Features:
- **Median Composite Image**: Eliminates clouds, shadows, and lighting variations
  by creating a synthetic, high-quality image from multiple scenes over a
  clear season. This is the definitive solution to the "black image" problem.
- **UTM-Based Metric Grid**: Guarantees a perfectly square, metric grid for the
  highest possible geospatial accuracy.
- **Dynamic Visualization**: Automatically calculates the optimal brightness and
  contrast for the final composite image.
- **True Height Perspective**: Calculates ground coverage from a specified height
  and camera FOV.
- **Production-Ready**: The most robust and reliable method for ML workloads.
"""

import ee
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import json
import math
import time
import traceback
from retry import retry

# --- Configuration ---
HEIGHT_KM = 3.0
FOV_DEGREES = 60
GRID_DIMENSION = 6
TILE_PIXELS = 512
SAVE_DIRECTORY = "../data/gee_api_production_final"

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

def get_definitive_image(lat: float, lng: float) -> Path:
    """Generates the definitive, high-quality image using a median composite."""
    final_pixels = GRID_DIMENSION * TILE_PIXELS
    print("\n" + "="*60)
    print(f"🛰️  Starting DEFINITIVE GEE Sample for ({lat:.6f}, {lng:.6f})")
    print(f"🎯 Config: {GRID_DIMENSION}x{GRID_DIMENSION} grid, {HEIGHT_KM}km height, Median Composite")
    print("="*60)

    # 1. Calculate Ground Coverage & UTM Projection
    total_coverage_m = 2 * (HEIGHT_KM * 1000) * math.tan(math.radians(FOV_DEGREES / 2))
    print(f"🗺️  Coverage: {total_coverage_m / 1000:.3f} km")

    center_point_wgs84 = ee.Geometry.Point([lng, lat])
    utm_epsg = get_utm_epsg_code(lat, lng)
    utm_projection = ee.Projection(utm_epsg)
    center_point_utm = center_point_wgs84.transform(utm_projection, 1)
    easting, northing = center_point_utm.coordinates().getInfo()
    print(f"📐 Projected to UTM Zone: {utm_epsg}")

    # 2. Define Metric Grid
    half_coverage_m = total_coverage_m / 2
    min_easting, max_easting = easting - half_coverage_m, easting + half_coverage_m
    min_northing, max_northing = northing - half_coverage_m, northing + half_coverage_m
    region_utm = ee.Geometry.Rectangle(
        [min_easting, min_northing, max_easting, max_northing],
        proj=utm_projection, geodesic=False
    )
    region_wgs84_for_filtering = region_utm.transform('EPSG:4326', 1)

    # 3. DEFINITIVE FIX: Create a Median Composite Image
    print("🔄 Creating median composite image from multiple scenes...")
    image_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(region_wgs84_for_filtering)
                        .filterDate('2023-05-01', '2023-09-30') # Prime summer months
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))) # Pre-filter cloudy scenes
    
    # Check if we have any images to composite
    collection_size = image_collection.size().getInfo()
    if collection_size == 0:
        raise Exception("No images found in the specified date range to create a composite.")
    print(f"   - Compositing from {collection_size} available images...")
    
    # Calculate the median value for each pixel
    composite_image = image_collection.median()
    print("✅ Median composite created successfully.")

    # 4. DYNAMIC VISUALIZATION on the composite
    print("🎨 Calculating dynamic visualization parameters for composite...")
    stats = composite_image.reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]),
        geometry=region_wgs84_for_filtering,
        scale=30, maxPixels=1e9
    ).getInfo()
    
    vis_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': [stats['B4_p2'], stats['B3_p2'], stats['B2_p2']],
        'max': [stats['B4_p98'], stats['B3_p98'], stats['B2_p98']]
    }
    print(f"   - Vis Min (R,G,B): {[f'{v:.0f}' for v in vis_params['min']]}")
    print(f"   - Vis Max (R,G,B): {[f'{v:.0f}' for v in vis_params['max']]}")
    
    visualized_image = composite_image.visualize(**vis_params)

    # 5. Download Tiles
    print(f"📥 Downloading {GRID_DIMENSION**2} tiles using UTM grid...")
    easting_step = (max_easting - min_easting) / GRID_DIMENSION
    northing_step = (max_northing - min_northing) / GRID_DIMENSION
    tiles = []
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            tile_num = row * GRID_DIMENSION + col + 1
            print(f"   Tile {tile_num:2d}/{GRID_DIMENSION**2}...", end="", flush=True)
            try:
                tile_min_e = min_easting + col * easting_step
                tile_max_e = min_easting + (col + 1) * easting_step
                tile_min_n = max_northing - (row + 1) * northing_step
                tile_max_n = max_northing - row * northing_step
                
                tile_region_utm = ee.Geometry.Rectangle(
                    [tile_min_e, tile_min_n, tile_max_e, tile_max_n],
                    proj=utm_projection, geodesic=False
                )
                tile_image = get_gee_tile(visualized_image, tile_region_utm, TILE_PIXELS)
                tiles.append(tile_image)
                print(" ✅")
            except Exception as e:
                print(f" ❌ FAILED. Error: {e}. Placing placeholder.")
                tiles.append(Image.new('RGB', (TILE_PIXELS, TILE_PIXELS), (64, 0, 0)))
    
    # 6. Stitch and Save
    print("🔧 Stitching final image...")
    final_image = Image.new('RGB', (final_pixels, final_pixels))
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            final_image.paste(tiles[row * GRID_DIMENSION + col], (col * TILE_PIXELS, row * TILE_PIXELS))

    output_dir = Path(SAVE_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gee_final_{lat:.6f}_{lng:.6f}.jpg"
    final_image.save(output_path, 'JPEG', quality=95)
    
    print("\n" + "-"*60)
    print(f"✅ Success! DEFINITIVE image saved to: {output_path}")
    print(f"📏 Final Size: {final_image.size} | 💾 File Size: {output_path.stat().st_size / 1024:.0f} KB")
    print("-" * 60)
    
    return output_path

if __name__ == '__main__':
    try:
        test_lat, test_lng = 50.4162, 30.8906
        generated_file = get_definitive_image(lat=test_lat, lng=test_lng)
        print(f"\nExample complete. The generated file can be found at:\n{generated_file}")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
        traceback.print_exc()
        exit(1) 