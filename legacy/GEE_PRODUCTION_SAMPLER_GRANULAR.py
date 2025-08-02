#!/usr/bin/env python3
"""
Production-Ready GEE Image Sampler (GRANULAR - UTM Projection)
--------------------------------------------------------------
This script provides the most precise method for sampling satellite imagery from
Google Earth Engine (GEE) by using a projected coordinate system (UTM) to
define the sampling grid. This is the definitive version for ML workloads.

Key Features:
- **UTM-Based Grid**: Defines the sampling grid in meters within the correct
  UTM zone, eliminating distortions from using lat/lng degrees.
- **High Precision Coordinates**: Guarantees a perfectly square, metric grid
  for the highest possible geospatial accuracy.
- **True Height Perspective**: Calculates ground coverage from a specified height
  and camera FOV.
- **Seamless Tiling**: Downloads and stitches a grid of tiles with pixel-perfect
  geographic alignment.
- **Production-Focused**: The definitive, recommended approach for sampling
  imagery for ML model training and inference.
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
SAVE_DIRECTORY = "../data/gee_api_production_granular"

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
    traceback.print_exc()
    exit(1)


def get_utm_epsg_code(lat: float, lng: float) -> str:
    """Calculates the EPSG code for the UTM zone of a given lat/lng."""
    utm_zone = math.floor((lng + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    return f"EPSG:{epsg_code}"

@retry(tries=3, delay=5, backoff=2, jitter=(1, 3))
def get_gee_tile(image: ee.Image, region: ee.Geometry, tile_size: int) -> Image.Image:
    """Downloads a single tile from GEE with retry logic."""
    url = image.getThumbURL({
        'region': region.getInfo()['coordinates'],
        'dimensions': f'{tile_size}x{tile_size}',
        'format': 'png'
    })
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def get_production_image_granular(lat: float, lng: float) -> Path:
    """Generates a high-precision, stitched satellite image using a UTM grid."""
    final_pixels = GRID_DIMENSION * TILE_PIXELS
    print("\n" + "="*60)
    print(f"🛰️  Starting GRANULAR GEE Sample for ({lat:.6f}, {lng:.6f})")
    print(f"🎯 Config: {GRID_DIMENSION}x{GRID_DIMENSION} grid, {HEIGHT_KM}km height, UTM-based metric grid")
    print("="*60)

    # 1. Calculate Ground Coverage
    total_coverage_m = 2 * (HEIGHT_KM * 1000) * math.tan(math.radians(FOV_DEGREES / 2))
    print(f"🗺️  Calculated Ground Coverage: {total_coverage_m / 1000:.3f} km")

    # 2. Determine UTM Projection and Transform Center Point
    center_point_wgs84 = ee.Geometry.Point([lng, lat])
    utm_epsg = get_utm_epsg_code(lat, lng)
    utm_projection = ee.Projection(utm_epsg)
    center_point_utm = center_point_wgs84.transform(utm_projection, 1)
    center_coords_utm = center_point_utm.coordinates().getInfo()
    easting, northing = center_coords_utm[0], center_coords_utm[1]
    
    print(f"📐 Projected to UTM Zone: {utm_epsg}")
    print(f"   Center (Easting, Northing): {easting:.2f}m, {northing:.2f}m")

    # 3. Define Metric Grid in UTM Coordinates
    half_coverage_m = total_coverage_m / 2
    min_easting = easting - half_coverage_m
    max_easting = easting + half_coverage_m
    min_northing = northing - half_coverage_m
    max_northing = northing + half_coverage_m
    
    # This is the full region of interest, defined in UTM meters
    region_utm = ee.Geometry.Rectangle(
        [min_easting, min_northing, max_easting, max_northing],
        proj=utm_projection,
        geodesic=False
    )
    # For filtering, we still need a geographic representation
    region_wgs84_for_filtering = region_utm.transform('EPSG:4326', 1)

    # 4. Select Best GEE Image
    print("☁️ Selecting best available cloud-free image...")
    image_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(region_wgs84_for_filtering)
                        .filterDate('2023-01-01', '2024-12-31')
                        .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    best_image = image_collection.first()
    if not best_image.bandNames().size().getInfo():
        raise Exception("Could not find any image for the location.")

    image_date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    print(f"✅ Selected image from {image_date} (Cloud cover: {cloud_cover:.2f}%)")
    
    visualized_image = best_image.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)

    # 5. Download Tiles using Metric Grid
    print(f"📥 Downloading {GRID_DIMENSION**2} tiles using UTM grid...")
    easting_step = (max_easting - min_easting) / GRID_DIMENSION
    northing_step = (max_northing - min_northing) / GRID_DIMENSION
    
    tiles = []
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            tile_num = row * GRID_DIMENSION + col + 1
            print(f"   Tile {tile_num:2d}/{GRID_DIMENSION**2}...", end="", flush=True)
            try:
                # Define tile region using precise meter-based coordinates
                tile_min_e = min_easting + col * easting_step
                tile_max_e = min_easting + (col + 1) * easting_step
                tile_min_n = max_northing - (row + 1) * northing_step # Start from top
                tile_max_n = max_northing - row * northing_step
                
                tile_region_utm = ee.Geometry.Rectangle(
                    [tile_min_e, tile_min_n, tile_max_e, tile_max_n],
                    proj=utm_projection, geodesic=False
                )
                
                tile_image = get_gee_tile(visualized_image, tile_region_utm, TILE_PIXELS)
                tiles.append(tile_image)
                print(" ✅")
            except Exception as e:
                print(f" ❌ FAILED. Error: {e}. Placing a placeholder.")
                tiles.append(Image.new('RGB', (TILE_PIXELS, TILE_PIXELS), (64, 0, 0)))
    
    # 6. Stitch and Save
    print("🔧 Stitching final image...")
    final_image = Image.new('RGB', (final_pixels, final_pixels))
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            final_image.paste(tiles[row * GRID_DIMENSION + col], (col * TILE_PIXELS, row * TILE_PIXELS))

    output_dir = Path(SAVE_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gee_granular_{lat:.6f}_{lng:.6f}.jpg"
    final_image.save(output_path, 'JPEG', quality=95)
    
    print("\n" + "-"*60)
    print(f"✅ Success! GRANULAR image saved to: {output_path}")
    print(f"📏 Final Size: {final_image.size} | 💾 File Size: {output_path.stat().st_size / 1024:.0f} KB")
    print("-" * 60)
    
    return output_path

if __name__ == '__main__':
    try:
        test_lat, test_lng = 50.4162, 30.8906
        generated_file = get_production_image_granular(lat=test_lat, lng=test_lng)
        print(f"\nExample complete. The generated file can be found at:\n{generated_file}")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
        traceback.print_exc()
        exit(1) 