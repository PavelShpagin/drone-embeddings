#!/usr/bin/env python3
"""
Production-Ready GEE Image Sampler
----------------------------------
This script provides a high-precision, production-grade method for sampling
satellite imagery from Google Earth Engine (GEE). It is designed to create
perfectly stitched, georeferenced images suitable for machine learning tasks.

Key Features:
- **High Precision Coordinates**: Uses accurate latitude-based longitude conversion.
- **True Height Perspective**: Calculates ground coverage from a specified height (e.g., 3km)
  and camera field-of-view (FOV), ensuring a consistent scale.
- **Seamless Tiling**: Downloads a grid of tiles (e.g., 6x6) and stitches them with
  pixel-perfect geographic alignment.
- **Robust Error Handling**: Includes retries and placeholders for failed tile downloads.
- **Production-Focused**: The correct and recommended approach for sampling imagery
  for ML model training and inference, ensuring data consistency with GEE.
"""

import ee
import requests
from PIL import Image
from pathlib import Path
import json
import math
import time
import traceback
from retry import retry
from io import BytesIO

# --- Configuration ---
HEIGHT_KM = 3.0           # Desired height perspective in kilometers.
FOV_DEGREES = 60          # Camera field of view in degrees.
GRID_DIMENSION = 6        # Creates a 6x6 grid of tiles.
TILE_PIXELS = 512         # The size of each downloaded tile in pixels.
SAVE_DIRECTORY = "../data/gee_api_production" # Directory to save final images.

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
    print("Please ensure your GEE credentials are set up correctly.")
    traceback.print_exc()
    exit(1)


@retry(tries=3, delay=5, backoff=2, jitter=(1, 3))
def get_gee_tile(image: ee.Image, region: ee.Geometry, tile_size: int) -> Image.Image:
    """
    Downloads a single tile from GEE with retry logic.
    Handles transient network errors and API rate limits.
    """
    url = image.getThumbURL({
        'region': region.getInfo()['coordinates'],
        'dimensions': f'{tile_size}x{tile_size}',
        'format': 'png'
    })
    
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    
    return Image.open(BytesIO(response.content))

def get_production_image(lat: float, lng: float) -> Path:
    """
    Generates a high-precision, stitched satellite image for a given location.

    This function performs all the necessary calculations to convert a height
    perspective into a precise geographic area, downloads the corresponding
    image tiles from GEE, and stitches them into a single, seamless image.

    Args:
        lat: The latitude of the center point.
        lng: The longitude of the center point.

    Returns:
        The Path object of the successfully created image.
    """
    final_pixels = GRID_DIMENSION * TILE_PIXELS
    print("\n" + "="*60)
    print(f"🛰️  Starting GEE Production Image Sample for ({lat:.6f}, {lng:.6f})")
    print(f"🎯 Configuration: {GRID_DIMENSION}x{GRID_DIMENSION} grid, {HEIGHT_KM}km height, {final_pixels}x{final_pixels}px final image")
    print("="*60)

    # 1. Calculate Ground Coverage from Height
    total_coverage_km = 2 * HEIGHT_KM * math.tan(math.radians(FOV_DEGREES / 2))
    print(f"🗺️  Calculated Ground Coverage: {total_coverage_km:.3f} km")

    # 2. Precise Coordinate Conversion
    km_per_degree_lat = 111.32
    km_per_degree_lng = km_per_degree_lat * math.cos(math.radians(lat))
    
    half_coverage_km = total_coverage_km / 2
    lat_offset = half_coverage_km / km_per_degree_lat
    lng_offset = half_coverage_km / km_per_degree_lng
    
    lat_min, lat_max = lat - lat_offset, lat + lat_offset
    lng_min, lng_max = lng - lng_offset, lng + lng_offset
    
    print(f"🌏 Defining precise geographic bounds...")
    region = ee.Geometry.Rectangle([lng_min, lat_min, lng_max, lat_max])

    # 3. Select Best GEE Image
    print("☁️ Selecting best available cloud-free image...")
    image_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(region)
                        .filterDate('2023-01-01', '2024-12-31')
                        .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    best_image = image_collection.first()
    
    # Check if a valid image was found by checking its band names.
    if not best_image.bandNames().size().getInfo():
        print("❌ No valid images found for the specified date range and location.")
        # Fallback to a wider date range
        image_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                            .filterBounds(region)
                            .sort('CLOUDY_PIXEL_PERCENTAGE'))
        best_image = image_collection.first()
        if not best_image.bandNames().size().getInfo():
             raise Exception("Could not find any image for the location, even with wide date range.")

    image_date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    print(f"✅ Selected image from {image_date} (Cloud cover: {cloud_cover:.2f}%)")
    
    # Visualize the image
    visualized_image = best_image.visualize(bands=['B4', 'B3', 'B2'], min=0, max=3000)

    # 4. Download Tiles in a Grid
    print(f"📥 Downloading {GRID_DIMENSION**2} tiles...")
    lat_step = (lat_max - lat_min) / GRID_DIMENSION
    lng_step = (lng_max - lng_min) / GRID_DIMENSION
    
    tiles = []
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            tile_num = row * GRID_DIMENSION + col + 1
            print(f"   Downloading tile {tile_num:2d}/{GRID_DIMENSION**2}...", end="", flush=True)
            try:
                # Define tile region
                tile_lat_min = lat_max - (row + 1) * lat_step
                tile_lat_max = lat_max - row * lat_step
                tile_lng_min = lng_min + col * lng_step
                tile_lng_max = lng_min + (col + 1) * lng_step
                tile_region = ee.Geometry.Rectangle([tile_lng_min, tile_lat_min, tile_lng_max, tile_lat_max])
                
                # Download tile
                tile_image = get_gee_tile(visualized_image, tile_region, TILE_PIXELS)
                tiles.append(tile_image)
                print(" ✅")
            except Exception as e:
                print(f" ❌ FAILED. Error: {e}. Placing a placeholder.")
                tiles.append(Image.new('RGB', (TILE_PIXELS, TILE_PIXELS), (64, 0, 0)))
    
    # 5. Stitch Tiles
    print("🔧 Stitching tiles into final image...")
    final_image = Image.new('RGB', (final_pixels, final_pixels))
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            final_image.paste(tiles[row * GRID_DIMENSION + col], (col * TILE_PIXELS, row * TILE_PIXELS))

    # 6. Save Final Image
    output_dir = Path(SAVE_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gee_prod_{lat:.6f}_{lng:.6f}.jpg"
    final_image.save(output_path, 'JPEG', quality=95)
    
    print("\n" + "-"*60)
    print(f"✅ Success! Production image saved to: {output_path}")
    print(f"📏 Final Size: {final_image.size} | 💾 File Size: {output_path.stat().st_size / 1024:.0f} KB")
    print("-" * 60)
    
    return output_path

if __name__ == '__main__':
    try:
        # --- Example Usage ---
        # Define the location for which you want to sample an image.
        test_lat, test_lng = 50.4162, 30.8906
        
        # Generate the production-quality image.
        generated_file = get_production_image(lat=test_lat, lng=test_lng)
        
        # You can now use 'generated_file' in your ML pipeline.
        print(f"\nExample complete. The generated file can be found at:\n{generated_file}")

    except Exception as e:
        print(f"\n💥 An unexpected error occurred during the main execution: {e}")
        traceback.print_exc()
        exit(1) 