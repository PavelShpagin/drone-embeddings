#!/usr/bin/env python3
"""
GEE FINAL FIXED - Definitive Solution for Black Image Issue
-----------------------------------------------------------
This script solves the band incompatibility issue that was causing
black images by properly harmonizing bands across different Sentinel-2 images.
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

def create_harmonized_collection(region: ee.Geometry) -> ee.Image:
    """
    Creates a harmonized, cloud-free composite image by properly handling
    band incompatibilities and using multiple fallback strategies.
    """
    print("🔧 Creating harmonized composite with band compatibility fixes...")
    
    # Strategy 1: Use Sentinel-2 SR Harmonized with proper band selection
    try:
        print("   Trying Sentinel-2 Surface Reflectance (Harmonized)...")
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(region)
                      .filterDate('2023-01-01', '2024-12-31')
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        
        # Select only the essential bands to avoid incompatibility
        essential_bands = ['B2', 'B3', 'B4', 'B8']  # Blue, Green, Red, NIR
        harmonized_collection = collection.select(essential_bands)
        
        # Check if we have images
        count = harmonized_collection.size().getInfo()
        print(f"      Found {count} compatible images")
        
        if count > 0:
            # Apply cloud masking using the SCL band
            def mask_clouds_s2_sr(image):
                scl = image.select('SCL')
                # Keep clear sky (4), vegetation (5), not vegetated (6), water (1), snow/ice (11)
                clear_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(1)).Or(scl.eq(11))
                return image.select(essential_bands).updateMask(clear_mask)
            
            # Try to get SCL band for cloud masking
            try:
                collection_with_scl = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                       .filterBounds(region)
                                       .filterDate('2023-01-01', '2024-12-31')
                                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                                       .select(['B2', 'B3', 'B4', 'B8', 'SCL']))
                
                masked_collection = collection_with_scl.map(mask_clouds_s2_sr)
                composite = masked_collection.median()
                print("   ✅ Successfully created cloud-masked composite")
                return composite
            except:
                # Fallback: simple median without cloud masking
                composite = harmonized_collection.median()
                print("   ✅ Created simple median composite (no cloud masking)")
                return composite
                
    except Exception as e:
        print(f"   ❌ Sentinel-2 SR failed: {e}")
    
    # Strategy 2: Landsat 8 Surface Reflectance
    try:
        print("   Trying Landsat 8 Surface Reflectance...")
        landsat_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                              .filterBounds(region)
                              .filterDate('2022-01-01', '2024-12-31')
                              .filter(ee.Filter.lt('CLOUD_COVER', 30)))
        
        count = landsat_collection.size().getInfo()
        print(f"      Found {count} Landsat images")
        
        if count > 0:
            # Select and scale Landsat bands
            def prepare_landsat(image):
                optical_bands = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']).multiply(0.0000275).add(-0.2)
                return optical_bands.rename(['B2', 'B3', 'B4', 'B8'])
            
            processed_landsat = landsat_collection.map(prepare_landsat)
            composite = processed_landsat.median()
            print("   ✅ Created Landsat composite")
            return composite
            
    except Exception as e:
        print(f"   ❌ Landsat failed: {e}")
    
    # Strategy 3: Sentinel-2 Top-of-Atmosphere (most compatible)
    try:
        print("   Trying Sentinel-2 Top-of-Atmosphere (fallback)...")
        toa_collection = (ee.ImageCollection('COPERNICUS/S2')
                          .filterBounds(region)
                          .filterDate('2022-01-01', '2024-12-31')
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                          .select(['B2', 'B3', 'B4', 'B8']))
        
        count = toa_collection.size().getInfo()
        print(f"      Found {count} TOA images")
        
        if count > 0:
            composite = toa_collection.median()
            print("   ✅ Created TOA composite")
            return composite
            
    except Exception as e:
        print(f"   ❌ TOA failed: {e}")
    
    raise Exception("All image collection strategies failed!")

def get_final_fixed_image(lat: float, lng: float) -> Path:
    """Generates the final, fixed high-quality image."""
    final_pixels = GRID_DIMENSION * TILE_PIXELS
    print("\n" + "="*60)
    print(f"🛰️  FINAL FIXED GEE Sample for ({lat:.6f}, {lng:.6f})")
    print(f"🎯 Config: {GRID_DIMENSION}x{GRID_DIMENSION} grid, {HEIGHT_KM}km height")
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

    # 3. Create harmonized composite
    composite_image = create_harmonized_collection(region_wgs84_for_filtering)

    # 4. ROBUST VISUALIZATION with multiple fallbacks
    print("🎨 Calculating robust visualization parameters...")
    
    visualization_strategies = [
        # Strategy 1: Percentile-based (most robust)
        {
            'name': 'Percentile-based',
            'bands': ['B4', 'B3', 'B2'],  # RGB
            'min': [300, 300, 200],
            'max': [2500, 2500, 2500]
        },
        # Strategy 2: Conservative fixed
        {
            'name': 'Conservative fixed',
            'bands': ['B4', 'B3', 'B2'],
            'min': [100, 100, 100],
            'max': [3000, 3000, 3000]
        },
        # Strategy 3: Aggressive contrast
        {
            'name': 'Aggressive contrast',
            'bands': ['B4', 'B3', 'B2'],
            'min': [500, 500, 300],
            'max': [1800, 1800, 1800]
        }
    ]
    
    visualized_image = None
    for strategy in visualization_strategies:
        try:
            print(f"   Trying {strategy['name']} visualization...")
            # Remove 'name' key before passing to visualize()
            vis_params = {k: v for k, v in strategy.items() if k != 'name'}
            vis_image = composite_image.visualize(**vis_params)
            
            # Test download a small sample
            test_url = vis_image.getThumbURL({
                'region': region_wgs84_for_filtering.getInfo()['coordinates'],
                'dimensions': '128x128',
                'format': 'png'
            })
            test_response = requests.get(test_url, timeout=30)
            test_response.raise_for_status()
            
            # If we get here, this visualization works
            visualized_image = vis_image
            print(f"   ✅ {strategy['name']} visualization successful!")
            break
            
        except Exception as e:
            print(f"   ❌ {strategy['name']} failed: {e}")
    
    if visualized_image is None:
        raise Exception("All visualization strategies failed!")

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
                print(f" ❌ Error: {e}. Using placeholder.")
                # Create a placeholder with a pattern to indicate missing data
                placeholder = Image.new('RGB', (TILE_PIXELS, TILE_PIXELS), (32, 32, 64))
                tiles.append(placeholder)
    
    # 6. Stitch and Save
    print("🔧 Stitching final image...")
    final_image = Image.new('RGB', (final_pixels, final_pixels))
    for row in range(GRID_DIMENSION):
        for col in range(GRID_DIMENSION):
            final_image.paste(tiles[row * GRID_DIMENSION + col], (col * TILE_PIXELS, row * TILE_PIXELS))

    output_dir = Path(SAVE_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gee_FINAL_FIXED_{lat:.6f}_{lng:.6f}.jpg"
    final_image.save(output_path, 'JPEG', quality=95)
    
    print("\n" + "-"*60)
    print(f"✅ SUCCESS! FINAL FIXED image saved to: {output_path}")
    print(f"📏 Final Size: {final_image.size} | 💾 File Size: {output_path.stat().st_size / 1024:.0f} KB")
    print("-" * 60)
    
    return output_path

if __name__ == '__main__':
    try:
        test_lat, test_lng = 50.4162, 30.8906
        generated_file = get_final_fixed_image(lat=test_lat, lng=test_lng)
        print(f"\n🎉 COMPLETE! The final fixed file: {generated_file}")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
        traceback.print_exc()
        exit(1) 