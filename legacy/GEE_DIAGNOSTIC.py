#!/usr/bin/env python3
"""
GEE DIAGNOSTIC SCRIPT - Black Image Troubleshooting
---------------------------------------------------
This script will diagnose and fix the persistent black image issue
by testing multiple approaches and providing detailed debugging output.
"""

import ee
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import json
import math
import numpy as np
import traceback

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
    print("✅ GEE Initialized")

except Exception as e:
    print(f"💥 FATAL ERROR during GEE initialization: {e}")
    exit(1)

def get_utm_epsg_code(lat: float, lng: float) -> str:
    """Calculates the EPSG code for the UTM zone of a given lat/lng."""
    utm_zone = math.floor((lng + 180) / 6) + 1
    return f"EPSG:{32600 + utm_zone if lat >= 0 else 32700 + utm_zone}"

def diagnose_and_fix(lat: float, lng: float):
    """Comprehensive diagnosis and fix for the black image problem."""
    print("\n" + "="*70)
    print(f"🔍 DIAGNOSTIC SESSION for ({lat:.6f}, {lng:.6f})")
    print("="*70)
    
    # Setup region
    center_point = ee.Geometry.Point([lng, lat])
    utm_epsg = get_utm_epsg_code(lat, lng)
    utm_projection = ee.Projection(utm_epsg)
    
    # Define a small test region (1km x 1km)
    buffer_m = 500  # 500m radius = 1km x 1km area
    center_utm = center_point.transform(utm_projection, 1)
    easting, northing = center_utm.coordinates().getInfo()
    
    test_region_utm = ee.Geometry.Rectangle([
        easting - buffer_m, northing - buffer_m,
        easting + buffer_m, northing + buffer_m
    ], proj=utm_projection, geodesic=False)
    test_region_wgs84 = test_region_utm.transform('EPSG:4326', 1)
    
    print(f"📍 Test Region: 1km x 1km around target")
    print(f"📐 UTM Zone: {utm_epsg}")
    
    # TEST 1: Check available imagery
    print("\n🔍 TEST 1: Checking available imagery...")
    
    collections_to_test = [
        ('COPERNICUS/S2_SR_HARMONIZED', 'Sentinel-2 Surface Reflectance'),
        ('COPERNICUS/S2', 'Sentinel-2 Top-of-Atmosphere'),
        ('LANDSAT/LC08/C02/T1_L2', 'Landsat 8 Surface Reflectance'),
        ('LANDSAT/LC08/C02/T1_TOA', 'Landsat 8 Top-of-Atmosphere')
    ]
    
    best_collection = None
    best_count = 0
    
    for collection_id, name in collections_to_test:
        try:
            # Check multiple date ranges
            date_ranges = [
                ('2024-05-01', '2024-09-30', '2024 Summer'),
                ('2023-05-01', '2023-09-30', '2023 Summer'),
                ('2022-05-01', '2022-09-30', '2022 Summer'),
                ('2024-01-01', '2024-12-31', '2024 Full Year')
            ]
            
            for start_date, end_date, period_name in date_ranges:
                collection = (ee.ImageCollection(collection_id)
                            .filterBounds(test_region_wgs84)
                            .filterDate(start_date, end_date))
                
                if 'S2' in collection_id:
                    collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                elif 'LANDSAT' in collection_id:
                    collection = collection.filter(ee.Filter.lt('CLOUD_COVER', 30))
                
                count = collection.size().getInfo()
                print(f"   {name} ({period_name}): {count} images")
                
                if count > best_count:
                    best_count = count
                    best_collection = (collection_id, name, start_date, end_date)
                    
        except Exception as e:
            print(f"   {name}: ERROR - {e}")
    
    if best_count == 0:
        print("❌ No suitable imagery found in any collection!")
        return
    
    collection_id, collection_name, start_date, end_date = best_collection
    print(f"\n✅ Best collection: {collection_name} with {best_count} images")
    print(f"   Date range: {start_date} to {end_date}")
    
    # TEST 2: Create and analyze composite
    print(f"\n🔍 TEST 2: Creating composite from {collection_name}...")
    
    collection = (ee.ImageCollection(collection_id)
                  .filterBounds(test_region_wgs84)
                  .filterDate(start_date, end_date))
    
    if 'S2' in collection_id:
        collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        rgb_bands = ['B4', 'B3', 'B2']  # Red, Green, Blue
    elif 'LANDSAT' in collection_id:
        collection = collection.filter(ee.Filter.lt('CLOUD_COVER', 30))
        if 'L2' in collection_id:
            rgb_bands = ['SR_B4', 'SR_B3', 'SR_B2']  # Surface Reflectance
        else:
            rgb_bands = ['B4', 'B3', 'B2']  # Top-of-Atmosphere
    
    # Create median composite
    composite = collection.median()
    
    # TEST 3: Analyze pixel statistics
    print(f"\n🔍 TEST 3: Analyzing pixel statistics...")
    
    # Get comprehensive statistics
    stats = composite.select(rgb_bands).reduceRegion(
        reducer=ee.Reducer.percentile([0, 2, 5, 25, 50, 75, 95, 98, 100]),
        geometry=test_region_wgs84,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    
    print("Pixel value statistics:")
    for band in rgb_bands:
        print(f"   {band}:")
        print(f"      Min: {stats.get(f'{band}_p0', 'N/A')}")
        print(f"      2%:  {stats.get(f'{band}_p2', 'N/A')}")
        print(f"      50%: {stats.get(f'{band}_p50', 'N/A')}")
        print(f"      98%: {stats.get(f'{band}_p98', 'N/A')}")
        print(f"      Max: {stats.get(f'{band}_p100', 'N/A')}")
    
    # TEST 4: Try multiple visualization approaches
    print(f"\n🔍 TEST 4: Testing different visualization approaches...")
    
    approaches = [
        {
            'name': 'Dynamic Percentile (2-98%)',
            'params': {
                'bands': rgb_bands,
                'min': [stats.get(f'{band}_p2', 0) for band in rgb_bands],
                'max': [stats.get(f'{band}_p98', 3000) for band in rgb_bands]
            }
        },
        {
            'name': 'Dynamic Percentile (5-95%)',
            'params': {
                'bands': rgb_bands,
                'min': [stats.get(f'{band}_p5', 0) for band in rgb_bands],
                'max': [stats.get(f'{band}_p95', 3000) for band in rgb_bands]
            }
        },
        {
            'name': 'Fixed Range (Conservative)',
            'params': {
                'bands': rgb_bands,
                'min': [0, 0, 0],
                'max': [3000, 3000, 3000]
            }
        },
        {
            'name': 'Fixed Range (Aggressive)',
            'params': {
                'bands': rgb_bands,
                'min': [100, 100, 100],
                'max': [1500, 1500, 1500]
            }
        }
    ]
    
    for i, approach in enumerate(approaches, 1):
        try:
            print(f"\n   Approach {i}: {approach['name']}")
            print(f"      Min: {approach['params']['min']}")
            print(f"      Max: {approach['params']['max']}")
            
            # Create visualization
            vis_image = composite.visualize(**approach['params'])
            
            # Download a small test tile
            url = vis_image.getThumbURL({
                'region': test_region_wgs84.getInfo()['coordinates'],
                'dimensions': '256x256',
                'format': 'png'
            })
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Analyze the downloaded image
            pil_image = Image.open(BytesIO(response.content))
            np_image = np.array(pil_image)
            
            # Calculate image statistics
            mean_brightness = np.mean(np_image)
            std_brightness = np.std(np_image)
            min_val = np.min(np_image)
            max_val = np.max(np_image)
            
            print(f"      Result: Mean={mean_brightness:.1f}, Std={std_brightness:.1f}, Range=[{min_val}-{max_val}]")
            
            # Save test image
            output_dir = Path("../data/gee_diagnostic")
            output_dir.mkdir(parents=True, exist_ok=True)
            test_path = output_dir / f"test_{i}_{approach['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
            pil_image.save(test_path)
            print(f"      Saved: {test_path}")
            
            # If this approach produced a good image, use it for the final version
            if mean_brightness > 50 and std_brightness > 10:  # Good contrast and brightness
                print(f"   ✅ FOUND WORKING APPROACH: {approach['name']}")
                
                # Generate the final high-resolution image
                print(f"\n🚀 Generating final 6x6 grid using working approach...")
                
                # Calculate full coverage
                HEIGHT_KM = 3.0
                FOV_DEGREES = 60
                total_coverage_m = 2 * (HEIGHT_KM * 1000) * math.tan(math.radians(FOV_DEGREES / 2))
                
                # Full region
                half_coverage_m = total_coverage_m / 2
                full_region_utm = ee.Geometry.Rectangle([
                    easting - half_coverage_m, northing - half_coverage_m,
                    easting + half_coverage_m, northing + half_coverage_m
                ], proj=utm_projection, geodesic=False)
                
                # Get full collection for this region
                full_collection = (ee.ImageCollection(collection_id)
                                  .filterBounds(full_region_utm.transform('EPSG:4326', 1))
                                  .filterDate(start_date, end_date))
                
                if 'S2' in collection_id:
                    full_collection = full_collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                elif 'LANDSAT' in collection_id:
                    full_collection = full_collection.filter(ee.Filter.lt('CLOUD_COVER', 30))
                
                full_composite = full_collection.median()
                final_vis_image = full_composite.visualize(**approach['params'])
                
                # Download final image
                final_url = final_vis_image.getThumbURL({
                    'region': full_region_utm.transform('EPSG:4326', 1).getInfo()['coordinates'],
                    'dimensions': '3072x3072',
                    'format': 'png'
                })
                
                print("   Downloading final image...")
                final_response = requests.get(final_url, timeout=120)
                final_response.raise_for_status()
                
                final_image = Image.open(BytesIO(final_response.content))
                final_output_dir = Path("../data/gee_api_production_final")
                final_output_dir.mkdir(parents=True, exist_ok=True)
                final_path = final_output_dir / f"gee_FIXED_{lat:.6f}_{lng:.6f}.jpg"
                final_image.save(final_path, 'JPEG', quality=95)
                
                print(f"\n🎉 SUCCESS! Fixed image saved to: {final_path}")
                print(f"📏 Size: {final_image.size}")
                print(f"💾 File size: {final_path.stat().st_size / 1024:.0f} KB")
                
                # Verify the final image
                final_np = np.array(final_image)
                final_mean = np.mean(final_np)
                final_std = np.std(final_np)
                print(f"🔍 Final image stats: Mean={final_mean:.1f}, Std={final_std:.1f}")
                
                return final_path
                
        except Exception as e:
            print(f"      ERROR: {e}")
    
    print("\n❌ None of the visualization approaches produced a satisfactory result.")
    print("   This suggests a fundamental issue with the data or location.")

if __name__ == '__main__':
    try:
        test_lat, test_lng = 50.4162, 30.8906
        result = diagnose_and_fix(test_lat, test_lng)
        if result:
            print(f"\n✅ DIAGNOSTIC COMPLETE. Fixed image: {result}")
        else:
            print("\n❌ DIAGNOSTIC FAILED. Could not generate a visible image.")
    except Exception as e:
        print(f"\n💥 DIAGNOSTIC ERROR: {e}")
        traceback.print_exc() 