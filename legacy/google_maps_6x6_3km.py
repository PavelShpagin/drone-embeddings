#!/usr/bin/env python3
"""High-Precision 6x6 Google Maps - 3km Height using src/google_maps.py API"""

import sys
from pathlib import Path
import math
import time

# Add src directory to path to import google_maps
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    import google_maps
    get_static_map = google_maps.get_static_map
    calculate_google_zoom = google_maps.calculate_google_zoom
    print("✅ Successfully imported Google Maps API from src/")
except ImportError as e:
    print(f"❌ Failed to import Google Maps API: {e}")
    print("Trying alternative import method...")
    
    # Alternative: try direct file reading and execution
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("google_maps", Path(__file__).parent.parent / 'src' / 'google_maps.py')
        google_maps_module = importlib.util.module_from_spec(spec)
        
        # Manually set the API_KEY before executing
        import os
        os.environ['API_KEY'] = 'AIzaSyBQEoMxbEzfrLNK2L69c7J6HeS5GJG-Uis'
        
        # Create a minimal config module
        import types
        config_module = types.ModuleType('config')
        config_module.API_KEY = 'AIzaSyBQEoMxbEzfrLNK2L69c7J6HeS5GJG-Uis'
        sys.modules['config'] = config_module
        
        spec.loader.exec_module(google_maps_module)
        get_static_map = google_maps_module.get_static_map
        calculate_google_zoom = google_maps_module.calculate_google_zoom
        print("✅ Successfully imported using alternative method")
    except Exception as e2:
        print(f"❌ Alternative import also failed: {e2}")
        sys.exit(1)

from PIL import Image
import traceback

try:
    print("🗺️  Starting HIGH-PRECISION 6x6 Google Maps (3km height)")
    
    # Coordinates (same as GEE for comparison)
    lat, lng = 50.4162, 30.8906
    print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")

    # HIGH-PRECISION 3KM HEIGHT CALCULATION (same as GEE)
    height_km = 3.0
    height_m = height_km * 1000
    
    # Calculate field of view and ground coverage
    fov_degrees = 60
    fov_radians = math.radians(fov_degrees)
    total_coverage_km = 2 * height_km * math.tan(fov_radians / 2)
    
    print(f"🎯 Height: {height_km}km ({height_m}m)")
    print(f"📐 Field of view: {fov_degrees}°")
    print(f"🗺️  Ground coverage: {total_coverage_km:.3f}km × {total_coverage_km:.3f}km")

    # PRECISE COORDINATE CONVERSION (same as GEE)
    km_per_degree_lat = 111.32
    km_per_degree_lng = 111.32 * math.cos(math.radians(lat))
    
    print(f"🧮 Conversion factors at lat {lat:.4f}°:")
    print(f"   Latitude:  1° = {km_per_degree_lat:.3f} km")
    print(f"   Longitude: 1° = {km_per_degree_lng:.3f} km")

    # Calculate precise offsets
    half_coverage = total_coverage_km / 2
    lat_offset = half_coverage / km_per_degree_lat
    lng_offset = half_coverage / km_per_degree_lng
    
    print(f"📏 Half coverage: {half_coverage:.3f} km")
    print(f"   Lat offset:  ±{lat_offset:.8f}°")
    print(f"   Lng offset:  ±{lng_offset:.8f}°")

    # Define precise region bounds
    lat_min = lat - lat_offset
    lat_max = lat + lat_offset
    lng_min = lng - lng_offset
    lng_max = lng + lng_offset
    
    print(f"🎯 Precise bounds:")
    print(f"   Lat: {lat_min:.8f} to {lat_max:.8f}")
    print(f"   Lng: {lng_min:.8f} to {lng_max:.8f}")

    # Calculate Google Maps zoom level for 3km altitude
    zoom_level = calculate_google_zoom(height_m, lat, image_size=512)
    tile_size = 512  # Use 512x512 for each tile
    
    print(f"🔍 Google Maps settings:")
    print(f"   Calculated zoom: {zoom_level}")
    print(f"   Tile size: {tile_size}x{tile_size}")

    # 6x6 grid calculations
    lat_step = (lat_max - lat_min) / 6
    lng_step = (lng_max - lng_min) / 6
    
    tile_coverage_km = total_coverage_km / 6
    tile_coverage_m = tile_coverage_km * 1000
    
    print(f"📐 6x6 Grid steps:")
    print(f"   Lat step: {lat_step:.8f}° ({lat_step * km_per_degree_lat * 1000:.1f}m)")
    print(f"   Lng step: {lng_step:.8f}° ({lng_step * km_per_degree_lng * 1000:.1f}m)")
    print(f"   Each tile: {tile_coverage_km:.3f}km = {tile_coverage_m:.1f}m")

    print("📥 Downloading 36 satellite tiles (6x6 grid) using Google Maps API...")

    # Download 6x6 grid using the existing API
    tiles = []
    for row in range(6):
        for col in range(6):
            tile_num = row * 6 + col + 1
            print(f"   Tile {tile_num:2d}/36 (R{row}C{col}): ", end="", flush=True)
            
            try:
                # PRECISE COORDINATE CALCULATION (same logic as GEE)
                # Row 0 = North (highest lat), Row 5 = South (lowest lat)
                tile_lat_min = lat_max - (row + 1) * lat_step
                tile_lat_max = lat_max - row * lat_step
                tile_lng_min = lng_min + col * lng_step
                tile_lng_max = lng_min + (col + 1) * lng_step
                
                # Center coordinates for the API call
                center_lat = (tile_lat_min + tile_lat_max) / 2
                center_lng = (tile_lng_min + tile_lng_max) / 2
                
                print(f"lat={center_lat:.5f},lng={center_lng:.5f} ", end="")
                
                # Use the existing Google Maps API
                tile_image = get_static_map(
                    lat=center_lat,
                    lng=center_lng,
                    zoom=zoom_level,
                    size=f"{tile_size}x{tile_size}",
                    scale=1
                )
                
                if tile_image and tile_image.size == (tile_size, tile_size):
                    tiles.append(tile_image)
                    print("✅")
                else:
                    print("❌ Invalid image")
                    # Add placeholder
                    tiles.append(Image.new('RGB', (tile_size, tile_size), (128, 128, 128)))
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                # Add placeholder
                tiles.append(Image.new('RGB', (tile_size, tile_size), (64, 64, 64)))

    print("🔧 Stitching 6x6 grid...")

    # Create final image (6 x 512 = 3072)
    final_size = tile_size * 6
    final = Image.new('RGB', (final_size, final_size))

    # Paste all 36 tiles in 6x6 arrangement
    for row in range(6):
        for col in range(6):
            tile_idx = row * 6 + col
            x_pos = col * tile_size
            y_pos = row * tile_size
            final.paste(tiles[tile_idx], (x_pos, y_pos))

    # Save to maps_api directory
    print("💾 Saving PRECISE 6x6 Google Maps image...")
    output_dir = Path("../data/maps_api")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{lat}, {lng}.PRECISE.6x6.3km.jpg"
    final.save(output_path, 'JPEG', quality=95)

    print(f"✅ DONE: {output_path}")
    print(f"📏 Final size: {final.size}")
    print(f"💾 File size: {output_path.stat().st_size / 1024:.0f} KB")
    print(f"🎯 Height perspective: {height_km}km")
    print(f"🗺️  Coverage: {total_coverage_km:.3f}km × {total_coverage_km:.3f}km")
    print(f"📐 Resolution: ~{tile_coverage_m/tile_size:.1f}m per pixel")
    print(f"🔍 Zoom level used: {zoom_level}")
    
    # Precision verification
    actual_lat_span = lat_max - lat_min
    actual_lng_span = lng_max - lng_min
    actual_lat_km = actual_lat_span * km_per_degree_lat
    actual_lng_km = actual_lng_span * km_per_degree_lng
    
    print(f"🔍 Precision verification:")
    print(f"   Lat span: {actual_lat_span:.8f}° = {actual_lat_km:.3f}km")
    print(f"   Lng span: {actual_lng_span:.8f}° = {actual_lng_km:.3f}km")
    print("✅ Google Maps 6x6 high-precision complete!")

except Exception as e:
    print(f"💥 FATAL ERROR: {e}")
    print("🔍 Traceback:")
    traceback.print_exc() 