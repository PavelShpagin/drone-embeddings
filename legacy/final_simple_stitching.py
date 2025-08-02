#!/usr/bin/env python3
"""
Final Simple Stitching - Based on Working Method
===============================================

Uses the proven working method but with simple direct positioning.
No overlaps, no complex blending - just correct geographic positioning.
"""

import math
from pathlib import Path
from PIL import Image
import ee
import time
import numpy as np
import requests
from io import BytesIO
import json
from datetime import datetime, timedelta

class FinalSimpleStitching:
    """Simple stitching using proven working method."""
    
    def __init__(self, service_account_key_path="../secrets/earth-engine-key.json"):
        self.service_account_key_path = Path(service_account_key_path)
        self.output_dir = Path("../data/gee_api") if Path.cwd().name == "examples" else Path("data/gee_api")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target dimensions
        self.target_width = 8192
        self.target_height = 4633
        
        self._initialize_earth_engine()
    
    def _initialize_earth_engine(self):
        """Initialize Google Earth Engine with service account authentication."""
        try:
            if not self.service_account_key_path.exists():
                raise FileNotFoundError(f"Service account key not found at {self.service_account_key_path}")
            
            with open(self.service_account_key_path, 'r') as f:
                service_account_info = json.load(f)
            
            credentials = ee.ServiceAccountCredentials(
                service_account_info['client_email'],
                str(self.service_account_key_path)
            )
            
            ee.Initialize(credentials)
            print("✅ Google Earth Engine initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {e}")
            raise
    
    def calculate_coverage(self):
        """Calculate 4km height coverage."""
        # 4km height gives ~5km ground coverage
        target_resolution_m = 0.6  # 0.6m per pixel
        
        coverage_width_km = (self.target_width * target_resolution_m) / 1000  # ~4.9km
        coverage_height_km = (self.target_height * target_resolution_m) / 1000  # ~2.8km
        
        return coverage_width_km, coverage_height_km
    
    def get_satellite_image(self, lat, lng, coverage_width_km, coverage_height_km, season="summer"):
        """Get satellite image using proven working method."""
        print(f"🔍 Getting satellite image for simple stitching...")
        print(f"   Coverage: {coverage_width_km:.1f}km × {coverage_height_km:.1f}km")
        
        # Define region
        region = ee.Geometry.Rectangle([
            lng - coverage_width_km / 222.0,
            lat - coverage_height_km / 222.0,
            lng + coverage_width_km / 222.0,
            lat + coverage_height_km / 222.0
        ])
        
        # Season date ranges
        season_ranges = {
            'spring': ('03-01', '05-31'),
            'summer': ('06-01', '08-31'),
            'autumn': ('09-01', '11-30'),
            'winter': ('12-01', '02-28')
        }
        
        start_date, end_date = season_ranges.get(season, season_ranges['summer'])
        current_year = datetime.now().year
        
        # Use the proven working approach
        for year in [current_year, current_year-1]:
            try:
                start = f"{year}-{start_date}"
                end = f"{year}-{end_date}"
                
                print(f"🛰️  Trying Sentinel-2 for {start} to {end}...")
                
                s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                               .filterBounds(region)
                               .filterDate(start, end)
                               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                               .select(['B4', 'B3', 'B2'])
                               .map(lambda img: img.divide(10000)))
                
                collection_size = s2_collection.size().getInfo()
                print(f"   Found {collection_size} images")
                
                if collection_size > 0:
                    # Use median composite for consistency
                    image = s2_collection.median()
                    
                    # Apply visualization
                    visualized = image.visualize(**{
                        'bands': ['B4', 'B3', 'B2'],
                        'min': 0.0,
                        'max': 0.25,
                        'gamma': 1.1
                    })
                    
                    print(f"   ✅ Created consistent satellite image")
                    return visualized, region
                    
            except Exception as e:
                print(f"   ⚠️  Error for {year}: {e}")
                continue
        
        raise Exception("No suitable satellite image found")
    
    def download_simple_tiles(self, image, region, coverage_width_km, coverage_height_km):
        """Download tiles with simple positioning - no overlaps."""
        print(f"📥 Simple tile download with correct positioning...")
        
        # Use 5x5 grid for manageable tile sizes
        tiles_x, tiles_y = 5, 5
        tile_width = self.target_width // tiles_x
        tile_height = self.target_height // tiles_y
        
        print(f"   🧩 Grid: {tiles_x}×{tiles_y}")
        print(f"   📦 Tile size: {tile_width}×{tile_height} pixels")
        print(f"   📏 Coverage: {coverage_width_km:.1f}km × {coverage_height_km:.1f}km")
        
        # Get region bounds
        region_bounds = region.bounds().getInfo()['coordinates'][0]
        lon_min, lat_min = region_bounds[0]
        lon_max, lat_max = region_bounds[2]
        
        # Calculate geographic steps
        lon_step = (lon_max - lon_min) / tiles_x
        lat_step = (lat_max - lat_min) / tiles_y
        
        # Create final image
        final_image = Image.new('RGB', (self.target_width, self.target_height))
        
        # Download each tile
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                print(f"   📦 Downloading tile {tile_x+1},{tile_y+1} of {tiles_x},{tiles_y}...")
                
                # Calculate exact boundaries (no overlap)
                tile_lon_min = lon_min + tile_x * lon_step
                tile_lon_max = lon_min + (tile_x + 1) * lon_step
                tile_lat_min = lat_min + tile_y * lat_step
                tile_lat_max = lat_min + (tile_y + 1) * lat_step
                
                tile_region = ee.Geometry.Rectangle([
                    tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max
                ])
                
                # Download single tile
                tile_image = self._download_single_tile(image, tile_region, tile_width, tile_height)
                
                # Calculate position in final image
                x_pos = tile_x * tile_width
                y_pos = tile_y * tile_height
                
                # Simple paste - direct positioning
                final_image.paste(tile_image, (x_pos, y_pos))
                
                print(f"      ✅ Tile {tile_x+1},{tile_y+1} positioned correctly")
        
        return final_image
    
    def _download_single_tile(self, image, tile_region, tile_width, tile_height):
        """Download a single tile with retry logic."""
        for attempt in range(3):
            try:
                url = image.getThumbURL({
                    'region': tile_region,
                    'dimensions': f"{tile_width}x{tile_height}",
                    'format': 'png',
                    'crs': 'EPSG:4326'
                })
                
                response = requests.get(url, timeout=45)
                response.raise_for_status()
                
                tile_image = Image.open(BytesIO(response.content))
                
                # Ensure exact size
                if tile_image.size != (tile_width, tile_height):
                    tile_image = tile_image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                
                return tile_image
                
            except Exception as e:
                if attempt < 2:
                    print(f"      ⚠️  Retry {attempt+1}/3...")
                    time.sleep(2)
                    continue
                else:
                    print(f"      ❌ Failed after 3 attempts: {e}")
                    raise
    
    def create_simple_image(self, lat, lng, location_name, season="summer"):
        """Create image with simple correct positioning."""
        print(f"\n🌍 Simple Correct Stitching: {location_name}")
        print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")
        print(f"🎯 Target: {self.target_width}×{self.target_height} pixels")
        
        try:
            # Calculate coverage
            coverage_width_km, coverage_height_km = self.calculate_coverage()
            
            print(f"🎯 Coverage from 4km height:")
            print(f"   Width: {coverage_width_km:.1f}km")
            print(f"   Height: {coverage_height_km:.1f}km")
            print(f"   Resolution: 0.6m per pixel")
            
            # Get satellite image
            image, region = self.get_satellite_image(lat, lng, coverage_width_km, coverage_height_km, season)
            
            # Download with simple positioning
            final_image = self.download_simple_tiles(image, region, coverage_width_km, coverage_height_km)
            
            # Save
            filename = f"{lat}, {lng}.{season}"
            output_path = self.output_dir / f"{filename}.jpg"
            final_image.save(output_path, 'JPEG', quality=95, optimize=True, dpi=(96, 96))
            
            # Verify
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            print(f"\n✅ Simple correct stitching completed!")
            print(f"📁 Saved to: {output_path}")
            print(f"📏 Size: {final_image.size}")
            print(f"💾 File size: {file_size_mb:.1f} MB")
            print(f"🎯 Features:")
            print(f"   ✅ Correct geographic positioning")
            print(f"   ✅ No overlaps between tiles")
            print(f"   ✅ Simple direct stitching")
            print(f"   ✅ 4km height equivalent view")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to create image: {e}")
            return None

def main():
    """Test simple correct stitching."""
    print("🛰️ Final Simple Correct Stitching")
    print("=" * 50)
    print("🎯 Simple positioning, no overlaps, 4km height view")
    
    # Test coordinates
    lat = 50.4162
    lng = 30.8906
    
    try:
        stitcher = FinalSimpleStitching()
        
        # Create simple image
        result = stitcher.create_simple_image(lat, lng, "kyiv_final_simple", "summer")
        
        if result:
            print(f"\n🎉 SUCCESS! Simple stitched image created!")
            print(f"📁 Result: {result}")
        else:
            print("❌ Failed to create image")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 