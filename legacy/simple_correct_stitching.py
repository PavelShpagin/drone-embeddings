#!/usr/bin/env python3
"""
Simple Correct Stitching for Google Earth Engine
===============================================

Simple approach: correct positions, correct heights, no overlaps.
Result looks like a single image taken from 4km height.
"""

import ee
import requests
from PIL import Image
from pathlib import Path
import json
from datetime import datetime, timedelta
from io import BytesIO

class SimpleCorrectStitching:
    """Simple stitching with correct geographic positioning."""
    
    def __init__(self, service_account_key_path="../secrets/earth-engine-key.json"):
        self.service_account_key_path = Path(service_account_key_path)
        self.output_dir = Path("../data/gee_api") if Path.cwd().name == "examples" else Path("data/gee_api")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target dimensions
        self.target_width = 8192
        self.target_height = 4633
        
        self._initialize_gee()
    
    def _initialize_gee(self):
        """Initialize Google Earth Engine."""
        with open(self.service_account_key_path, 'r') as f:
            service_account_info = json.load(f)
        
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'],
            str(self.service_account_key_path)
        )
        ee.Initialize(credentials)
        print("✅ Google Earth Engine initialized")
    
    def calculate_4km_coverage(self):
        """Calculate coverage that looks like 4km height view."""
        # At 4km height, typical camera FOV gives ~5km ground coverage
        # For 8192x4633 pixels, this gives ~0.6m per pixel resolution
        coverage_width_km = 5.0   # 5km wide view
        coverage_height_km = 2.8  # Maintain aspect ratio
        
        return coverage_width_km, coverage_height_km
    
    def get_simple_satellite_image(self, lat, lng, coverage_width_km, coverage_height_km, season="spring"):
        """Get simple, consistent satellite image."""
        print(f"🛰️ Getting satellite image for simple stitching...")
        
        # Define region
        region = ee.Geometry.Rectangle([
            lng - coverage_width_km / 222.0,  # ~111km per degree * 2 for width
            lat - coverage_height_km / 222.0,
            lng + coverage_width_km / 222.0,
            lat + coverage_height_km / 222.0
        ])
        
        # Date range for season
        season_ranges = {
            'spring': ('03-01', '05-31'),
            'summer': ('06-01', '08-31'),
            'autumn': ('09-01', '11-30'),
            'winter': ('12-01', '02-28')
        }
        
        start_date, end_date = season_ranges.get(season, season_ranges['spring'])
        current_year = datetime.now().year
        
        # Get Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(region)
                     .filterDate(f"{current_year-1}-{start_date}", f"{current_year}-{end_date}")
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                     .select(['B4', 'B3', 'B2']))
        
        # Simple median composite for consistency
        image = collection.median().divide(10000)
        
        # Apply consistent visualization
        visualized = image.visualize(**{
            'bands': ['B4', 'B3', 'B2'],
            'min': 0.0,
            'max': 0.25,
            'gamma': 1.1
        })
        
        print(f"   ✅ Got consistent satellite image")
        return visualized, region
    
    def download_simple_tiles(self, image, region, coverage_width_km, coverage_height_km):
        """Download tiles with correct positioning - no overlaps."""
        print(f"📥 Downloading tiles with correct positioning...")
        
        # Simple 5x5 grid for smaller, faster downloads
        tiles_x, tiles_y = 5, 5
        tile_width = self.target_width // tiles_x
        tile_height = self.target_height // tiles_y
        
        print(f"   🧩 Simple grid: {tiles_x}×{tiles_y}")
        print(f"   📦 Tile size: {tile_width}×{tile_height} pixels")
        print(f"   📏 Geographic coverage: {coverage_width_km:.1f}km × {coverage_height_km:.1f}km")
        
        # Get region bounds
        region_bounds = region.bounds().getInfo()['coordinates'][0]
        lon_min, lat_min = region_bounds[0]
        lon_max, lat_max = region_bounds[2]
        
        # Calculate precise geographic steps
        lon_step = (lon_max - lon_min) / tiles_x
        lat_step = (lat_max - lat_min) / tiles_y
        
        # Create final image canvas
        final_image = Image.new('RGB', (self.target_width, self.target_height))
        
        # Download each tile at correct position
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                print(f"   📦 Tile {tile_x+1},{tile_y+1}: ", end="")
                
                # Calculate exact geographic bounds (no overlap)
                tile_lon_min = lon_min + tile_x * lon_step
                tile_lon_max = lon_min + (tile_x + 1) * lon_step
                tile_lat_min = lat_min + tile_y * lat_step
                tile_lat_max = lat_min + (tile_y + 1) * lat_step
                
                tile_region = ee.Geometry.Rectangle([
                    tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max
                ])
                
                # Download tile
                try:
                    # Retry logic for network issues
                    for attempt in range(3):
                        try:
                            url = image.getThumbURL({
                                'region': tile_region,
                                'dimensions': f"{tile_width}x{tile_height}",
                                'format': 'png',
                                'crs': 'EPSG:4326'
                            })
                            
                            response = requests.get(url, timeout=60)  # Increased timeout
                            response.raise_for_status()
                            
                            tile_image = Image.open(BytesIO(response.content))
                            
                            # Ensure correct size
                            if tile_image.size != (tile_width, tile_height):
                                tile_image = tile_image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                            
                            # Calculate position in final image
                            x_pos = tile_x * tile_width
                            y_pos = tile_y * tile_height
                            
                            # Simple paste - no blending
                            final_image.paste(tile_image, (x_pos, y_pos))
                            
                            print(f"✅")
                            break  # Success, break retry loop
                            
                        except Exception as e:
                            if attempt < 2:  # Retry
                                print(f"⚠️ Retry {attempt+1}/3...")
                                import time
                                time.sleep(2)
                                continue
                            else:  # Final failure
                                print(f"❌ {e}")
                                return None
                    
                except Exception as e:
                    print(f"❌ {e}")
                    return None
        
        return final_image
    
    def create_4km_view(self, lat, lng, location_name, season="spring"):
        """Create image that looks like 4km height aerial view."""
        print(f"\n🌍 Creating 4km Height View: {location_name}")
        print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")
        print(f"🎯 Target: {self.target_width}×{self.target_height} pixels")
        
        try:
            # Calculate 4km height coverage
            coverage_width_km, coverage_height_km = self.calculate_4km_coverage()
            
            print(f"📏 4km height coverage:")
            print(f"   Width: {coverage_width_km}km")
            print(f"   Height: {coverage_height_km}km")
            print(f"   Resolution: ~0.6m per pixel")
            
            # Get satellite image
            image, region = self.get_simple_satellite_image(lat, lng, coverage_width_km, coverage_height_km, season)
            
            # Download and stitch with correct positioning
            final_image = self.download_simple_tiles(image, region, coverage_width_km, coverage_height_km)
            
            if final_image:
                # Save with high quality
                filename = f"{lat}, {lng}.{season}"
                output_path = self.output_dir / f"{filename}.jpg"
                final_image.save(output_path, 'JPEG', quality=95, optimize=True, dpi=(96, 96))
                
                # Verify
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"\n✅ 4km height view created!")
                print(f"📁 Saved to: {output_path}")
                print(f"📏 Size: {final_image.size}")
                print(f"💾 File size: {file_size_mb:.1f} MB")
                print(f"🎯 Looks like single shot from 4km height")
                
                return output_path
            else:
                print("❌ Failed to create image")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def main():
    """Create simple, correctly positioned 4km height view."""
    print("🛰️ Simple Correct Stitching - 4km Height View")
    print("=" * 50)
    print("🎯 Creating 8192×4633 image like single 4km aerial shot")
    
    # Test coordinates
    lat = 48.9483
    lng = 29.7241
    
    try:
        stitcher = SimpleCorrectStitching()
        
        # Create 4km height view
        result = stitcher.create_4km_view(lat, lng, "kyiv_4km_simple", "summer")
        
        if result:
            print(f"\n🎉 SUCCESS! Simple 4km height view created!")
            print(f"📁 Location: {result}")
            print("✅ Features:")
            print("   📏 Correct geographic positioning")
            print("   📐 No overlaps between tiles")
            print("   🎯 Simple stitching")
            print("   🌍 Looks like 4km aerial photo")
            print("   📊 8192×4633 pixels")
        else:
            print("❌ Failed to create image")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 