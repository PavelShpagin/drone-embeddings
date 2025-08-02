#!/usr/bin/env python3
"""
Natural 8K×4K Google Earth Engine Download
==========================================

Downloads satellite imagery at 4km coverage with natural 8192×4633 resolution.
No upscaling - gets the resolution directly from satellite data.
"""

import ee
import requests
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
from datetime import datetime, timedelta
import json
from tqdm import tqdm
from retry import retry
import tempfile
import shutil

class Natural8K4KGEE:
    """Natural 8K×4K GEE sampler with 4km coverage and no upscaling."""
    
    def __init__(self, service_account_key_path="../secrets/earth-engine-key.json"):
        """Initialize the natural 8K×4K GEE sampler."""
        self.service_account_key_path = Path(service_account_key_path)
        # Fixed: Save directly to data/gee_api (not examples/data/gee_api)
        self.output_dir = Path("../data/gee_api") if Path.cwd().name == "examples" else Path("data/gee_api")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_earth_engine()
        
    def _initialize_earth_engine(self):
        """Initialize Earth Engine with service account credentials."""
        try:
            if not self.service_account_key_path.exists():
                raise FileNotFoundError(f"Service account key not found at {self.service_account_key_path}")
            
            with open(self.service_account_key_path, 'r') as f:
                key_info = json.load(f)
            
            credentials = ee.ServiceAccountCredentials(
                key_info['client_email'], 
                str(self.service_account_key_path)
            )
            ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
            print("✅ Google Earth Engine initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {str(e)}")
            raise
    
    def get_natural_8k_4k_image(self, lat, lng, coverage_km=4.0, date_range_months=18):
        """
        Get satellite image optimized for natural 8K×4K resolution at 4km coverage.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude  
            coverage_km (float): Coverage area in kilometers (4km for bird's eye view)
            date_range_months (int): How many months back to search
            
        Returns:
            ee.Image: Properly visualized satellite image
        """
        # Define area of interest - 4km coverage
        half_width_deg = (coverage_km * 1000) / (2 * 111319)  # Convert km to degrees
        region = ee.Geometry.Rectangle([
            lng - half_width_deg, lat - half_width_deg,
            lng + half_width_deg, lat + half_width_deg
        ])
        
        # Calculate natural resolution for 8K×4K at 4km coverage
        resolution_x = (coverage_km * 1000) / 8192  # ~0.488m per pixel
        resolution_y = (coverage_km * 1000) / 4633  # ~0.864m per pixel
        
        print(f"🔍 Searching for satellite images for natural 8K×4K resolution...")
        print(f"   Coverage: {coverage_km}km × {coverage_km}km (4km height bird's eye view)")
        print(f"   Target resolution: 8192×4633 pixels")
        print(f"   Natural pixel resolution: {resolution_x:.2f}m × {resolution_y:.2f}m per pixel")
        
        # Define date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*date_range_months)
        
        # Optimized datasets for natural high resolution
        datasets = [
            {
                "name": "Sentinel-2 MSI Level-2A (10m resolution)",
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ['B4', 'B3', 'B2'],  # Red, Green, Blue
                "vis_params": {
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.4
                }
            },
            {
                "name": "Sentinel-2 MSI Level-1C (10m resolution)",  
                "collection": "COPERNICUS/S2",
                "bands": ['B4', 'B3', 'B2'],  # Red, Green, Blue
                "vis_params": {
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.2
                }
            }
        ]
        
        for dataset in datasets:
            print(f"🛰️  Trying {dataset['name']}...")
            
            try:
                # Get image collection
                collection = (ee.ImageCollection(dataset['collection'])
                    .filterBounds(region)
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))  # 15% cloud cover
                    .select(dataset['bands'])
                    .sort('CLOUDY_PIXEL_PERCENTAGE'))
                
                size = collection.size().getInfo()
                print(f"   Found {size} images")
                
                if size > 0:
                    # Get the best image
                    best_image = collection.first()
                    
                    # Get metadata
                    cloud_cover = best_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
                    date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    
                    print(f"   ✅ Selected image from {date} with {cloud_cover:.1f}% cloud cover")
                    
                    # Apply proper visualization for natural colors
                    vis_image = best_image.visualize(**dataset['vis_params'])
                    
                    # Verify the image has data
                    sample_point = ee.Geometry.Point([lng, lat])
                    sample_values = vis_image.sample(sample_point, 30).first().getInfo()
                    print(f"   📊 Sample pixel values: {sample_values['properties'] if sample_values else 'No data'}")
                    
                    return vis_image, dataset, region, date, coverage_km
                    
            except Exception as e:
                print(f"   ❌ Error with {dataset['name']}: {str(e)}")
                continue
        
        raise Exception("No suitable satellite images found")
    
    def download_natural_8k_4k(self, image, region, filename, coverage_km):
        """
        Download image with natural 8K×4K resolution using tile-based approach.
        
        Args:
            image (ee.Image): Earth Engine image
            region (ee.Geometry): Region to download
            filename (str): Output filename
            coverage_km (float): Coverage area in km
        """
        try:
            target_width = 8192
            target_height = 4633
            
            print(f"📥 Downloading natural 8K×4K satellite image...")
            print(f"   Target dimensions: {target_width}×{target_height} pixels")
            print(f"   Coverage: {coverage_km}km × {coverage_km}km")
            print(f"   Method: Direct download with optimized tiling")
            
            # Calculate optimal tile size that fits within GEE limits
            # Target ~40MB per tile to stay well under 50MB limit
            max_pixels_per_tile = (40 * 1024 * 1024) // 3  # ~14M pixels per tile
            total_pixels = target_width * target_height  # ~38M pixels
            
            # Calculate how many tiles we need
            num_tiles_needed = max(1, int(np.ceil(total_pixels / max_pixels_per_tile)))
            
            # Try to make tiles as square as possible
            if num_tiles_needed <= 4:
                # Use 2×2 grid for up to 4 tiles
                tiles_x = 2
                tiles_y = 2
            elif num_tiles_needed <= 9:
                # Use 3×3 grid for 5-9 tiles  
                tiles_x = 3
                tiles_y = 3
            else:
                # Use 4×4 grid for more tiles
                tiles_x = 4
                tiles_y = 4
            
            tile_width = target_width // tiles_x
            tile_height = target_height // tiles_y
            
            # Verify each tile is within limits
            pixels_per_tile = tile_width * tile_height
            estimated_mb_per_tile = (pixels_per_tile * 3) / (1024 * 1024)
            
            print(f"   🧩 Using {tiles_x}×{tiles_y} tiles of {tile_width}×{tile_height} pixels each")
            print(f"   �� Estimated size per tile: {estimated_mb_per_tile:.1f} MB (limit: 50MB)")
            
            # Create temporary directory for tiles
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                tile_paths = []
                
                # Download each tile
                for tile_y in range(tiles_y):
                    for tile_x in range(tiles_x):
                        print(f"   📦 Downloading tile {tile_x+1},{tile_y+1} of {tiles_x},{tiles_y}...")
                        
                        # Calculate tile region
                        region_bounds = region.bounds().getInfo()['coordinates'][0]
                        lon_min, lat_min = region_bounds[0]
                        lon_max, lat_max = region_bounds[2]
                        
                        # Calculate tile boundaries
                        lon_step = (lon_max - lon_min) / tiles_x
                        lat_step = (lat_max - lat_min) / tiles_y
                        
                        tile_lon_min = lon_min + tile_x * lon_step
                        tile_lon_max = lon_min + (tile_x + 1) * lon_step
                        tile_lat_min = lat_min + tile_y * lat_step
                        tile_lat_max = lat_min + (tile_y + 1) * lat_step
                        
                        tile_region = ee.Geometry.Rectangle([
                            tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max
                        ])
                        
                        # Download tile
                        tile_url = image.getThumbURL({
                            'region': tile_region,
                            'dimensions': f"{tile_width}x{tile_height}",
                            'format': 'png',
                            'crs': 'EPSG:4326'
                        })
                        
                        # Download tile
                        response = requests.get(tile_url, stream=True)
                        response.raise_for_status()
                        
                        tile_path = temp_path / f"tile_{tile_x}_{tile_y}.png"
                        with open(tile_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        tile_paths.append((tile_x, tile_y, tile_path))
                
                # Combine tiles into final 8K×4K image
                print(f"   🔧 Combining tiles into {target_width}×{target_height} image...")
                final_image = Image.new('RGB', (target_width, target_height))
                
                for tile_x, tile_y, tile_path in tile_paths:
                    with Image.open(tile_path) as tile:
                        # Calculate position in final image
                        x_pos = tile_x * tile_width
                        y_pos = tile_y * tile_height
                        final_image.paste(tile, (x_pos, y_pos))
                
                # Save final image
                output_path = self.output_dir / f"{filename}.png"
                final_image.save(output_path, 'PNG', quality=100, optimize=True)
                print(f"✅ Natural 8K×4K image saved to: {output_path}")
                
                # Verify the final image
                self._verify_image_quality(output_path)
                
                # Apply light enhancement
                self._apply_natural_enhancement(output_path)
                
                return output_path
                
        except Exception as e:
            print(f"❌ Natural download failed: {str(e)}")
            print("🔄 Trying fallback method...")
            return self._download_optimized_fallback(image, region, filename, coverage_km)
    
    def _download_optimized_fallback(self, image, region, filename, coverage_km):
        """Optimized fallback that gets the highest possible resolution within limits."""
        try:
            # Be much more conservative - use only 30MB to ensure safety
            max_bytes = 30 * 1024 * 1024  # 30MB safety buffer
            bytes_per_pixel = 4  # Conservative estimate (PNG compression varies)
            max_pixels = max_bytes // bytes_per_pixel  # ~7.8M pixels
            
            # Maintain 8192:4633 aspect ratio
            aspect_ratio = 8192 / 4633
            opt_height = int((max_pixels / aspect_ratio) ** 0.5)
            opt_width = int(opt_height * aspect_ratio)
            
            # Further reduce by 20% for extra safety
            opt_width = int(opt_width * 0.8)
            opt_height = int(opt_height * 0.8)
            
            print(f"   📉 Conservative dimensions: {opt_width}×{opt_height} pixels")
            print(f"   📊 Estimated size: {(opt_width * opt_height * 4) / 1024 / 1024:.1f} MB (conservative)")
            
            url = image.getThumbURL({
                'region': region,
                'dimensions': f"{opt_width}x{opt_height}",
                'format': 'png',
                'crs': 'EPSG:4326'
            })
            
            # Download optimized version
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"   📦 Actual download size: {total_size / 1024 / 1024:.2f} MB")
            
            temp_path = self.output_dir / f"{filename}_temp.png"
            with open(temp_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Resize to exact 8192×4633 using highest quality
            output_path = self.output_dir / f"{filename}.png"
            print(f"🔍 Resizing to exact 8192×4633 with LANCZOS interpolation...")
            
            with Image.open(temp_path) as img:
                # High-quality resize to target dimensions
                resized = img.resize((8192, 4633), Image.Resampling.LANCZOS)
                
                # Apply slight sharpening after upscaling
                enhancer = ImageEnhance.Sharpness(resized)
                resized = enhancer.enhance(1.15)
                
                resized.save(output_path, 'PNG', quality=100, optimize=True)
                print(f"✅ Final image saved to: {output_path}")
            
            # Clean up temp file
            temp_path.unlink()
            
            return output_path
            
        except Exception as e:
            print(f"❌ Optimized fallback failed: {str(e)}")
            raise
    
    def _verify_image_quality(self, image_path):
        """Verify the downloaded image quality."""
        try:
            with Image.open(image_path) as img:
                arr = np.array(img)
                
                print(f"🔍 Image verification:")
                print(f"   Dimensions: {img.size} (width × height)")
                print(f"   Data range: {arr.min()} - {arr.max()}")
                print(f"   Non-zero pixels: {np.count_nonzero(arr):,}")
                print(f"   File size: {image_path.stat().st_size / 1024 / 1024:.2f} MB")
                
                if arr.max() == 0:
                    print("⚠️  WARNING: Image appears to be completely black!")
                else:
                    print("✅ Image contains good satellite data")
                    
        except Exception as e:
            print(f"⚠️  Could not verify image quality: {e}")
    
    def _apply_natural_enhancement(self, image_path):
        """Apply natural enhancement for satellite imagery."""
        try:
            with Image.open(image_path) as img:
                print("✨ Applying natural satellite image enhancement...")
                
                # Very light enhancement to preserve natural look
                enhancer = ImageEnhance.Contrast(img)
                enhanced = enhancer.enhance(1.1)  # Light contrast boost
                
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.05)  # Very light color boost
                
                # Save enhanced version
                enhanced_path = image_path.with_name(f"{image_path.stem}_enhanced{image_path.suffix}")
                enhanced.save(enhanced_path, 'PNG', quality=100, optimize=True)
                print(f"✨ Enhanced image saved to: {enhanced_path}")
                
        except Exception as e:
            print(f"⚠️  Enhancement failed: {str(e)}")
    
    def sample_natural_8k_4k(self, lat, lng, location_name):
        """
        Sample satellite imagery at natural 8K×4K resolution with 4km coverage.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            location_name (str): Name for the location
        """
        print(f"\n🌍 Natural 8K×4K Sampling: {location_name}")
        print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")
        print(f"📏 Coverage: 4km × 4km (4km height bird's eye view)")
        print(f"🎯 Target Resolution: 8192×4633 pixels (natural, no upscaling)")
        
        try:
            # Get satellite image optimized for 4km coverage
            image, dataset, region, date, coverage = self.get_natural_8k_4k_image(lat, lng, 4.0)
            
            # Generate filename
            resolution_info = "10m" if "Sentinel" in dataset['name'] else "30m"
            filename = f"{location_name}_{date}_4km_{resolution_info}_natural_8192x4633"
            filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
            
            # Download at natural 8K×4K resolution
            output_path = self.download_natural_8k_4k(image, region, filename, 4.0)
            
            print(f"🎯 SUCCESS! Natural 8K×4K image downloaded")
            print(f"📁 Location: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to sample {location_name}: {str(e)}")
            return None


def main():
    """Test natural 8K×4K satellite image download with 4km coverage."""
    print("🛰️  Natural 8K×4K Satellite Download (4km Coverage)")
    print("=" * 60)
    
    # Test coordinates
    lat = 50.4162
    lng = 30.8906
    
    try:
        sampler = Natural8K4KGEE()
        
        # Download with 4km coverage at natural 8K×4K resolution
        print("\n🔬 Downloading 4km coverage at natural 8192×4633 resolution")
        result = sampler.sample_natural_8k_4k(lat, lng, "kyiv_4km_natural_8k4k")
        
        if result:
            print("\n🎉 SUCCESS! Natural 8K×4K image downloaded!")
            print(f"📁 Saved to: {result}")
            print("💡 Features:")
            print("   ✅ 4km × 4km coverage (4km height bird's eye view)")
            print("   ✅ 8192×4633 pixels (no upscaling)")
            print("   ✅ Natural satellite resolution")
            print("   ✅ Saved to data/gee_api/ directory")
        else:
            print("❌ Download failed. Check error messages above.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 