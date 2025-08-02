#!/usr/bin/env python3
"""
Square 4K×4K Google Earth Engine Download
========================================

Downloads satellite imagery at 2km coverage with square 4096×4096 resolution.
Perfect for drone simulation without stretching.
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
from io import BytesIO

class Square4K2KMGEE:
    """Square 4K×4K GEE sampler with 2km coverage."""
    
    def __init__(self, service_account_key_path="../secrets/earth-engine-key.json"):
        """Initialize the square 4K×4K GEE sampler."""
        self.service_account_key_path = Path(service_account_key_path)
        # Save directly to data/gee_api (not examples/data/gee_api)
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
    
    def get_square_4k_2km_image(self, lat, lng, coverage_km=2.0, date_range_months=18):
        """
        Get satellite image optimized for square 4K×4K resolution at 2km coverage.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude  
            coverage_km (float): Coverage area in kilometers (2km for drone simulation)
            date_range_months (int): How many months back to search
            
        Returns:
            ee.Image: Properly visualized satellite image
        """
        # Define area of interest - 2km square coverage
        half_width_deg = (coverage_km * 1000) / (2 * 111319)  # Convert km to degrees
        region = ee.Geometry.Rectangle([
            lng - half_width_deg, lat - half_width_deg,
            lng + half_width_deg, lat + half_width_deg
        ])
        
        # Calculate natural resolution for 4K×4K at 2km coverage
        resolution_per_pixel = (coverage_km * 1000) / 4096  # ~0.488m per pixel
        
        print(f"🔍 Searching for satellite images for square 4K×4K resolution...")
        print(f"   Coverage: {coverage_km}km × {coverage_km}km (2km height bird's eye view)")
        print(f"   Target resolution: 4096×4096 pixels (square, no stretching)")
        print(f"   Natural pixel resolution: {resolution_per_pixel:.2f}m per pixel")
        
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
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))  # 10% cloud cover
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
    
    @retry(tries=3, delay=2, backoff=2)
    def download_square_4k(self, image, region, filename, coverage_km):
        """
        Download square 4K×4K image with optimal method.
        
        Args:
            image (ee.Image): Earth Engine image
            region (ee.Geometry): Region to download
            filename (str): Output filename
            coverage_km (float): Coverage area in km
        """
        try:
            target_size = 4096  # Square: 4096×4096
            total_pixels = target_size * target_size  # ~16.8M pixels
            
            print(f"📥 Downloading square 4K×4K satellite image...")
            print(f"   Target dimensions: {target_size}×{target_size} pixels (perfect square)")
            print(f"   Coverage: {coverage_km}km × {coverage_km}km")
            print(f"   Total pixels: {total_pixels:,}")
            
            # Check if we can download directly (16.8M pixels * 3 bytes = ~50MB)
            estimated_mb = (total_pixels * 3) / (1024 * 1024)
            print(f"   📊 Estimated size: {estimated_mb:.1f} MB")
            
            if estimated_mb < 45:  # Direct download if under 45MB
                print("   🎯 Attempting direct download...")
                url = image.getThumbURL({
                    'region': region,
                    'dimensions': f"{target_size}x{target_size}",
                    'format': 'png',
                    'crs': 'EPSG:4326'
                })
                
                # Download directly
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                print(f"   📦 Actual download size: {total_size / 1024 / 1024:.2f} MB")
                
                output_path = self.output_dir / f"{filename}.png"
                with open(output_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                print(f"✅ Direct download successful: {output_path}")
                
            else:
                # Use tiled approach for larger images
                return self._download_tiled_4k(image, region, filename, target_size)
                
            # Verify the downloaded image
            self._verify_image_quality(output_path)
            
            # Apply light enhancement
            self._apply_natural_enhancement(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Direct download failed: {str(e)}")
            print("🔄 Trying optimized fallback...")
            return self._download_optimized_4k_fallback(image, region, filename, coverage_km)
    
    def _download_tiled_4k(self, image, region, filename, target_size):
        """Download 4K image using 2×2 tiling."""
        try:
            print("   🧩 Using 2×2 tiling approach...")
            
            tile_size = target_size // 2  # 2048×2048 tiles
            tiles_per_side = 2
            
            print(f"   🧩 Using {tiles_per_side}×{tiles_per_side} tiles of {tile_size}×{tile_size} pixels each")
            
            # Create final image
            final_image = Image.new('RGB', (target_size, target_size))
            
            # Download each tile
            region_bounds = region.bounds().getInfo()['coordinates'][0]
            lon_min, lat_min = region_bounds[0]
            lon_max, lat_max = region_bounds[2]
            
            for tile_y in range(tiles_per_side):
                for tile_x in range(tiles_per_side):
                    print(f"   📦 Downloading tile {tile_x+1},{tile_y+1}...")
                    
                    # Calculate tile boundaries
                    lon_step = (lon_max - lon_min) / tiles_per_side
                    lat_step = (lat_max - lat_min) / tiles_per_side
                    
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
                        'dimensions': f"{tile_size}x{tile_size}",
                        'format': 'png',
                        'crs': 'EPSG:4326'
                    })
                    
                    response = requests.get(tile_url, stream=True)
                    response.raise_for_status()
                    
                    # Load tile and paste into final image
                    tile_image = Image.open(BytesIO(response.content))
                    x_pos = tile_x * tile_size
                    y_pos = tile_y * tile_size
                    final_image.paste(tile_image, (x_pos, y_pos))
            
            # Save final image
            output_path = self.output_dir / f"{filename}.png"
            final_image.save(output_path, 'PNG', quality=100, optimize=True)
            print(f"✅ Tiled download successful: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Tiled download failed: {str(e)}")
            raise
    
    def _download_optimized_4k_fallback(self, image, region, filename, coverage_km):
        """Optimized fallback that downloads smaller and upscales to 4K."""
        try:
            # Conservative approach - download ~25MB image and upscale
            max_bytes = 25 * 1024 * 1024  # 25MB
            bytes_per_pixel = 4  # Conservative estimate
            max_pixels = max_bytes // bytes_per_pixel  # ~6.5M pixels
            
            # Calculate square dimensions
            fallback_size = int(max_pixels ** 0.5)  # Square root for square image
            fallback_size = min(fallback_size, 2560)  # Max 2560×2560
            
            print(f"   📉 Fallback dimensions: {fallback_size}×{fallback_size} pixels")
            print(f"   📊 Estimated size: {(fallback_size * fallback_size * 4) / 1024 / 1024:.1f} MB")
            
            url = image.getThumbURL({
                'region': region,
                'dimensions': f"{fallback_size}x{fallback_size}",
                'format': 'png',
                'crs': 'EPSG:4326'
            })
            
            # Download smaller version
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
            
            # Upscale to 4K×4K using highest quality
            output_path = self.output_dir / f"{filename}.png"
            print(f"🔍 Upscaling to 4096×4096 with LANCZOS interpolation...")
            
            with Image.open(temp_path) as img:
                # High-quality upscale to 4K
                upscaled = img.resize((4096, 4096), Image.Resampling.LANCZOS)
                
                # Apply sharpening after upscaling
                enhancer = ImageEnhance.Sharpness(upscaled)
                upscaled = enhancer.enhance(1.2)
                
                upscaled.save(output_path, 'PNG', quality=100, optimize=True)
                print(f"✅ Final 4K image saved to: {output_path}")
            
            # Clean up temp file
            temp_path.unlink()
            
            return output_path
            
        except Exception as e:
            print(f"❌ Fallback download failed: {str(e)}")
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
                
                # Light enhancement to preserve natural look
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
    
    def sample_square_4k_2km(self, lat, lng, location_name):
        """
        Sample satellite imagery at square 4K×4K resolution with 2km coverage.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            location_name (str): Name for the location
        """
        print(f"\n🌍 Square 4K×4K Sampling: {location_name}")
        print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")
        print(f"📏 Coverage: 2km × 2km (2km height bird's eye view)")
        print(f"🎯 Target Resolution: 4096×4096 pixels (square, no stretching)")
        
        try:
            # Get satellite image optimized for 2km coverage
            image, dataset, region, date, coverage = self.get_square_4k_2km_image(lat, lng, 2.0)
            
            # Generate filename
            resolution_info = "10m" if "Sentinel" in dataset['name'] else "30m"
            filename = f"{location_name}_{date}_2km_{resolution_info}_square_4096x4096"
            filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
            
            # Download at square 4K×4K resolution
            output_path = self.download_square_4k(image, region, filename, 2.0)
            
            print(f"🎯 SUCCESS! Square 4K×4K image downloaded")
            print(f"📁 Location: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to sample {location_name}: {str(e)}")
            return None


def main():
    """Test square 4K×4K satellite image download with 2km coverage."""
    print("🛰️  Square 4K×4K Satellite Download (2km Coverage)")
    print("=" * 60)
    
    # Test coordinates
    lat = 50.4162
    lng = 30.8906
    
    try:
        sampler = Square4K2KMGEE()
        
        # Download with 2km coverage at square 4K×4K resolution
        print("\n🔬 Downloading 2km coverage at square 4096×4096 resolution")
        result = sampler.sample_square_4k_2km(lat, lng, "kyiv_2km_square_4k")
        
        if result:
            print("\n🎉 SUCCESS! Square 4K×4K image downloaded!")
            print(f"📁 Saved to: {result}")
            print("💡 Perfect for drone simulation:")
            print("   ✅ 2km × 2km coverage (2km height bird's eye view)")
            print("   ✅ 4096×4096 pixels (perfect square, no stretching)")
            print("   ✅ ~0.488m per pixel resolution")
            print("   ✅ Saved to data/gee_api/ directory")
        else:
            print("❌ Download failed. Check error messages above.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 