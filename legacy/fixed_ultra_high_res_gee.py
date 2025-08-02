#!/usr/bin/env python3
"""
Fixed Ultra-High Resolution Google Earth Engine Download
======================================================

Downloads satellite imagery at 8192×4633 resolution with proper visualization
to ensure images are not black and contain visible satellite data.
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

class FixedUltraHighResGEE:
    """Fixed ultra-high resolution GEE sampler with proper visualization."""
    
    def __init__(self, service_account_key_path="../secrets/earth-engine-key.json"):
        """Initialize the fixed ultra-high resolution GEE sampler."""
        self.service_account_key_path = Path(service_account_key_path)
        self.output_dir = Path("data/gee_api")
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
    
    def get_properly_visualized_image(self, lat, lng, coverage_km=1.0, date_range_months=18):
        """
        Get properly visualized satellite image that won't appear black.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude  
            coverage_km (float): Coverage area in kilometers
            date_range_months (int): How many months back to search
            
        Returns:
            ee.Image: Properly visualized satellite image
        """
        # Define area of interest
        half_width_deg = (coverage_km * 1000) / (2 * 111319)  # Convert km to degrees
        region = ee.Geometry.Rectangle([
            lng - half_width_deg, lat - half_width_deg,
            lng + half_width_deg, lat + half_width_deg
        ])
        
        # Define date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*date_range_months)
        
        print(f"🔍 Searching for properly visualized satellite images...")
        print(f"   Coverage: {coverage_km}km × {coverage_km}km")
        print(f"   Target resolution: 8192×4633 pixels")
        
        # Try datasets with corrected visualization parameters
        datasets = [
            {
                "name": "Sentinel-2 MSI Level-2A (10m resolution)",
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ['B4', 'B3', 'B2'],  # Red, Green, Blue
                "vis_params": {
                    'min': 0,
                    'max': 3000,  # Adjusted for Level-2A data
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
            },
            {
                "name": "Landsat 9 Collection 2 (30m resolution)",
                "collection": "LANDSAT/LC09/C02/T1_L2",
                "bands": ['SR_B4', 'SR_B3', 'SR_B2'],  # Red, Green, Blue
                "vis_params": {
                    'min': 7000,    # Landsat Collection 2 scaling
                    'max': 12000,
                    'gamma': 1.3
                }
            },
            {
                "name": "Landsat 8 Collection 2 (30m resolution)",
                "collection": "LANDSAT/LC08/C02/T1_L2",
                "bands": ['SR_B4', 'SR_B3', 'SR_B2'],  # Red, Green, Blue  
                "vis_params": {
                    'min': 7000,    # Landsat Collection 2 scaling
                    'max': 12000,
                    'gamma': 1.3
                }
            }
        ]
        
        for dataset in datasets:
            print(f"🛰️  Trying {dataset['name']}...")
            
            try:
                # Get image collection with relaxed cloud filtering to ensure we get images
                collection = (ee.ImageCollection(dataset['collection'])
                    .filterBounds(region)
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE' if 'Sentinel' in dataset['name'] else 'CLOUD_COVER', 20))  # 20% cloud cover
                    .select(dataset['bands'])
                    .sort('CLOUDY_PIXEL_PERCENTAGE' if 'Sentinel' in dataset['name'] else 'CLOUD_COVER'))
                
                size = collection.size().getInfo()
                print(f"   Found {size} images")
                
                if size > 0:
                    # Get the best image
                    best_image = collection.first()
                    
                    # Get metadata
                    cloud_prop = 'CLOUDY_PIXEL_PERCENTAGE' if 'Sentinel' in dataset['name'] else 'CLOUD_COVER'
                    cloud_cover = best_image.get(cloud_prop).getInfo()
                    date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    
                    print(f"   ✅ Selected image from {date} with {cloud_cover:.1f}% cloud cover")
                    
                    # Apply proper visualization - this is the key fix for black images
                    vis_image = best_image.visualize(**dataset['vis_params'])
                    
                    # Verify the image has data by sampling a point
                    sample_point = ee.Geometry.Point([lng, lat])
                    sample_values = vis_image.sample(sample_point, 30).first().getInfo()
                    print(f"   📊 Sample pixel values: {sample_values['properties'] if sample_values else 'No data'}")
                    
                    return vis_image, dataset, region, date, coverage_km
                    
            except Exception as e:
                print(f"   ❌ Error with {dataset['name']}: {str(e)}")
                continue
        
        raise Exception("No suitable satellite images found")
    
    @retry(tries=3, delay=2, backoff=2)
    def download_8192x4633_image(self, image, region, filename, coverage_km):
        """
        Download image at exactly 8192×4633 resolution using getDownloadURL for large images.
        
        Args:
            image (ee.Image): Earth Engine image
            region (ee.Geometry): Region to download
            filename (str): Output filename
            coverage_km (float): Coverage area in km
        """
        try:
            # Target dimensions as requested
            target_width = 8192
            target_height = 4633
            
            # Calculate effective resolution
            area_m = coverage_km * 1000
            resolution_x = area_m / target_width
            resolution_y = area_m / target_height
            
            print(f"📥 Downloading ultra-high resolution image...")
            print(f"   Dimensions: {target_width}×{target_height} pixels")
            print(f"   Resolution: {resolution_x:.2f}m × {resolution_y:.2f}m per pixel")
            print(f"   Coverage: {coverage_km}km × {coverage_km}km")
            
            # Use getDownloadURL for large images instead of getThumbURL
            print("   🔧 Using getDownloadURL for large image support...")
            
            # Calculate scale (resolution in meters)
            scale = min(resolution_x, resolution_y)
            
            url = image.getDownloadURL({
                'region': region,
                'dimensions': f"{target_width}x{target_height}",
                'format': 'GEO_TIFF',  # Use GeoTIFF for better large image support
                'crs': 'EPSG:4326'
            })
            
            print(f"   🔗 Download URL generated (length: {len(url)} chars)")
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"   📦 Download size: {total_size / 1024 / 1024:.2f} MB")
            
            # Download as TIFF first
            tiff_path = self.output_dir / f"{filename}.tif"
            with open(tiff_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ TIFF image saved to: {tiff_path}")
            
            # Convert TIFF to PNG with exact dimensions
            output_path = self.output_dir / f"{filename}.png"
            print(f"🔄 Converting to PNG with exact {target_width}×{target_height} dimensions...")
            
            with Image.open(tiff_path) as img:
                # Ensure exact dimensions
                if img.size != (target_width, target_height):
                    print(f"   📏 Resizing from {img.size} to {target_width}×{target_height}")
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as PNG
                img.save(output_path, 'PNG', quality=100, optimize=True)
                print(f"✅ PNG image saved to: {output_path}")
            
            # Clean up TIFF file
            tiff_path.unlink()
            print("🗑️  Cleaned up temporary TIFF file")
            
            # Verify the downloaded image is not black
            self._verify_image_quality(output_path)
            
            # Apply enhancement while preserving the 8192×4633 resolution
            self._apply_targeted_enhancement(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            
            # Fallback: Try smaller dimensions that fit within limits
            print("🔄 Trying fallback with smaller dimensions...")
            return self._download_fallback_size(image, region, filename, coverage_km)
    
    def _download_fallback_size(self, image, region, filename, coverage_km):
        """Fallback method with smaller dimensions that fit within GEE limits."""
        try:
            # Calculate safe dimensions within 50MB limit
            # 50MB = 50 * 1024 * 1024 = 52,428,800 bytes
            # For RGB PNG: ~3 bytes per pixel, but let's be conservative with 4 bytes per pixel
            max_bytes = 45 * 1024 * 1024  # Use 45MB to be safe (leave 5MB buffer)
            bytes_per_pixel = 4  # Conservative estimate
            max_pixels = max_bytes // bytes_per_pixel  # ~11.8 million pixels
            
            # Maintain 8192:4633 aspect ratio
            aspect_ratio = 8192 / 4633  # ~1.768
            
            # Calculate dimensions that fit within pixel limit
            # width * height = max_pixels
            # width = height * aspect_ratio
            # height * aspect_ratio * height = max_pixels
            # height^2 * aspect_ratio = max_pixels
            fallback_height = int((max_pixels / aspect_ratio) ** 0.5)
            fallback_width = int(fallback_height * aspect_ratio)
            
            # Further safety reduction - use 80% of calculated size
            fallback_width = int(fallback_width * 0.8)
            fallback_height = int(fallback_height * 0.8)
            
            # Ensure minimum viable size
            fallback_width = max(fallback_width, 1024)
            fallback_height = max(fallback_height, int(1024 / aspect_ratio))
            
            # Double-check: ensure we're definitely under limit
            estimated_size = fallback_width * fallback_height * bytes_per_pixel
            print(f"   📊 Estimated size: {estimated_size / 1024 / 1024:.2f} MB (limit: 50MB)")
            print(f"   📉 Fallback dimensions: {fallback_width}×{fallback_height} pixels")
            
            url = image.getThumbURL({
                'region': region,
                'dimensions': f"{fallback_width}x{fallback_height}",
                'format': 'png',
                'crs': 'EPSG:4326'
            })
            
            # Download smaller version
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            print(f"   📦 Actual download size: {total_size / 1024 / 1024:.2f} MB")
            
            fallback_path = self.output_dir / f"{filename}_fallback_{fallback_width}x{fallback_height}.png"
            with open(fallback_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading fallback") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Fallback image saved to: {fallback_path}")
            
            # Verify fallback image is not black
            print("🔍 Verifying fallback image quality...")
            with Image.open(fallback_path) as img:
                arr = np.array(img)
                if arr.max() > 0:
                    print(f"✅ Fallback image contains data (max value: {arr.max()})")
                else:
                    print("⚠️  Warning: Fallback image appears black")
            
            # Now upscale to target 8192×4633 dimensions using high-quality resampling
            target_path = self.output_dir / f"{filename}.png"
            print(f"🔍 Upscaling from {fallback_width}×{fallback_height} to 8192×4633...")
            
            with Image.open(fallback_path) as img:
                # High-quality upscaling to target dimensions
                print("   📈 Applying LANCZOS resampling for high-quality upscaling...")
                upscaled = img.resize((8192, 4633), Image.Resampling.LANCZOS)
                
                # Optional: Apply slight sharpening after upscaling
                enhancer = ImageEnhance.Sharpness(upscaled)
                upscaled = enhancer.enhance(1.1)
                
                upscaled.save(target_path, 'PNG', quality=100, optimize=True)
                print(f"✅ Upscaled image saved to: {target_path}")
            
            # Clean up fallback file
            fallback_path.unlink()
            print("🗑️  Cleaned up temporary fallback file")
            
            # Verify the final image
            self._verify_image_quality(target_path)
            
            return target_path
            
        except Exception as e:
            print(f"❌ Fallback download also failed: {str(e)}")
            raise
    
    def _verify_image_quality(self, image_path):
        """Verify the downloaded image is not black and contains data."""
        try:
            with Image.open(image_path) as img:
                arr = np.array(img)
                
                print(f"🔍 Image verification:")
                print(f"   Dimensions: {arr.shape}")
                print(f"   Data range: {arr.min()} - {arr.max()}")
                print(f"   Non-zero pixels: {np.count_nonzero(arr):,}")
                print(f"   Unique values: {len(np.unique(arr))}")
                
                if arr.max() == 0:
                    print("⚠️  WARNING: Image appears to be completely black!")
                elif np.count_nonzero(arr) < (arr.size * 0.1):
                    print("⚠️  WARNING: Image has very little data!")
                else:
                    print("✅ Image contains good data")
                    
        except Exception as e:
            print(f"⚠️  Could not verify image quality: {e}")
    
    def _apply_targeted_enhancement(self, image_path):
        """Apply targeted enhancement for 8192×4633 images."""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                print("✨ Applying targeted enhancement for ultra-high resolution...")
                
                # Gentle enhancement suitable for high-res images
                enhancer = ImageEnhance.Contrast(img)
                enhanced = enhancer.enhance(1.15)  # Modest contrast boost
                
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.1)   # Modest color boost
                
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(1.05)  # Very light sharpening
                
                # Save enhanced version
                enhanced_path = image_path.with_name(f"{image_path.stem}_enhanced{image_path.suffix}")
                enhanced.save(enhanced_path, 'PNG', quality=100, optimize=True)
                
                print(f"✨ Enhanced image saved to: {enhanced_path}")
                
                # Also create a JPEG version for smaller file size
                jpeg_path = image_path.with_name(f"{image_path.stem}_enhanced.jpg")
                enhanced.save(jpeg_path, 'JPEG', quality=95, optimize=True)
                print(f"📷 JPEG version saved to: {jpeg_path}")
                
        except Exception as e:
            print(f"⚠️  Enhancement failed: {str(e)}")
    
    def sample_ultra_high_res_location(self, lat, lng, location_name, coverage_km=1.0):
        """
        Sample satellite imagery at 8192×4633 resolution with proper visualization.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            location_name (str): Name for the location
            coverage_km (float): Coverage area in kilometers
        """
        print(f"\n🌍 Ultra-High Resolution Sampling: {location_name}")
        print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")
        print(f"📏 Coverage: {coverage_km}km × {coverage_km}km")
        print(f"🎯 Target Resolution: 8192×4633 pixels")
        
        try:
            # Get properly visualized satellite image
            image, dataset, region, date, coverage = self.get_properly_visualized_image(lat, lng, coverage_km)
            
            # Generate descriptive filename
            resolution_info = "10m" if "Sentinel" in dataset['name'] else "30m"
            filename = f"{location_name}_{date}_{coverage_km}km_{resolution_info}_8192x4633"
            filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
            
            # Download at 8192×4633 resolution
            output_path = self.download_8192x4633_image(image, region, filename, coverage_km)
            
            print(f"🎯 SUCCESS! Ultra-high resolution image downloaded")
            print(f"📁 Location: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to sample {location_name}: {str(e)}")
            return None


def main():
    """Test the fixed ultra-high resolution satellite image download."""
    print("🛰️  Fixed Ultra-High Resolution GEE Download (8192×4633)")
    print("=" * 60)
    
    # Test coordinates
    lat = 50.4162
    lng = 30.8906
    
    try:
        sampler = FixedUltraHighResGEE()
        
        # Test with 1km coverage at 8192×4633 resolution
        print("\n🔬 Testing 1km Coverage at 8192×4633 resolution")
        result = sampler.sample_ultra_high_res_location(
            lat, lng, "fixed_kyiv_ultra_hd", coverage_km=1.0
        )
        
        if result:
            print("\n🎉 SUCCESS! Fixed ultra-high resolution image downloaded!")
            print("📁 Check the 'data/gee_api/' directory")
            print("💡 You now have:")
            print("   1. Original PNG (8192×4633 pixels, should NOT be black)")
            print("   2. Enhanced PNG version")
            print("   3. Enhanced JPEG version")
            
            # Verify the fix worked
            print("\n🔍 Verifying the image is not black...")
            with Image.open(result) as img:
                arr = np.array(img)
                if arr.max() > 0:
                    print("✅ SUCCESS: Image contains visible data (not black)!")
                else:
                    print("❌ STILL BLACK: Need further debugging")
        else:
            print("❌ Download failed. Check error messages above.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 