#!/usr/bin/env python3
"""
High-Quality Google Earth Engine Image Download
==============================================

Downloads high-resolution satellite imagery with configurable coverage areas.
Optimized for bird's eye view quality at 1km or 4km height coverage.
"""

import ee
import requests
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from datetime import datetime, timedelta
import json
from tqdm import tqdm
from retry import retry

class HighQualityGEESampler:
    """High-quality Google Earth Engine image sampler with enhanced resolution."""
    
    def __init__(self, service_account_key_path="../secrets/earth-engine-key.json"):
        """Initialize the high-quality GEE sampler."""
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
    
    def get_ultra_high_quality_image(self, lat, lng, coverage_km=1.0, date_range_months=18):
        """
        Get ultra-high quality satellite image with enhanced processing.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude  
            coverage_km (float): Coverage area in kilometers (1.0 or 4.0 recommended)
            date_range_months (int): How many months back to search
            
        Returns:
            ee.Image: Enhanced high-quality satellite image
        """
        # Define area of interest
        half_width_deg = (coverage_km * 1000) / (2 * 111319)  # Convert km to degrees
        region = ee.Geometry.Rectangle([
            lng - half_width_deg, lat - half_width_deg,
            lng + half_width_deg, lat + half_width_deg
        ])
        
        # Define date range - prioritize recent summer months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*date_range_months)
        
        print(f"🔍 Searching for ultra-high quality images...")
        print(f"   Coverage: {coverage_km}km × {coverage_km}km")
        print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Try different datasets in order of quality preference
        datasets = [
            {
                "name": "Sentinel-2 MSI Level-2A (10m resolution)",
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ['B4', 'B3', 'B2'],  # RGB (10m resolution)
                "scale_factor": 0.0001,
                "cloud_property": "CLOUDY_PIXEL_PERCENTAGE",
                "vis_params": {'min': 0, 'max': 3000, 'gamma': 1.3}
            },
            {
                "name": "Sentinel-2 MSI Level-1C (10m resolution)",  
                "collection": "COPERNICUS/S2",
                "bands": ['B4', 'B3', 'B2'],  # RGB (10m resolution)
                "scale_factor": 1,
                "cloud_property": "CLOUDY_PIXEL_PERCENTAGE",
                "vis_params": {'min': 0, 'max': 3000, 'gamma': 1.2}
            },
            {
                "name": "Landsat 9 OLI-2 (30m resolution)",
                "collection": "LANDSAT/LC09/C02/T1_L2", 
                "bands": ['SR_B4', 'SR_B3', 'SR_B2'],  # RGB (30m resolution)
                "scale_factor": 0.0000275,
                "cloud_property": "CLOUD_COVER",
                "vis_params": {'min': 0.0, 'max': 0.3, 'gamma': 1.4}
            }
        ]
        
        for dataset in datasets:
            print(f"🛰️  Trying {dataset['name']}...")
            
            try:
                # Get image collection with strict cloud filtering
                collection = (ee.ImageCollection(dataset['collection'])
                    .filterBounds(region)
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    .filter(ee.Filter.lt(dataset['cloud_property'], 5))  # Very strict: <5% clouds
                    .select(dataset['bands'])
                    .sort(dataset['cloud_property']))
                
                size = collection.size().getInfo()
                print(f"   Found {size} images with <5% cloud cover")
                
                if size > 0:
                    # Get the best image
                    best_image = collection.first()
                    
                    # Get metadata
                    cloud_cover = best_image.get(dataset['cloud_property']).getInfo()
                    date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    
                    print(f"   ✅ Selected image from {date} with {cloud_cover:.1f}% cloud cover")
                    
                    # Apply scaling and enhancement
                    if dataset['scale_factor'] != 1:
                        scaled_image = best_image.multiply(dataset['scale_factor'])
                    else:
                        scaled_image = best_image
                    
                    # Apply advanced visualization with gamma correction
                    enhanced_image = scaled_image.visualize(**dataset['vis_params'])
                    
                    return enhanced_image, dataset, region, date, coverage_km
                    
            except Exception as e:
                print(f"   ❌ Error with {dataset['name']}: {str(e)}")
                continue
        
        raise Exception("No suitable high-quality satellite images found")
    
    @retry(tries=3, delay=2, backoff=2)
    def download_ultra_high_quality_chip(self, image, region, filename, coverage_km, target_pixels=1024):
        """
        Download ultra-high quality image chip.
        
        Args:
            image (ee.Image): Earth Engine image
            region (ee.Geometry): Region to download
            filename (str): Output filename
            coverage_km (float): Coverage area in km
            target_pixels (int): Target image dimensions (1024 = 1024x1024 pixels)
        """
        try:
            dimensions = f"{target_pixels}x{target_pixels}"
            resolution_m = (coverage_km * 1000) / target_pixels
            
            print(f"📥 Downloading ultra-high quality image...")
            print(f"   Dimensions: {dimensions} pixels")
            print(f"   Effective resolution: {resolution_m:.1f}m per pixel")
            print(f"   Coverage: {coverage_km}km × {coverage_km}km")
            
            # Get download URL with maximum quality settings
            url = image.getThumbURL({
                'region': region,
                'dimensions': dimensions,
                'format': 'png',
                'crs': 'EPSG:4326'
            })
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            output_path = self.output_dir / f"{filename}.png"
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Image saved to: {output_path}")
            
            # Apply advanced post-processing
            self._apply_advanced_enhancement(output_path, coverage_km)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            raise
    
    def _apply_advanced_enhancement(self, image_path, coverage_km):
        """Apply advanced image enhancement techniques."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                print("✨ Applying advanced image enhancement...")
                
                # 1. Noise reduction with gentle blur
                enhanced = img.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                # 2. Enhanced contrast (stronger for smaller coverage areas)
                contrast_factor = 1.3 if coverage_km <= 1.0 else 1.2
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast_factor)
                
                # 3. Sharpness enhancement
                sharpness_factor = 1.2 if coverage_km <= 1.0 else 1.1
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness_factor)
                
                # 4. Color saturation boost
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.15)
                
                # 5. Brightness optimization
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(1.05)
                
                # Save ultra-enhanced version
                enhanced_path = image_path.with_name(f"{image_path.stem}_ultra_enhanced{image_path.suffix}")
                enhanced.save(enhanced_path, 'PNG', quality=100, optimize=True)
                
                print(f"✨ Ultra-enhanced image saved to: {enhanced_path}")
                
                # Also save a JPEG version for smaller file size
                jpeg_path = image_path.with_name(f"{image_path.stem}_ultra_enhanced.jpg")
                enhanced.save(jpeg_path, 'JPEG', quality=95, optimize=True)
                print(f"📷 JPEG version saved to: {jpeg_path}")
                
        except Exception as e:
            print(f"⚠️  Advanced enhancement failed: {str(e)}")
    
    def sample_high_quality_location(self, lat, lng, location_name, coverage_km=1.0):
        """
        Sample ultra-high quality satellite imagery for a specific location.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            location_name (str): Name for the location
            coverage_km (float): Coverage area in kilometers (1.0 or 4.0 recommended)
        """
        print(f"\n🌍 High-Quality Sampling: {location_name}")
        print(f"📍 Coordinates: {lat:.6f}, {lng:.6f}")
        print(f"📏 Coverage: {coverage_km}km × {coverage_km}km (bird's eye view)")
        
        try:
            # Get ultra-high quality satellite image
            image, dataset, region, date, coverage = self.get_ultra_high_quality_image(lat, lng, coverage_km)
            
            # Generate descriptive filename
            resolution_info = "10m" if "Sentinel" in dataset['name'] else "30m"
            filename = f"{location_name}_{date}_{coverage_km}km_{resolution_info}_ultra"
            filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
            
            # Download ultra-high quality image (1024x1024 pixels)
            output_path = self.download_ultra_high_quality_chip(image, region, filename, coverage_km, target_pixels=1024)
            
            print(f"🎯 SUCCESS! Ultra-high quality image downloaded")
            print(f"📁 Location: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to sample {location_name}: {str(e)}")
            return None


def main():
    """Test ultra-high quality satellite image download."""
    print("🛰️  Ultra-High Quality GEE Image Download")
    print("=" * 50)
    
    # Your specified coordinates (looks like Kyiv area)
    lat = 50.4162
    lng = 30.8906
    
    try:
        sampler = HighQualityGEESampler()
        
        # Test 1: 1km coverage (bird's eye view from ~1km height)
        print("\n🔬 Test 1: 1km Coverage (1km height bird's eye view)")
        result1 = sampler.sample_high_quality_location(
            lat, lng, "kyiv_1km_birdseye", coverage_km=1.0
        )
        
        if result1:
            print("✅ 1km coverage image downloaded successfully!")
            
            # Test 2: 4km coverage (bird's eye view from ~4km height)  
            print("\n🔭 Test 2: 4km Coverage (4km height bird's eye view)")
            result2 = sampler.sample_high_quality_location(
                lat, lng, "kyiv_4km_birdseye", coverage_km=4.0
            )
            
            if result2:
                print("✅ 4km coverage image downloaded successfully!")
                print("\n🎉 Both ultra-high quality images downloaded!")
                print("📁 Check the 'data/gee_api/' directory for your images")
                print("💡 You now have:")
                print("   - Original PNG (1024×1024 pixels)")
                print("   - Ultra-enhanced PNG version")
                print("   - Ultra-enhanced JPEG version")
            else:
                print("⚠️  4km image failed, but 1km succeeded")
        else:
            print("❌ Image download failed. Check your connection and coordinates.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 