#!/usr/bin/env python3
"""
Google Earth Engine Image Sampler
==================================

This script demonstrates how to sample high-resolution satellite imagery from Google Earth Engine
using the geemap library for efficient image chip downloading.

Features:
- 1km coverage area with maximum available resolution
- Multiple satellite data sources (Sentinel-2, Landsat 8/9)
- Automatic cloud filtering and best image selection
- Progress monitoring and error handling
- Saves images to data/gee_api/ directory

Requirements:
- Google Earth Engine service account
- Service account key file in secrets/earth-engine-key.json
"""

import ee
import geemap
import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
from datetime import datetime, timedelta
import json
from google.auth import exceptions
import requests
import shutil
from tqdm import tqdm
import multiprocessing
from retry import retry

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class GEEImageSampler:
    """Google Earth Engine Image Sampler with high-resolution capabilities."""
    
    def __init__(self, service_account_key_path="../secrets/earth-engine-key.json"):
        """Initialize the GEE Image Sampler."""
        self.service_account_key_path = Path(service_account_key_path)
        self.output_dir = Path("data/gee_api")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Earth Engine
        self._initialize_earth_engine()
        
        # Initialize interactive map for visualization
        self.map = geemap.Map()
        
    def _initialize_earth_engine(self):
        """Initialize Earth Engine with service account credentials."""
        try:
            if not self.service_account_key_path.exists():
                raise FileNotFoundError(
                    f"Service account key not found at {self.service_account_key_path}\n"
                    "Please create a GEE service account and place the JSON key file there."
                )
            
            # Read service account info
            with open(self.service_account_key_path, 'r') as f:
                key_info = json.load(f)
            
            # Initialize with service account
            credentials = ee.ServiceAccountCredentials(
                key_info['client_email'], 
                str(self.service_account_key_path)
            )
            ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
            
            print("✅ Google Earth Engine initialized successfully with service account credentials")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {str(e)}")
            print("\n🔧 Troubleshooting steps:")
            print("1. Create a service account in Google Cloud Console")
            print("2. Enable Earth Engine API for your project")
            print("3. Download the JSON key and place it at secrets/earth-engine-key.json")
            print("4. Ensure the service account has 'Earth Engine Resource Writer' role")
            raise
    
    def get_best_satellite_image(self, lat, lng, area_km=1.0, date_range_months=12):
        """
        Get the best available satellite image for a location.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude  
            area_km (float): Coverage area in kilometers (default: 1km)
            date_range_months (int): How many months back to search (default: 12)
            
        Returns:
            ee.Image: Best available satellite image
        """
        # Define area of interest (1km square)
        half_width_deg = (area_km * 1000) / (2 * 111319)  # Convert km to degrees
        region = ee.Geometry.Rectangle([
            lng - half_width_deg, lat - half_width_deg,
            lng + half_width_deg, lat + half_width_deg
        ])
        
        # Define date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*date_range_months)
        
        print(f"🔍 Searching for images from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Try different satellite datasets in order of preference
        datasets = [
            {
                "name": "Sentinel-2 MSI Level-2A (10m resolution)",
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ['B4', 'B3', 'B2'],  # RGB
                "scale_factor": 0.0001,
                "cloud_property": "CLOUDY_PIXEL_PERCENTAGE"
            },
            {
                "name": "Landsat 9 OLI-2 (30m resolution)", 
                "collection": "LANDSAT/LC09/C02/T1_L2",
                "bands": ['SR_B4', 'SR_B3', 'SR_B2'],  # RGB
                "scale_factor": 0.0000275,
                "cloud_property": "CLOUD_COVER"
            },
            {
                "name": "Landsat 8 OLI (30m resolution)",
                "collection": "LANDSAT/LC08/C02/T1_L2", 
                "bands": ['SR_B4', 'SR_B3', 'SR_B2'],  # RGB
                "scale_factor": 0.0000275,
                "cloud_property": "CLOUD_COVER"
            }
        ]
        
        for dataset in datasets:
            print(f"🛰️  Trying {dataset['name']}...")
            
            try:
                # Get image collection
                collection = (ee.ImageCollection(dataset['collection'])
                    .filterBounds(region)
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                    .filter(ee.Filter.lt(dataset['cloud_property'], 10))  # <10% clouds
                    .select(dataset['bands'])
                    .sort(dataset['cloud_property']))
                
                # Check if any images found
                size = collection.size().getInfo()
                print(f"   Found {size} images")
                
                if size > 0:
                    # Get the best image
                    best_image = collection.first()
                    
                    # Get metadata
                    cloud_cover = best_image.get(dataset['cloud_property']).getInfo()
                    date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    
                    print(f"   ✅ Selected image from {date} with {cloud_cover:.1f}% cloud cover")
                    
                    # Scale and enhance the image
                    scaled_image = best_image.multiply(dataset['scale_factor']).add(-0.2)
                    
                    # Apply visualization parameters for RGB
                    vis_params = {
                        'bands': dataset['bands'],
                        'min': 0.0,
                        'max': 0.3,
                        'gamma': 1.2
                    }
                    
                    # Create RGB composite
                    rgb_image = scaled_image.visualize(**vis_params)
                    
                    return rgb_image, dataset, region, date
                    
            except Exception as e:
                print(f"   ❌ Error with {dataset['name']}: {str(e)}")
                continue
        
        raise Exception("No suitable satellite images found for the specified location and time range")
    
    @retry(tries=3, delay=2, backoff=2)
    def download_image_chip(self, image, region, filename, resolution_m=10):
        """
        Download an image chip using Earth Engine's getThumbURL.
        
        Args:
            image (ee.Image): Earth Engine image
            region (ee.Geometry): Region to download
            filename (str): Output filename
            resolution_m (int): Resolution in meters (default: 10m for highest quality)
        """
        try:
            # Calculate optimal dimensions based on 1km area and resolution
            area_m = 1000  # 1km
            pixels = int(area_m / resolution_m)
            dimensions = f"{pixels}x{pixels}"
            
            print(f"📥 Downloading {dimensions} pixels at {resolution_m}m resolution...")
            
            # Get download URL
            url = image.getThumbURL({
                'region': region,
                'dimensions': dimensions,
                'format': 'png',
                'crs': 'EPSG:4326'
            })
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total size
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
            
            # Enhance and save high-quality version
            self._enhance_image(output_path)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            raise
    
    def _enhance_image(self, image_path):
        """Enhance image quality and save enhanced version."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply enhancements
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.1)  # Slight contrast boost
                
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.05)  # Subtle sharpening
                
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.1)  # Enhance colors
                
                # Save enhanced version
                enhanced_path = image_path.with_name(f"{image_path.stem}_enhanced{image_path.suffix}")
                img.save(enhanced_path, 'PNG', quality=100, optimize=True)
                
                print(f"✨ Enhanced image saved to: {enhanced_path}")
                
        except Exception as e:
            print(f"⚠️  Image enhancement failed: {str(e)}")
    
    def sample_location(self, lat, lng, location_name=None):
        """
        Sample satellite imagery for a specific location.
        
        Args:
            lat (float): Latitude
            lng (float): Longitude
            location_name (str): Optional name for the location
        """
        if location_name is None:
            location_name = f"loc_{lat:.4f}_{lng:.4f}"
        
        print(f"\n🌍 Sampling location: {location_name} ({lat:.4f}, {lng:.4f})")
        
        try:
            # Get best satellite image
            image, dataset, region, date = self.get_best_satellite_image(lat, lng)
            
            # Generate filename
            filename = f"{location_name}_{date}_{dataset['name'].split()[0].lower()}"
            filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
            
            # Download the image
            output_path = self.download_image_chip(image, region, filename)
            
            # Add to visualization map
            self.map.addLayer(image, {}, f"{location_name} - {dataset['name']}")
            self.map.centerObject(region, 15)
            
            print(f"🎯 Successfully sampled {location_name}")
            return output_path
            
        except Exception as e:
            print(f"❌ Failed to sample {location_name}: {str(e)}")
            return None
    
    def batch_sample_locations(self, locations):
        """
        Sample multiple locations in batch.
        
        Args:
            locations (list): List of dicts with 'lat', 'lng', and optional 'name' keys
        """
        print(f"\n🚀 Starting batch sampling of {len(locations)} locations...")
        
        results = []
        for i, location in enumerate(locations, 1):
            print(f"\n--- Location {i}/{len(locations)} ---")
            
            result = self.sample_location(
                lat=location['lat'],
                lng=location['lng'], 
                location_name=location.get('name')
            )
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r is not None)
        print(f"\n📊 Batch sampling completed: {successful}/{len(locations)} successful")
        
        return results
    
    def show_map(self):
        """Display the interactive map with all sampled locations."""
        return self.map


def main():
    """Main function to demonstrate GEE image sampling."""
    print("🛰️  Google Earth Engine Image Sampler")
    print("=" * 50)
    
    # Initialize sampler
    try:
        sampler = GEEImageSampler()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    # Test locations (similar to your existing data/earth_imagery/loc1-10)
    test_locations = [
        {"lat": 46.4925, "lng": 30.7487, "name": "odesa_port"},
        {"lat": 46.4825, "lng": 30.7233, "name": "odesa_city"}, 
        {"lat": 46.4603, "lng": 30.7658, "name": "odesa_beach"},
        {"lat": 50.4501, "lng": 30.5234, "name": "kyiv_center"},
        {"lat": 49.9935, "lng": 36.2304, "name": "kharkiv_center"},
    ]
    
    # Sample single location first
    print("\n🧪 Testing single location sampling...")
    result = sampler.sample_location(46.4925, 30.7487, "test_location")
    
    if result:
        print("\n✅ Single location test successful!")
        
        # Batch sample multiple locations
        print("\n🔄 Testing batch sampling...")
        results = sampler.batch_sample_locations(test_locations)
        
        print(f"\n🎉 Sampling complete! Check the 'data/gee_api/' directory for downloaded images.")
        print(f"📁 Output directory: {sampler.output_dir.absolute()}")
        
        # Display map (in Jupyter notebook this would show interactive map)
        print("\n🗺️  Map object created (use .show_map() in Jupyter for interactive view)")
        
    else:
        print("❌ Single location test failed. Check your authentication and network connection.")


if __name__ == "__main__":
    main() 