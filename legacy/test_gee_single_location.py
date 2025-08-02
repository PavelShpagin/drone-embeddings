#!/usr/bin/env python3
"""
Simple Google Earth Engine Image Download Test
==============================================

Quick test script to download a single high-resolution satellite image.
Perfect for testing your GEE setup and downloading specific locations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from gee_image_sampler import GEEImageSampler

def main():
    """Download a single satellite image for testing."""
    print("🛰️  Simple GEE Image Download Test")
    print("=" * 40)
    
    # Test coordinates (you can change these)
    lat = 50.4501  # Kyiv center
    lng = 30.5234
    location_name = "kyiv_center_test"
    
    try:
        # Initialize sampler
        sampler = GEEImageSampler()
        
        print(f"📍 Downloading satellite image for: {location_name}")
        print(f"   Coordinates: {lat:.4f}, {lng:.4f}")
        print(f"   Coverage: 1km × 1km")
        print(f"   Resolution: 10m (Sentinel-2) or 30m (Landsat)")
        
        # Download the image
        result = sampler.sample_location(lat, lng, location_name)
        
        if result:
            print(f"\n✅ SUCCESS! Image saved to: {result}")
            print("🎯 You can now use this setup to download satellite images!")
        else:
            print("❌ Download failed. Check the error messages above.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 