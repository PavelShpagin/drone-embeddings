#!/usr/bin/env python3
"""
Custom High-Quality Satellite Image Download
===========================================

Simple script to download ultra-high quality satellite imagery.
Just modify the coordinates and settings below, then run!
"""

from test_high_quality_gee import HighQualityGEESampler

def main():
    """Download custom satellite imagery - modify settings below."""
    
    # ========================================
    # 🛠️ CUSTOMIZE THESE SETTINGS
    # ========================================
    
    # Coordinates (latitude, longitude)
    LATITUDE = 50.4162   # Your target latitude
    LONGITUDE = 30.8906  # Your target longitude
    
    # Location name (for filename)
    LOCATION_NAME = "my_location"
    
    # Coverage options: 1.0 (1km) or 4.0 (4km) or any other value
    COVERAGE_KM = 1.0  # Bird's eye view height in kilometers
    
    # ========================================
    # 🚀 DOWNLOAD PROCESS
    # ========================================
    
    print("🛰️  Custom Satellite Image Download")
    print("=" * 40)
    print(f"📍 Target: {LOCATION_NAME}")
    print(f"🗺️  Coordinates: {LATITUDE:.6f}, {LONGITUDE:.6f}")
    print(f"📏 Coverage: {COVERAGE_KM}km × {COVERAGE_KM}km")
    print(f"🎯 Resolution: ~{COVERAGE_KM}m per pixel (1024×1024 pixels)")
    
    try:
        # Initialize the sampler
        sampler = HighQualityGEESampler()
        
        # Download the image
        result = sampler.sample_high_quality_location(
            lat=LATITUDE,
            lng=LONGITUDE,
            location_name=LOCATION_NAME,
            coverage_km=COVERAGE_KM
        )
        
        if result:
            print("\n🎉 SUCCESS! Your custom satellite image is ready!")
            print(f"📁 Files saved to: data/gee_api/")
            print("💡 You have 3 versions:")
            print("   1. Original PNG (raw satellite data)")
            print("   2. Ultra-enhanced PNG (processed)")
            print("   3. Ultra-enhanced JPEG (compressed)")
        else:
            print("❌ Download failed. Try different coordinates or check your connection.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have properly set up Google Earth Engine authentication.")

if __name__ == "__main__":
    main()

# ========================================
# 📖 USAGE EXAMPLES
# ========================================
"""
To use this script:

1. Edit the coordinates above:
   LATITUDE = 40.7128   # New York
   LONGITUDE = -74.0060

2. Change the location name:
   LOCATION_NAME = "new_york_downtown"

3. Adjust coverage (bird's eye view height):
   COVERAGE_KM = 1.0   # 1km coverage (1m per pixel)
   COVERAGE_KM = 4.0   # 4km coverage (4m per pixel) 
   COVERAGE_KM = 0.5   # 500m coverage (0.5m per pixel)

4. Run the script:
   python download_custom_satellite_image.py

The script will automatically:
- Find the best cloud-free satellite image
- Download in ultra-high quality (1024×1024 pixels)
- Apply advanced image enhancement
- Save multiple versions (PNG + JPEG)
""" 