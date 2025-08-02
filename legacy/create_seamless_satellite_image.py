#!/usr/bin/env python3
"""
Create Seamless Satellite Image
==============================

Simple script to create seamless 8192×4633 satellite images 
with no stitching lines, matching reference quality.
"""

from tiled_high_res_gee import TiledHighResGEE

def create_seamless_image(lat, lng, location_name, season="spring"):
    """
    Create a seamless satellite image with no stitching lines.
    
    Args:
        lat (float): Latitude
        lng (float): Longitude
        location_name (str): Name for the location
        season (str): Season name (spring, summer, autumn, winter)
    """
    print("🛰️  Creating Seamless Satellite Image")
    print("=" * 50)
    
    try:
        # Initialize sampler
        sampler = TiledHighResGEE()
        
        # Create seamless image
        result = sampler.sample_high_res_location(lat, lng, location_name, season)
        
        if result:
            print("\n🎉 SUCCESS! Seamless satellite image created!")
            print(f"📁 Saved to: {result}")
            print("✨ Features:")
            print("   🎯 8192×4633 pixels (perfect reference quality)")
            print("   🔗 Seamless blending (no stitching lines)")
            print("   📏 ~0.6m per pixel resolution")
            print("   🌍 ~5km bird's eye view coverage")
            return result
        else:
            print("❌ Failed to create image")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """
    Customize these settings to create your own seamless satellite images!
    """
    
    # ========================================
    # 🛠️ CUSTOMIZE THESE SETTINGS
    # ========================================
    
    LATITUDE = 50.4162      # Your target latitude
    LONGITUDE = 30.8906     # Your target longitude
    LOCATION_NAME = "my_location"  # Name for the image
    SEASON = "autumn"       # spring, summer, autumn, winter
    
    # ========================================
    # 🚀 CREATE IMAGE
    # ========================================
    
    print(f"📍 Target: {LOCATION_NAME}")
    print(f"🗺️  Coordinates: {LATITUDE:.6f}, {LONGITUDE:.6f}")
    print(f"🌿 Season: {SEASON}")
    
    result = create_seamless_image(LATITUDE, LONGITUDE, LOCATION_NAME, SEASON)
    
    if result:
        print(f"\n🎯 Your seamless satellite image is ready!")
        print(f"📂 Check: data/gee_api/{LATITUDE}, {LONGITUDE}.{SEASON}.jpg")
    else:
        print("\n❌ Image creation failed.")

if __name__ == "__main__":
    main()

# ========================================
# 📖 USAGE EXAMPLES
# ========================================
"""
To create custom seamless satellite images:

1. Edit the coordinates above:
   LATITUDE = 40.7128   # New York
   LONGITUDE = -74.0060

2. Change the location name:
   LOCATION_NAME = "new_york_downtown"

3. Pick a season:
   SEASON = "spring"    # spring, summer, autumn, winter

4. Run the script:
   python create_seamless_satellite_image.py

The script will automatically:
✅ Sample satellite imagery at 8192×4633 resolution
✅ Use 10% overlap between tiles for seamless blending
✅ Apply edge feathering to eliminate stitching lines
✅ Apply global smoothing for perfect seam removal
✅ Save as high-quality JPEG matching reference format
✅ Cover ~5km area with 0.6m per pixel resolution
""" 