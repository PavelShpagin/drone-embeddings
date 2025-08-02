#!/usr/bin/env python3
"""
Test script for GEE Sampler API
==============================
Simple test of the clean GEE API with a single configuration.
"""

import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gee_sampler import sample_satellite_image

def main():
    """Test the GEE sampler API with a single configuration."""
    
    print("Testing GEE Sampler API")
    print("=" * 30)
    
    # Test coordinates (Kyiv area)
    test_lat, test_lng = 50.4162, 30.8906
    
    # Single test: 4x4 grid with 128x128 patches
    print("\nTesting: 4x4 grid, 128x128px patches")
    try:
        result = sample_satellite_image(
            lat=test_lat,
            lng=test_lng,
            grid_size=(4, 4),
            patch_pixels=(128, 128)
        )
        print(f"Success: {result}")
        print("\nTest completed successfully!")
        print("Check the 'data/gee_api/' folder for the generated image.")
    except Exception as e:
        print(f"Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()