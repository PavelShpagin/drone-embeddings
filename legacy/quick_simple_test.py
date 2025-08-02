#!/usr/bin/env python3
"""Quick Simple Test - Debug version"""

import ee
import json
from pathlib import Path

def test_gee_init():
    """Test if GEE initializes properly."""
    print("🔍 Testing Google Earth Engine initialization...")
    
    try:
        service_account_key_path = Path("../secrets/earth-engine-key.json")
        
        if not service_account_key_path.exists():
            print(f"❌ Service account key not found at {service_account_key_path}")
            return False
        
        print("📁 Loading service account credentials...")
        with open(service_account_key_path, 'r') as f:
            service_account_info = json.load(f)
        
        print("🔑 Creating credentials...")
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'],
            str(service_account_key_path)
        )
        
        print("🌍 Initializing Earth Engine...")
        ee.Initialize(credentials)
        
        print("✅ Google Earth Engine initialized successfully!")
        
        # Test a simple operation
        print("🧪 Testing simple operation...")
        test_region = ee.Geometry.Point([30.8906, 50.4162])
        test_image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20240601T084601_20240601T084556_T36UYA')
        
        print("✅ Test operation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_gee_init() 