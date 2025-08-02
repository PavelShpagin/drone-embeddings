#!/usr/bin/env python3
"""Coordinate Precision Analysis - Original vs Corrected"""

import math

def analyze_coordinate_precision():
    print("🔍 COORDINATE PRECISION ANALYSIS")
    print("="*50)
    
    # Test coordinates
    lat, lng = 50.4162, 30.8906
    coverage_km = 3.464  # From 3km height calculation
    
    print(f"📍 Test location: {lat:.6f}, {lng:.6f}")
    print(f"🎯 Required coverage: {coverage_km:.3f} km")
    print()
    
    # ORIGINAL METHOD (INCORRECT)
    print("❌ ORIGINAL METHOD (from old script):")
    original_offset_deg = coverage_km / 222  # This was the error!
    print(f"   Used: coverage/222 = {coverage_km}/222 = {original_offset_deg:.8f}°")
    
    # Calculate actual distances with original method
    km_per_degree_lat = 111.32
    km_per_degree_lng = 111.32 * math.cos(math.radians(lat))
    
    original_lat_km = original_offset_deg * km_per_degree_lat
    original_lng_km = original_offset_deg * km_per_degree_lng
    
    print(f"   Actual lat coverage: {original_lat_km:.3f} km")
    print(f"   Actual lng coverage: {original_lng_km:.3f} km")
    print(f"   ERROR in lat: {abs(original_lat_km - coverage_km):.3f} km ({abs(original_lat_km - coverage_km)/coverage_km*100:.1f}%)")
    print(f"   ERROR in lng: {abs(original_lng_km - coverage_km):.3f} km ({abs(original_lng_km - coverage_km)/coverage_km*100:.1f}%)")
    print()
    
    # CORRECTED METHOD
    print("✅ CORRECTED METHOD (high precision):")
    print(f"   Lat conversion: 1° = {km_per_degree_lat:.3f} km")
    print(f"   Lng conversion: 1° = {km_per_degree_lng:.3f} km (cos({lat:.1f}°))")
    
    correct_lat_offset = (coverage_km / 2) / km_per_degree_lat
    correct_lng_offset = (coverage_km / 2) / km_per_degree_lng
    
    print(f"   Lat offset: ±{correct_lat_offset:.8f}° = ±{correct_lat_offset * km_per_degree_lat:.3f} km")
    print(f"   Lng offset: ±{correct_lng_offset:.8f}° = ±{correct_lng_offset * km_per_degree_lng:.3f} km")
    
    # Calculate total coverage
    total_lat_coverage = correct_lat_offset * 2 * km_per_degree_lat
    total_lng_coverage = correct_lng_offset * 2 * km_per_degree_lng
    
    print(f"   Total lat coverage: {total_lat_coverage:.3f} km")
    print(f"   Total lng coverage: {total_lng_coverage:.3f} km")
    print(f"   ✅ Lat accuracy: {abs(total_lat_coverage - coverage_km):.6f} km")
    print(f"   ✅ Lng accuracy: {abs(total_lng_coverage - coverage_km):.6f} km")
    print()
    
    # HEIGHT ANALYSIS
    print("🎯 HEIGHT PERSPECTIVE ANALYSIS:")
    print("="*50)
    
    height_km = 3.0
    fov_degrees = 60
    
    print(f"📐 Input: {height_km}km height, {fov_degrees}° field of view")
    
    # Calculate ground coverage from height
    fov_radians = math.radians(fov_degrees)
    calculated_coverage = 2 * height_km * math.tan(fov_radians / 2)
    
    print(f"🧮 Calculation: 2 × {height_km} × tan({fov_degrees}°/2)")
    print(f"   = 2 × {height_km} × tan({fov_degrees/2}°)")
    print(f"   = 2 × {height_km} × {math.tan(fov_radians/2):.3f}")
    print(f"   = {calculated_coverage:.3f} km")
    print()
    
    print("🔍 GRID PRECISION:")
    print("="*50)
    
    grid_size = 6
    tile_coverage_km = calculated_coverage / grid_size
    tile_coverage_m = tile_coverage_km * 1000
    
    print(f"📐 {grid_size}×{grid_size} grid:")
    print(f"   Each tile covers: {tile_coverage_km:.3f} km = {tile_coverage_m:.1f} m")
    print(f"   Pixel resolution (512px tiles): {tile_coverage_m/512:.1f} m/pixel")
    print(f"   Total pixels: {grid_size * 512} × {grid_size * 512} = {(grid_size * 512)**2:,}")
    print()
    
    print("🎯 PRECISION IMPROVEMENTS:")
    print("="*50)
    print(f"1. ❌ OLD: Used rough approximation (coverage/222)")
    print(f"   ✅ NEW: Precise lat/lng conversion at specific latitude")
    print()
    print(f"2. ❌ OLD: Arbitrary coverage value")
    print(f"   ✅ NEW: Calculated from 3km height + 60° FOV")
    print()
    print(f"3. ❌ OLD: 4×4 grid (16 tiles)")
    print(f"   ✅ NEW: 6×6 grid (36 tiles) for higher resolution")
    print()
    print(f"4. ❌ OLD: ~{(coverage_km/4*1000/512):.1f}m/pixel resolution (estimated)")
    print(f"   ✅ NEW: ~{tile_coverage_m/512:.1f}m/pixel resolution")

if __name__ == "__main__":
    analyze_coordinate_precision() 