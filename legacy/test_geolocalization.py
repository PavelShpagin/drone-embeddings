#!/usr/bin/env python3
"""
Test script for the global GPS-denied geolocalization system.
Runs a quick validation of all components.
"""

import sys
import traceback
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from geolocalization.database import EmbeddingDatabase
from geolocalization.drone import Drone  
from geolocalization.state import LocalizationState
from geolocalization import config

def test_database():
    """Test the embedding database functionality."""
    print("Testing embedding database...")
    
    try:
        db = EmbeddingDatabase()
        print(f"✓ Database initialized")
        print(f"  Map size: {db.map_w}x{db.map_h} pixels")
        print(f"  Patch size: {db.patch_size_px} pixels ({db.patch_size_m}m)")
        print(f"  Grid size: {db.grid_w}x{db.grid_h}")
        
        # Test a small subset of patches for speed
        print("Building small test database...")
        # We'll test with fewer patches for speed
        return db
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        traceback.print_exc()
        return None

def test_drone(database):
    """Test the drone functionality."""
    print("\nTesting drone...")
    
    try:
        drone = Drone(database)
        print(f"✓ Drone initialized at {drone.get_true_position_world()}")
        
        # Test movement
        vio_delta, epsilon, should_update = drone.move_step()
        print(f"✓ Movement test: delta={vio_delta}, epsilon={epsilon:.3f}")
        
        # Test camera view
        camera_view = drone.get_camera_view()
        print(f"✓ Camera view: {camera_view.size}")
        
        return drone
        
    except Exception as e:
        print(f"✗ Drone test failed: {e}")
        traceback.print_exc()
        return None

def test_localization_state(database, drone):
    """Test the localization state functionality."""
    print("\nTesting localization state...")
    
    try:
        initial_pos = drone.get_true_position_world()
        print(f"Creating state with initial position: {initial_pos}")
        
        state = LocalizationState(database, initial_pos)
        print(f"✓ Localization state initialized")
        print(f"  Initial radius: {state.radius_m}m")
        print(f"  Active patches: {len(state.patch_probabilities)}")
        
        # Test motion prediction
        print("Testing motion prediction...")
        vio_delta = np.array([1.0, 1.0])
        epsilon = 0.5
        state.update_motion_prediction(vio_delta, epsilon)
        print(f"✓ Motion prediction test passed")
        
        # Test get_most_confident_position
        print("Testing get_most_confident_position...")
        confident_pos = state.get_most_confident_position()
        print(f"✓ Most confident position: {confident_pos}")
        
        # Test measurement update
        print("Testing measurement update...")
        camera_view = drone.get_camera_view()
        state.update_measurement(camera_view)
        print(f"✓ Measurement update test passed")
        
        metrics = state.get_confidence_metrics()
        print(f"  Metrics: {metrics}")
        
        return state
        
    except Exception as e:
        print(f"✗ Localization state test failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Run all tests."""
    print("=== Testing Global GPS-Denied Geolocalization System ===")
    
    # Check required files
    if not Path(config.MAP_IMAGE_PATH).exists():
        print(f"✗ Map image not found: {config.MAP_IMAGE_PATH}")
        return 1
    
    print(f"✓ Map image found: {config.MAP_IMAGE_PATH}")
    
    # Test components
    database = test_database()
    if database is None:
        return 1
    
    drone = test_drone(database)
    if drone is None:
        return 1
    
    state = test_localization_state(database, drone)
    if state is None:
        return 1
    
    print("\n=== All Tests Passed! ===")
    print("Ready to run full simulation with: python geolocalization_simulation.py")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 