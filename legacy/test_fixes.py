#!/usr/bin/env python3
"""
Test script to verify all SuperPoint fixes work correctly.
"""

import torch
import numpy as np
import cv2
from simple_superpoint import SuperPoint, SuperPointNet

def test_descriptor_normalization():
    """Test that descriptor normalization doesn't create zero vectors."""
    print("🧪 Testing descriptor normalization...")
    
    # Create a simple model
    model = SuperPointNet()
    model.eval()
    
    # Create test input
    test_input = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        semi, desc = model(test_input)
    
    # Check descriptor properties
    desc_np = desc.cpu().numpy()
    
    print(f"  Descriptor shape: {desc_np.shape}")
    print(f"  Descriptor range: [{desc_np.min():.3f}, {desc_np.max():.3f}]")
    print(f"  Descriptor mean: {desc_np.mean():.3f}")
    print(f"  Descriptor std: {desc_np.std():.3f}")
    
    # Check if descriptors are normalized (L2 norm should be ~1)
    norms = np.linalg.norm(desc_np, axis=1)  # Norm along channel dimension
    print(f"  L2 norms - mean: {norms.mean():.3f}, std: {norms.std():.3f}")
    
    # Check for zero descriptors
    zero_descriptors = np.sum(np.all(desc_np == 0, axis=1))
    print(f"  Zero descriptors: {zero_descriptors}")
    
    if zero_descriptors == 0 and abs(norms.mean() - 1.0) < 0.1:
        print("  ✅ Descriptor normalization PASSED")
        return True
    else:
        print("  ❌ Descriptor normalization FAILED")
        return False

def test_homography_generation():
    """Test homography generation doesn't create degenerate matrices."""
    print("\n🧪 Testing homography generation...")
    
    from train_superpoint_uav import UAVDataset
    
    # Create a test image
    test_img = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    cv2.imwrite('test_img.png', test_img)
    
    # Create temporary dataset
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = os.path.join(temp_dir, 'test_img.png')
        cv2.imwrite(test_path, test_img)
        
        try:
            dataset = UAVDataset(temp_dir)
            
            # Test multiple homographies
            degenerate_count = 0
            valid_count = 0
            
            for i in range(10):
                sample = dataset[0]
                H = sample['homography'].numpy()
                
                # Check if homography is degenerate
                det = np.linalg.det(H[:2, :2])
                
                if abs(det) < 1e-6:
                    degenerate_count += 1
                else:
                    valid_count += 1
            
            print(f"  Valid homographies: {valid_count}/10")
            print(f"  Degenerate homographies: {degenerate_count}/10")
            
            if degenerate_count <= 2:  # Allow some degenerates
                print("  ✅ Homography generation PASSED")
                return True
            else:
                print("  ❌ Homography generation FAILED")
                return False
                
        except Exception as e:
            print(f"  ❌ Homography test failed with error: {e}")
            return False

def test_data_quality():
    """Test data generation quality checks."""
    print("\n🧪 Testing data quality checks...")
    
    from generate_uav_data import is_good_crop
    
    # Test with good crop (textured)
    good_crop = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    # Add some texture
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            good_crop[i:i+16, j:j+16] = np.random.randint(100, 200)
    
    # Test with bad crop (uniform)
    bad_crop = np.full((256, 256), 128, dtype=np.uint8)
    
    good_result = is_good_crop(good_crop)
    bad_result = is_good_crop(bad_crop)
    
    print(f"  Good crop accepted: {good_result}")
    print(f"  Bad crop rejected: {not bad_result}")
    
    if good_result and not bad_result:
        print("  ✅ Data quality checks PASSED")
        return True
    else:
        print("  ❌ Data quality checks FAILED")
        return False

def test_descriptor_loss():
    """Test descriptor loss function logic."""
    print("\n🧪 Testing descriptor loss function...")
    
    from train_superpoint_uav import descriptor_loss_fixed
    
    # Create test descriptors
    desc1 = torch.randn(10, 256)  # 10 descriptors
    desc2 = torch.randn(15, 256)  # 15 descriptors
    
    # Normalize them
    desc1 = torch.nn.functional.normalize(desc1, p=2, dim=1)
    desc2 = torch.nn.functional.normalize(desc2, p=2, dim=1)
    
    # Create test keypoints
    kpts1 = np.random.rand(10, 2) * 256
    kpts2 = np.random.rand(15, 2) * 256
    
    # Create identity homography (should create some matches)
    H = np.eye(3, dtype=np.float32)
    
    try:
        loss = descriptor_loss_fixed(desc1, desc2, kpts1, kpts2, H)
        
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Loss is finite: {torch.isfinite(loss).item()}")
        print(f"  Loss is positive: {loss.item() >= 0}")
        
        if torch.isfinite(loss) and loss.item() >= 0:
            print("  ✅ Descriptor loss PASSED")
            return True
        else:
            print("  ❌ Descriptor loss FAILED")
            return False
            
    except Exception as e:
        print(f"  ❌ Descriptor loss failed with error: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 Testing SuperPoint Fixes")
    print("=" * 50)
    
    tests = [
        test_descriptor_normalization,
        test_homography_generation,
        test_data_quality,
        test_descriptor_loss
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes working correctly!")
    else:
        print("⚠️ Some issues remain - check failed tests above")
    
    # Clean up
    import os
    if os.path.exists('test_img.png'):
        os.remove('test_img.png')

if __name__ == "__main__":
    main() 