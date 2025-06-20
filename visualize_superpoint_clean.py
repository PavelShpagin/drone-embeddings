#!/usr/bin/env python3
"""
Clean SuperPoint visualization for UAV image matching.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from simple_superpoint import SuperPoint

def match_descriptors(desc1, desc2, threshold=0.8):
    """Match descriptors using ratio test."""
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    
    # Compute distances
    dists = np.linalg.norm(desc1[:, None, :] - desc2[None, :, :], axis=2)
    
    # Find best and second best matches for each descriptor in desc1
    matches = []
    for i in range(len(desc1)):
        sorted_indices = np.argsort(dists[i])
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1] if len(sorted_indices) > 1 else best_idx
        
        best_dist = dists[i, best_idx]
        second_best_dist = dists[i, second_best_idx]
        
        # Ratio test
        if best_dist < threshold * second_best_dist:
            matches.append((i, best_idx, best_dist))
    
    return matches

def visualize_matches(img1, img2, kpts1, kpts2, matches, title="SuperPoint Matches"):
    """Visualize keypoint matches between two images."""
    
    # Create side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    
    # Create output image
    if len(img1.shape) == 3:
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1+w2] = img2
    else:
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[:h1, :w1, 0] = img1
        vis[:h1, :w1, 1] = img1
        vis[:h1, :w1, 2] = img1
        vis[:h2, w1:w1+w2, 0] = img2
        vis[:h2, w1:w1+w2, 1] = img2
        vis[:h2, w1:w1+w2, 2] = img2
    
    # Draw keypoints
    # Filtered to draw only matched keypoints
    matched_kpts1_indices = {match[0] for match in matches}
    matched_kpts2_indices = {match[1] for match in matches}

    for i, kpt in enumerate(kpts1):
        if i in matched_kpts1_indices:
            cv2.circle(vis, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)
    
    for i, kpt in enumerate(kpts2):
        if i in matched_kpts2_indices:
            cv2.circle(vis, (int(kpt[0] + w1), int(kpt[1])), 3, (0, 255, 0), -1)
    
    # Draw matches
    colors = [(0, 255, 0)]  # Only Green for matches < 0.5
    
    for i, (idx1, idx2, dist) in enumerate(matches):
        # Filter out matches with distance >= 0.5
        if dist >= 0.5:
            continue

        pt1 = (int(kpts1[idx1][0]), int(kpts1[idx1][1]))
        pt2 = (int(kpts2[idx2][0] + w1), int(kpts2[idx2][1]))
        
        color = colors[0]  # All remaining matches will be green
        
        cv2.line(vis, pt1, pt2, color, 1)
    
    # Calculate average descriptor distance for displayed matches
    valid_distances = [match[2] for match in matches if match[2] < 0.5] # Only distances for drawn matches
    avg_dist = np.mean(valid_distances) if valid_distances else 0.0

    # Add text info
    info_text = f"Keypoints: {len(kpts1)} + {len(kpts2)} = {len(kpts1) + len(kpts2)} | Matches: {len(matches)} (filtered for dist < 0.5) | Avg Dist: {avg_dist:.3f}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, title, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis

def test_superpoint_matching(weights_path, img1_path, img2_path, output_path=None, rotate_img2_angle=0):
    """Test SuperPoint matching between two images."""
    
    # Load SuperPoint
    print(f"Loading SuperPoint from: {weights_path}")
    superpoint = SuperPoint(weights_path)
    
    # Load images
    print(f"Loading images...")
    img1_orig = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE) # Keep original img1 for warping later
    img2_orig = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE) # Keep original img2
    
    if img1_orig is None or img2_orig is None:
        print("Error: Could not load images")
        return (0, 0, 0) # Return zeros if images cannot be loaded
    
    img1 = img1_orig.copy()
    img2 = img2_orig.copy()

    # Apply rotation to img2 if specified
    if rotate_img2_angle != 0:
        h, w = img2.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate_img2_angle, 1.0)
        img2 = cv2.warpAffine(img2, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        print(f"Applied {rotate_img2_angle} degree rotation to Image 2.")

    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    
    # Detect keypoints and descriptors
    print("Detecting keypoints and computing descriptors...")
    kpts1, scores1, desc1 = superpoint.detect(img1)
    kpts2, scores2, desc2 = superpoint.detect(img2)
    
    print(f"Image 1: {len(kpts1)} keypoints")
    print(f"Image 2: {len(kpts2)} keypoints")
    print(f"Descriptor shapes: {desc1.shape}, {desc2.shape}")
    
    # Check descriptor quality
    if len(desc1) > 0:
        desc1_stats = f"range=[{desc1.min():.3f}, {desc1.max():.3f}], mean={desc1.mean():.3f}, std={desc1.std():.3f}"
        print(f"Descriptor 1 stats: {desc1_stats}")
    
    if len(desc2) > 0:
        desc2_stats = f"range=[{desc2.min():.3f}, {desc2.max():.3f}], mean={desc2.mean():.3f}, std={desc2.std():.3f}"
        print(f"Descriptor 2 stats: {desc2_stats}")
    
    # Match descriptors
    print("Matching descriptors...")
    matches = match_descriptors(desc1, desc2)
    print(f"Found {len(matches)} matches")
    
    # Separate matched keypoints for homography estimation
    mkpts1 = np.float32([kpts1[m[0]] for m in matches])
    mkpts2 = np.float32([kpts2[m[1]] for m in matches])

    # Visualize results for img1 vs img2
    vis_img1_img2 = visualize_matches(img1, img2, kpts1, kpts2, matches, title="SuperPoint Matches (Image 1 vs Image 2)")
    
    # Save or display first visualization
    if output_path:
        cv2.imwrite(output_path, vis_img1_img2)
        print(f"Visualization saved to: {output_path}")
    else:
        cv2.imshow("SuperPoint Matches (Image 1 vs Image 2)", vis_img1_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- New section for Homography Fitting and Comparison ---
    H_estimated = None
    if len(mkpts1) >= 4:
        # Estimate homography using RANSAC
        H_estimated, mask = cv2.findHomography(mkpts1, mkpts2, cv2.RANSAC, 5.0)
        print(f"Estimated Homography: ")
        print(H_estimated)

        # Calculate re-projection error
        if mask is not None and np.sum(mask) > 0:
            # Get inlier keypoints
            mkpts1_inliers = mkpts1[mask.ravel() == 1]
            mkpts2_inliers = mkpts2[mask.ravel() == 1]

            # Project mkpts1_inliers to img2 using estimated homography
            mkpts1_projected = cv2.perspectiveTransform(mkpts1_inliers.reshape(-1, 1, 2), H_estimated).reshape(-1, 2)

            # Calculate L2 distance between projected points and actual mkpts2_inliers
            reprojection_errors = np.linalg.norm(mkpts1_projected - mkpts2_inliers, axis=1)
            avg_reprojection_error = np.mean(reprojection_errors)
            print(f"Average Reprojection Error for Inliers: {avg_reprojection_error:.3f} pixels")
        else:
            avg_reprojection_error = float('inf') # No inliers, no meaningful error
            print("No inliers found for homography estimation, re-projection error N/A.")

    if H_estimated is not None:
        # Warp img1_orig to img2 using the estimated homography
        print("Warping Image 1 to Image 2 using estimated homography...")
        img1_warped = cv2.warpPerspective(img1_orig, H_estimated, (img2_orig.shape[1], img2_orig.shape[0]),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Now, detect and match SuperPoint features between img2_orig and img1_warped
        print("Detecting keypoints and matching between Image 2 and Warped Image 1...")
        kpts2_prime, scores2_prime, desc2_prime = superpoint.detect(img2_orig)
        kpts1_warped_prime, scores1_warped_prime, desc1_warped_prime = superpoint.detect(img1_warped)

        matches_comparison = match_descriptors(desc2_prime, desc1_warped_prime)
        print(f"Found {len(matches_comparison)} matches between Image 2 and Warped Image 1.")

        # Visualize the comparison
        output_path_comparison = None
        if output_path:
            output_path_comparison = Path(output_path).parent / f"{Path(output_path).stem}_homography_comp.png"

        vis_comparison = visualize_matches(
            img2_orig, img1_warped, kpts2_prime, kpts1_warped_prime, matches_comparison,
            title=f"SuperPoint Matches (Image 2 vs Warped Image 1) | Avg Reproj Error: {avg_reprojection_error:.3f}px"
        )

        if output_path_comparison:
            cv2.imwrite(str(output_path_comparison), vis_comparison)
            print(f"Homography comparison visualization saved to: {output_path_comparison}")
        else:
            cv2.imshow("SuperPoint Matches (Image 2 vs Warped Image 1)", vis_comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Not enough matches to estimate homography for comparison.")

    return len(kpts1), len(kpts2), len(matches)

def test_with_uav_data(weights_path, uav_data_dir, n_tests=5):
    """Test SuperPoint with random UAV image pairs."""
    
    uav_data_dir = Path(uav_data_dir)
    image_files = list(uav_data_dir.glob('*.png'))
    
    if len(image_files) < 2:
        print(f"Need at least 2 images in {uav_data_dir}")
        return
    
    print(f"Found {len(image_files)} UAV images")
    
    # Test with random pairs
    import random
    results = []
    
    for i in range(n_tests):
        # Pick two random images
        img1_path, img2_path = random.sample(image_files, 2)
        
        print(f"\n--- Test {i+1}/{n_tests} ---")
        print(f"Image 1: {img1_path.name}")
        print(f"Image 2: {img2_path.name}")
        
        output_path = f"superpoint_test_{i+1}.png"
        
        try:
            n_kpts1, n_kpts2, n_matches = test_superpoint_matching(
                weights_path, str(img1_path), str(img2_path), output_path
            )
            results.append((n_kpts1, n_kpts2, n_matches))
        except Exception as e:
            print(f"Error in test {i+1}: {e}")
    
    # Summary
    if results:
        avg_kpts1 = np.mean([r[0] for r in results])
        avg_kpts2 = np.mean([r[1] for r in results])
        avg_matches = np.mean([r[2] for r in results])
        
        print(f"\n--- Summary ---")
        print(f"Average keypoints per image: {avg_kpts1:.1f}, {avg_kpts2:.1f}")
        print(f"Average matches per pair: {avg_matches:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SuperPoint matches")
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to SuperPoint weights')
    parser.add_argument('--img1', type=str,
                       help='Path to first image')
    parser.add_argument('--img2', type=str,
                       help='Path to second image')
    parser.add_argument('--uav_data', type=str,
                       help='Directory with UAV data for random testing')
    parser.add_argument('--output', type=str,
                       help='Output path for visualization')
    parser.add_argument('--n_tests', type=int, default=5,
                       help='Number of random tests with UAV data')
    parser.add_argument('--rotate_img2_angle', type=float, default=0,
                       help='Angle in degrees to rotate img2 for testing (0 for no rotation).')
    
    args = parser.parse_args()
    
    if args.img1 and args.img2:
        # Test specific image pair
        test_superpoint_matching(args.weights, args.img1, args.img2, args.output, args.rotate_img2_angle)
    elif args.uav_data:
        # Test with random UAV data
        # Note: Rotation is not applied in this mode, as it expects original pairs.
        test_with_uav_data(args.weights, args.uav_data, args.n_tests)
    else:
        print("Please provide either --img1 and --img2, or --uav_data") 