#!/usr/bin/env python3
"""
Comprehensive SuperPoint Benchmarking Script
Evaluates keypoint detection, descriptor matching, and homography estimation.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from simple_superpoint import SuperPoint
import random

class SuperPointBenchmark:
    def __init__(self, device="cuda"):
        self.device = device
        self.results = {}
        
    def load_model(self, weights_path):
        """Load SuperPoint model from weights."""
        print(f"Loading SuperPoint from: {weights_path}")
        try:
            self.model = SuperPoint(weights_path, device=self.device)
            print("✓ Model loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def benchmark_keypoint_detection(self, test_images, n_tests=50):
        """Benchmark keypoint detection quality."""
        print(f"\n=== Keypoint Detection Benchmark ===")
        
        keypoint_stats = {
            'num_keypoints': [],
            'keypoint_scores': [],
            'keypoint_distribution': [],
            'detection_rate': 0
        }
        
        successful_detections = 0
        
        for i in tqdm(range(min(n_tests, len(test_images))), desc="Testing keypoint detection"):
            img_path = test_images[i]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # Detect keypoints
            kpts, scores, desc = self.model.detect(img)
            
            if len(kpts) > 0:
                successful_detections += 1
                keypoint_stats['num_keypoints'].append(len(kpts))
                keypoint_stats['keypoint_scores'].extend(scores.tolist())
                
                # Analyze keypoint distribution
                h, w = img.shape
                x_norm = kpts[:, 0] / w
                y_norm = kpts[:, 1] / h
                keypoint_stats['keypoint_distribution'].extend(list(zip(x_norm, y_norm)))
        
        # Calculate statistics
        if keypoint_stats['num_keypoints']:
            keypoint_stats['detection_rate'] = float(successful_detections / n_tests)
            keypoint_stats['avg_keypoints'] = float(np.mean(keypoint_stats['num_keypoints']))
            keypoint_stats['std_keypoints'] = float(np.std(keypoint_stats['num_keypoints']))
            keypoint_stats['min_keypoints'] = int(np.min(keypoint_stats['num_keypoints']))
            keypoint_stats['max_keypoints'] = int(np.max(keypoint_stats['num_keypoints']))
            keypoint_stats['avg_score'] = float(np.mean(keypoint_stats['keypoint_scores']))
            keypoint_stats['std_score'] = float(np.std(keypoint_stats['keypoint_scores']))
        
        print(f"Detection Rate: {keypoint_stats['detection_rate']:.3f}")
        print(f"Average Keypoints: {keypoint_stats.get('avg_keypoints', 0):.1f} ± {keypoint_stats.get('std_keypoints', 0):.1f}")
        print(f"Keypoint Range: {keypoint_stats.get('min_keypoints', 0)} - {keypoint_stats.get('max_keypoints', 0)}")
        print(f"Average Score: {keypoint_stats.get('avg_score', 0):.3f} ± {keypoint_stats.get('std_score', 0):.3f}")
        
        return keypoint_stats
    
    def benchmark_descriptor_matching(self, test_images, n_tests=30):
        """Benchmark descriptor matching performance."""
        print(f"\n=== Descriptor Matching Benchmark ===")
        
        matching_stats = {
            'num_matches': [],
            'match_ratios': [],
            'descriptor_distances': [],
            'homography_errors': [],
            'successful_homographies': 0
        }
        
        for i in tqdm(range(min(n_tests, len(test_images)//2)), desc="Testing descriptor matching"):
            # Pick two random images
            img1_path, img2_path = random.sample(test_images, 2)
            
            img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                continue
            
            # Detect keypoints and descriptors
            kpts1, scores1, desc1 = self.model.detect(img1)
            kpts2, scores2, desc2 = self.model.detect(img2)
            
            if len(kpts1) < 10 or len(kpts2) < 10:
                continue
            
            # Match descriptors
            matches = self._match_descriptors(desc1, desc2)
            
            if len(matches) >= 4:
                matching_stats['num_matches'].append(len(matches))
                matching_stats['match_ratios'].append(len(matches) / min(len(kpts1), len(kpts2)))
                
                # Calculate descriptor distances
                distances = [match[2] for match in matches]
                matching_stats['descriptor_distances'].extend(distances)
                
                # Estimate homography
                homography_error = self._estimate_homography_error(kpts1, kpts2, matches)
                if homography_error is not None:
                    matching_stats['homography_errors'].append(homography_error)
                    matching_stats['successful_homographies'] += 1
        
        # Calculate statistics
        if matching_stats['num_matches']:
            matching_stats['avg_matches'] = float(np.mean(matching_stats['num_matches']))
            matching_stats['avg_match_ratio'] = float(np.mean(matching_stats['match_ratios']))
            matching_stats['avg_descriptor_distance'] = float(np.mean(matching_stats['descriptor_distances']))
            matching_stats['homography_success_rate'] = float(matching_stats['successful_homographies'] / len(matching_stats['num_matches']))
            
            if matching_stats['homography_errors']:
                matching_stats['avg_homography_error'] = float(np.mean(matching_stats['homography_errors']))
        
        print(f"Average Matches: {matching_stats.get('avg_matches', 0):.1f}")
        print(f"Average Match Ratio: {matching_stats.get('avg_match_ratio', 0):.3f}")
        print(f"Average Descriptor Distance: {matching_stats.get('avg_descriptor_distance', 0):.3f}")
        print(f"Homography Success Rate: {matching_stats.get('homography_success_rate', 0):.3f}")
        if matching_stats.get('avg_homography_error'):
            print(f"Average Homography Error: {matching_stats['avg_homography_error']:.2f} pixels")
        
        return matching_stats
    
    def benchmark_homographic_adaptation(self, test_images, n_tests=20):
        """Benchmark performance on homographic pairs (ground truth available)."""
        print(f"\n=== Homographic Adaptation Benchmark ===")
        
        adaptation_stats = {
            'reprojection_errors': [],
            'inlier_ratios': [],
            'successful_adaptations': 0
        }
        
        for i in tqdm(range(min(n_tests, len(test_images))), desc="Testing homographic adaptation"):
            img_path = test_images[i]
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Generate homographic pair
            img1 = img.copy()
            img2, H_gt = self._apply_homography(img1)
            
            # Detect keypoints and descriptors
            kpts1, scores1, desc1 = self.model.detect(img1)
            kpts2, scores2, desc2 = self.model.detect(img2)
            
            if len(kpts1) < 10 or len(kpts2) < 10:
                continue
            
            # Match descriptors
            matches = self._match_descriptors(desc1, desc2)
            
            if len(matches) >= 4:
                # Estimate homography
                mkpts1 = kpts1[[m[0] for m in matches]]
                mkpts2 = kpts2[[m[1] for m in matches]]
                
                H_est, mask = cv2.findHomography(mkpts1, mkpts2, cv2.RANSAC, 5.0)
                
                if H_est is not None and mask is not None:
                    # Calculate reprojection error
                    inliers = mask.ravel() == 1
                    if np.sum(inliers) >= 4:
                        mkpts1_inliers = mkpts1[inliers]
                        mkpts2_inliers = mkpts2[inliers]
                        
                        # Project points using estimated homography
                        mkpts1_proj = cv2.perspectiveTransform(
                            mkpts1_inliers.reshape(-1, 1, 2), H_est
                        ).reshape(-1, 2)
                        
                        # Calculate reprojection error
                        errors = np.linalg.norm(mkpts1_proj - mkpts2_inliers, axis=1)
                        avg_error = np.mean(errors)
                        
                        adaptation_stats['reprojection_errors'].append(avg_error)
                        adaptation_stats['inlier_ratios'].append(np.sum(inliers) / len(matches))
                        adaptation_stats['successful_adaptations'] += 1
        
        # Calculate statistics
        if adaptation_stats['reprojection_errors']:
            adaptation_stats['avg_reprojection_error'] = float(np.mean(adaptation_stats['reprojection_errors']))
            adaptation_stats['avg_inlier_ratio'] = float(np.mean(adaptation_stats['inlier_ratios']))
            adaptation_stats['adaptation_success_rate'] = float(adaptation_stats['successful_adaptations'] / n_tests)
        
        print(f"Adaptation Success Rate: {adaptation_stats.get('adaptation_success_rate', 0):.3f}")
        if adaptation_stats.get('avg_reprojection_error'):
            print(f"Average Reprojection Error: {adaptation_stats['avg_reprojection_error']:.2f} pixels")
        if adaptation_stats.get('avg_inlier_ratio'):
            print(f"Average Inlier Ratio: {adaptation_stats['avg_inlier_ratio']:.3f}")
        
        return adaptation_stats
    
    def _match_descriptors(self, desc1, desc2, threshold=0.8):
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
    
    def _estimate_homography_error(self, kpts1, kpts2, matches):
        """Estimate homography and return error metric."""
        if len(matches) < 4:
            return None
        
        mkpts1 = kpts1[[m[0] for m in matches]]
        mkpts2 = kpts2[[m[1] for m in matches]]
        
        H, mask = cv2.findHomography(mkpts1, mkpts2, cv2.RANSAC, 5.0)
        
        if H is not None and mask is not None:
            inliers = mask.ravel() == 1
            if np.sum(inliers) >= 4:
                mkpts1_inliers = mkpts1[inliers]
                mkpts2_inliers = mkpts2[inliers]
                
                # Project points using estimated homography
                mkpts1_proj = cv2.perspectiveTransform(
                    mkpts1_inliers.reshape(-1, 1, 2), H
                ).reshape(-1, 2)
                
                # Calculate average reprojection error
                errors = np.linalg.norm(mkpts1_proj - mkpts2_inliers, axis=1)
                return np.mean(errors)
        
        return None
    
    def _apply_homography(self, img):
        """Apply random homography to image."""
        h, w = img.shape
        
        # Define corners
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        
        # Use smaller, more reasonable perturbations
        max_perturbation = min(w, h) * 0.1  # 10% of image size
        perturbation = np.random.uniform(-max_perturbation, max_perturbation, (4, 2)).astype(np.float32)
        perturbed_corners = corners + perturbation
        
        # Ensure corners stay within reasonable bounds
        margin = 10
        perturbed_corners[:, 0] = np.clip(perturbed_corners[:, 0], margin, w - margin - 1)
        perturbed_corners[:, 1] = np.clip(perturbed_corners[:, 1], margin, h - margin - 1)
        
        # Compute homography
        H = cv2.getPerspectiveTransform(corners, perturbed_corners)
        
        # Validate homography is not degenerate
        det = np.linalg.det(H[:2, :2])
        if abs(det) < 1e-6:
            H = np.eye(3, dtype=np.float32)
            warped = img.copy()
        else:
            warped = cv2.warpPerspective(img, H, (w, h))
        
        return warped, H
    
    def _assess_overall_performance(self, results):
        """Assess overall model performance and provide recommendations."""
        assessment = {
            'score': 0,
            'grade': 'F',
            'issues': [],
            'recommendations': []
        }
        
        score = 0
        
        # Keypoint detection assessment
        kp_stats = results['keypoint_detection']
        if kp_stats.get('avg_keypoints', 0) < 30:
            assessment['issues'].append("Low keypoint count (<30 per image)")
            assessment['recommendations'].append("Train detector head or increase training epochs")
        elif kp_stats.get('avg_keypoints', 0) > 100:
            score += 20
        else:
            score += 15
        
        if kp_stats.get('detection_rate', 0) < 0.8:
            assessment['issues'].append("Low detection rate (<80%)")
        else:
            score += 20
        
        # Descriptor matching assessment
        dm_stats = results['descriptor_matching']
        if dm_stats.get('avg_match_ratio', 0) < 0.1:
            assessment['issues'].append("Poor descriptor matching (<10% match ratio)")
            assessment['recommendations'].append("Train descriptor head with more epochs")
        elif dm_stats.get('avg_match_ratio', 0) > 0.3:
            score += 30
        else:
            score += 20
        
        if dm_stats.get('avg_descriptor_distance', 1.0) > 0.8:
            assessment['issues'].append("High descriptor distances (>0.8)")
            assessment['recommendations'].append("Improve descriptor training")
        else:
            score += 20
        
        # Homographic adaptation assessment
        ha_stats = results['homographic_adaptation']
        if ha_stats.get('avg_reprojection_error', float('inf')) > 5.0:
            assessment['issues'].append("High reprojection error (>5 pixels)")
            assessment['recommendations'].append("Train with homographic adaptation")
        else:
            score += 20
        
        # Determine grade
        if score >= 90:
            assessment['grade'] = 'A'
        elif score >= 80:
            assessment['grade'] = 'B'
        elif score >= 70:
            assessment['grade'] = 'C'
        elif score >= 60:
            assessment['grade'] = 'D'
        else:
            assessment['grade'] = 'F'
        
        assessment['score'] = score
        
        if score < 70:
            assessment['recommendations'].append("Consider training for more epochs (50-100)")
            assessment['recommendations'].append("Try training both detector and descriptor heads")
            assessment['recommendations'].append("Increase batch size and learning rate")
        
        return assessment
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def run_full_benchmark(self, weights_path, test_data_dir, output_dir=None):
        """Run complete benchmark suite."""
        print(f"🚁 SuperPoint Benchmark Suite")
        print(f"Model: {weights_path}")
        print(f"Test Data: {test_data_dir}")
        print("=" * 60)
        
        # Load model
        if not self.load_model(weights_path):
            return None
        
        # Load test images
        test_data_path = Path(test_data_dir)
        test_images = list(test_data_path.glob('*.png')) + list(test_data_path.glob('*.jpg'))
        
        if len(test_images) == 0:
            print(f"❌ No test images found in {test_data_dir}")
            return None
        
        print(f"Found {len(test_images)} test images")
        
        # Run benchmarks
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(weights_path),
            'test_data_dir': str(test_data_dir),
            'num_test_images': len(test_images)
        }
        
        # Keypoint detection benchmark
        results['keypoint_detection'] = self.benchmark_keypoint_detection(test_images)
        
        # Descriptor matching benchmark
        results['descriptor_matching'] = self.benchmark_descriptor_matching(test_images)
        
        # Homographic adaptation benchmark
        results['homographic_adaptation'] = self.benchmark_homographic_adaptation(test_images)
        
        # Overall assessment
        results['overall_assessment'] = self._assess_overall_performance(results)
        
        # Convert numpy types to Python types for JSON serialization
        results = self._convert_numpy_types(results)
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            results_file = output_path / f"superpoint_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate summary report
            self._generate_summary_report(results, output_path)
            
            print(f"\nResults saved to: {output_path}")
        
        return results
    
    def _generate_summary_report(self, results, output_path):
        """Generate a human-readable summary report."""
        report_file = output_path / "benchmark_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("SuperPoint Benchmark Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Test Data: {results['test_data_dir']}\n")
            f.write(f"Test Images: {results['num_test_images']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            # Keypoint detection
            kp = results['keypoint_detection']
            f.write("KEYPOINT DETECTION:\n")
            f.write(f"  Detection Rate: {kp.get('detection_rate', 0):.3f}\n")
            f.write(f"  Average Keypoints: {kp.get('avg_keypoints', 0):.1f} ± {kp.get('std_keypoints', 0):.1f}\n")
            f.write(f"  Average Score: {kp.get('avg_score', 0):.3f}\n\n")
            
            # Descriptor matching
            dm = results['descriptor_matching']
            f.write("DESCRIPTOR MATCHING:\n")
            f.write(f"  Average Matches: {dm.get('avg_matches', 0):.1f}\n")
            f.write(f"  Match Ratio: {dm.get('avg_match_ratio', 0):.3f}\n")
            f.write(f"  Descriptor Distance: {dm.get('avg_descriptor_distance', 0):.3f}\n")
            f.write(f"  Homography Success: {dm.get('homography_success_rate', 0):.3f}\n\n")
            
            # Homographic adaptation
            ha = results['homographic_adaptation']
            f.write("HOMOGRAPHIC ADAPTATION:\n")
            f.write(f"  Success Rate: {ha.get('adaptation_success_rate', 0):.3f}\n")
            if ha.get('avg_reprojection_error'):
                f.write(f"  Reprojection Error: {ha['avg_reprojection_error']:.2f} pixels\n")
            if ha.get('avg_inlier_ratio'):
                f.write(f"  Inlier Ratio: {ha['avg_inlier_ratio']:.3f}\n\n")
            
            # Overall assessment
            assessment = results['overall_assessment']
            f.write("OVERALL ASSESSMENT:\n")
            f.write(f"  Grade: {assessment['grade']} ({assessment['score']}/100)\n\n")
            
            if assessment['issues']:
                f.write("ISSUES FOUND:\n")
                for issue in assessment['issues']:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            if assessment['recommendations']:
                f.write("RECOMMENDATIONS:\n")
                for rec in assessment['recommendations']:
                    f.write(f"  - {rec}\n")

def main():
    parser = argparse.ArgumentParser(description="SuperPoint Benchmark Suite")
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to SuperPoint weights (.pth file)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='superpoint_benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = SuperPointBenchmark(device=args.device)
    results = benchmark.run_full_benchmark(args.weights, args.test_data, args.output_dir)
    
    if results:
        assessment = results['overall_assessment']
        print(f"\n🎯 Overall Grade: {assessment['grade']} ({assessment['score']}/100)")
        
        if assessment['issues']:
            print("\n⚠️ Issues Found:")
            for issue in assessment['issues']:
                print(f"  - {issue}")
        
        if assessment['recommendations']:
            print("\n💡 Recommendations:")
            for rec in assessment['recommendations']:
                print(f"  - {rec}")

if __name__ == "__main__":
    main() 