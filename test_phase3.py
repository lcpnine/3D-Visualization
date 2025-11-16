"""
Test script for Phase 3: Feature Matching
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import image_loader
from src.features import harris_detector, descriptor
from src.matching import matcher, ransac
from config import cfg

def test_phase3():
    print("=" * 70)
    print("Testing Phase 3: Feature Matching")
    print("=" * 70)

    # Load images
    print("\n[Preparation] Loading images...")
    print("-" * 70)
    images, metadata = image_loader.load_images("data/scene1")
    print(f"Loaded {len(images)} images")

    # Process first two images
    img1, img2 = images[0], images[1]
    print(f"\nTesting on:")
    print(f"  Image 1: {metadata[0]['filename']}")
    print(f"  Image 2: {metadata[1]['filename']}")

    # Detect features
    print("\nDetecting features...")
    corners1 = harris_detector.detect_harris_corners(img1)
    corners2 = harris_detector.detect_harris_corners(img2)
    print(f"  Image 1: {len(corners1)} corners")
    print(f"  Image 2: {len(corners2)} corners")

    # Compute descriptors
    print("\nComputing descriptors...")
    desc1 = descriptor.compute_descriptors(img1, corners1)
    desc2 = descriptor.compute_descriptors(img2, corners2)
    print(f"  Image 1: {len(desc1)} descriptors")
    print(f"  Image 2: {len(desc2)} descriptors")

    # Test 1: Descriptor Matching
    print("\n" + "=" * 70)
    print("[Test 1] Descriptor Matching with Ratio Test")
    print("-" * 70)

    try:
        start_time = time.time()
        matches, distances = matcher.match_descriptors(desc1, desc2)
        elapsed = time.time() - start_time

        print(f"\nFound {len(matches)} matches in {elapsed:.2f} seconds")
        print(f"\nMatch distance statistics:")
        print(f"  Mean: {distances.mean():.4f}")
        print(f"  Std: {distances.std():.4f}")
        print(f"  Min: {distances.min():.4f}")
        print(f"  Max: {distances.max():.4f}")

        print(f"\nFirst 5 matches (descriptor indices):")
        for i, (idx1, idx2) in enumerate(matches[:5]):
            print(f"  {i+1}. ({idx1:4d}, {idx2:4d}) - distance: {distances[i]:.4f}")

        # Get matched points
        points1, points2 = matcher.get_matched_points(corners1, corners2, matches)
        print(f"\nMatched points extracted: {len(points1)}")

    except Exception as e:
        print(f"Error in descriptor matching: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: RANSAC and Fundamental Matrix
    print("\n" + "=" * 70)
    print("[Test 2] RANSAC and Fundamental Matrix Estimation")
    print("-" * 70)

    try:
        start_time = time.time()
        F, inlier_mask = ransac.estimate_fundamental_matrix_ransac(points1, points2)
        elapsed = time.time() - start_time

        num_inliers = np.sum(inlier_mask)
        inlier_ratio = num_inliers / len(points1)

        print(f"\nRANSAC completed in {elapsed:.2f} seconds")
        print(f"\nFundamental Matrix F:")
        print(F)

        print(f"\nInlier statistics:")
        print(f"  Total matches: {len(points1)}")
        print(f"  Inliers: {num_inliers}")
        print(f"  Outliers: {len(points1) - num_inliers}")
        print(f"  Inlier ratio: {inlier_ratio:.1%}")

        # Validate fundamental matrix
        rank = np.linalg.matrix_rank(F)
        det_F = np.linalg.det(F)
        print(f"\nFundamental matrix properties:")
        print(f"  Rank: {rank} (should be 2)")
        print(f"  Determinant: {det_F:.10f} (should be ≈ 0)")
        print(f"  Frobenius norm: {np.linalg.norm(F):.6f}")

        # Compute errors on inliers
        inlier_pts1 = points1[inlier_mask]
        inlier_pts2 = points2[inlier_mask]

        sampson_dist = ransac.compute_sampson_distance(F, inlier_pts1, inlier_pts2)
        epipolar_err = ransac.compute_epipolar_error(F, inlier_pts1, inlier_pts2)

        print(f"\nInlier error statistics:")
        print(f"  Sampson distance:")
        print(f"    Mean: {sampson_dist.mean():.4f} pixels")
        print(f"    Max: {sampson_dist.max():.4f} pixels")
        print(f"    Std: {sampson_dist.std():.4f} pixels")
        print(f"  Epipolar error (algebraic):")
        print(f"    Mean: {epipolar_err.mean():.6f}")
        print(f"    Max: {epipolar_err.max():.6f}")

        # Check if inlier ratio is acceptable
        if inlier_ratio < cfg.MIN_INLIERS / len(points1):
            print(f"\nWarning: Low inlier ratio! May indicate poor matches or scene.")
        else:
            print(f"\nInlier ratio is good!")

    except Exception as e:
        print(f"Error in RANSAC: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Multiple Image Pairs
    print("\n" + "=" * 70)
    print("[Test 3] Testing Multiple Image Pairs")
    print("-" * 70)

    try:
        num_test_pairs = min(3, len(images) - 1)
        print(f"\nProcessing {num_test_pairs} image pairs...")

        results = []
        for i in range(num_test_pairs):
            print(f"\nPair {i+1}: {metadata[i]['filename']} <-> {metadata[i+1]['filename']}")

            # Detect and match
            c1 = harris_detector.detect_harris_corners(images[i])
            c2 = harris_detector.detect_harris_corners(images[i+1])
            d1 = descriptor.compute_descriptors(images[i], c1)
            d2 = descriptor.compute_descriptors(images[i+1], c2)
            m, _ = matcher.match_descriptors(d1, d2)

            if len(m) < cfg.MIN_MATCHES:
                print(f"  Too few matches: {len(m)} < {cfg.MIN_MATCHES}")
                continue

            # RANSAC
            p1, p2 = matcher.get_matched_points(c1, c2, m)
            F_i, inliers_i = ransac.estimate_fundamental_matrix_ransac(p1, p2)

            num_inliers_i = np.sum(inliers_i)
            results.append({
                'pair': (i, i+1),
                'matches': len(m),
                'inliers': num_inliers_i,
                'ratio': num_inliers_i / len(m)
            })

            print(f"  Matches: {len(m)}, Inliers: {num_inliers_i} ({100*num_inliers_i/len(m):.1f}%)")

        print(f"\nSummary of {len(results)} image pairs:")
        print(f"  Average matches per pair: {np.mean([r['matches'] for r in results]):.1f}")
        print(f"  Average inliers per pair: {np.mean([r['inliers'] for r in results]):.1f}")
        print(f"  Average inlier ratio: {np.mean([r['ratio'] for r in results]):.1%}")

    except Exception as e:
        print(f"Error processing multiple pairs: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("Phase 3 Test Complete - All tests passed!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_phase3()
    sys.exit(0 if success else 1)
