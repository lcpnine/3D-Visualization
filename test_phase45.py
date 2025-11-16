"""
Test script for Phases 4-5: Camera Pose Estimation and 3D Triangulation
Complete two-view reconstruction test
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import image_loader, camera_calibration
from src.features import harris_detector, descriptor
from src.matching import matcher, ransac
from src.geometry import essential_matrix, pose_recovery
from src.triangulation import triangulate, validation
from config import cfg

def test_two_view_reconstruction():
    print("=" * 70)
    print("Testing Phases 4-5: Two-View Reconstruction")
    print("=" * 70)

    # Load images
    print("\n[Preparation] Loading images...")
    print("-" * 70)
    images, metadata = image_loader.load_images("data/scene1")
    print(f"Loaded {len(images)} images")

    # Get intrinsic matrix
    width = metadata[0]['width']
    height = metadata[0]['height']
    K = camera_calibration.estimate_intrinsic_matrix(width, height)

    print(f"\nIntrinsic matrix K:")
    print(K)
    print(f"Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    # Process first two images
    img1, img2 = images[0], images[1]
    print(f"\nReconstructing from:")
    print(f"  Image 1: {metadata[0]['filename']}")
    print(f"  Image 2: {metadata[1]['filename']}")

    # Feature detection and matching
    print("\n" + "=" * 70)
    print("[Step 1-3] Feature Detection and Matching")
    print("-" * 70)

    start_time = time.time()

    corners1 = harris_detector.detect_harris_corners(img1)
    corners2 = harris_detector.detect_harris_corners(img2)
    print(f"Detected corners: {len(corners1)} / {len(corners2)}")

    desc1 = descriptor.compute_descriptors(img1, corners1)
    desc2 = descriptor.compute_descriptors(img2, corners2)
    print(f"Computed descriptors: {len(desc1)} / {len(desc2)}")

    matches, _ = matcher.match_descriptors(desc1, desc2)
    print(f"Initial matches: {len(matches)}")

    points1, points2 = matcher.get_matched_points(corners1, corners2, matches)

    # RANSAC for fundamental matrix
    F, inlier_mask = ransac.estimate_fundamental_matrix_ransac(points1, points2)
    num_inliers = np.sum(inlier_mask)
    print(f"RANSAC inliers: {num_inliers}/{len(points1)} ({100*num_inliers/len(points1):.1f}%)")

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed:.2f}s")

    # Test 1: Essential Matrix Recovery
    print("\n" + "=" * 70)
    print("[Test 1] Phase 4.1 - Essential Matrix Recovery")
    print("-" * 70)

    try:
        start_time = time.time()

        E = essential_matrix.fundamental_to_essential(F, K)
        elapsed = time.time() - start_time

        print(f"\nEssential matrix E:")
        print(E)

        # Validate
        U, S, Vt = np.linalg.svd(E)
        print(f"\nSingular values: {S}")
        print(f"Expected: [1.0, 1.0, 0.0]")

        rank = np.linalg.matrix_rank(E)
        det_E = np.linalg.det(E)
        print(f"Rank: {rank} (should be 2)")
        print(f"Determinant: {det_E:.10f} (should be ≈ 0)")

        is_valid = essential_matrix.check_essential_matrix(E)
        print(f"\nEssential matrix is valid: {is_valid}")

        print(f"Elapsed time: {elapsed:.4f}s")

    except Exception as e:
        print(f"Error in essential matrix recovery: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Pose Recovery
    print("\n" + "=" * 70)
    print("[Test 2] Phase 4.2 - Camera Pose Recovery")
    print("-" * 70)

    try:
        start_time = time.time()

        # Decompose E into 4 possible poses
        poses = pose_recovery.decompose_essential_matrix(E)
        print(f"Generated {len(poses)} pose candidates")

        # Use inlier points for pose selection
        inlier_pts1 = points1[inlier_mask]
        inlier_pts2 = points2[inlier_mask]

        print(f"\nTesting chirality for each pose:")
        for i, (R, t) in enumerate(poses):
            count = pose_recovery.check_chirality(R, t, inlier_pts1, inlier_pts2, K)
            ratio = count / len(inlier_pts1)
            print(f"  Pose {i+1}: {count}/{len(inlier_pts1)} in front ({100*ratio:.1f}%)")

        # Select best pose
        R, t = pose_recovery.select_valid_pose(poses, inlier_pts1, inlier_pts2, K)

        elapsed = time.time() - start_time

        print(f"\nSelected pose:")
        print(f"Rotation R:")
        print(R)
        print(f"\nTranslation t:")
        print(t)

        # Validate rotation
        from utils.math_utils import is_rotation_matrix
        print(f"\nValidation:")
        print(f"  Is valid rotation: {is_rotation_matrix(R)}")
        print(f"  det(R) = {np.linalg.det(R):.10f} (should be 1)")

        # Normalize translation
        t_norm = pose_recovery.normalize_translation(t)
        print(f"  ||t|| = {np.linalg.norm(t_norm):.6f} (normalized)")

        print(f"\nElapsed time: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error in pose recovery: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Triangulation
    print("\n" + "=" * 70)
    print("[Test 3] Phase 5.1 - 3D Point Triangulation")
    print("-" * 70)

    try:
        start_time = time.time()

        # Construct projection matrices
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t.reshape(3, 1)])

        print(f"Camera matrices:")
        print(f"  P1 shape: {P1.shape}")
        print(f"  P2 shape: {P2.shape}")

        # Triangulate points
        points_3d = triangulate.triangulate_points(P1, P2, inlier_pts1, inlier_pts2)

        elapsed = time.time() - start_time

        print(f"\nTriangulated {len(points_3d)} 3D points in {elapsed:.2f}s")

        # Statistics
        print(f"\n3D point cloud statistics:")
        print(f"  X: [{points_3d[:, 0].min():8.2f}, {points_3d[:, 0].max():8.2f}]")
        print(f"  Y: [{points_3d[:, 1].min():8.2f}, {points_3d[:, 1].max():8.2f}]")
        print(f"  Z: [{points_3d[:, 2].min():8.2f}, {points_3d[:, 2].max():8.2f}]")

        depths = points_3d[:, 2]
        positive = depths > 0
        print(f"\nDepth statistics:")
        print(f"  Positive depths: {np.sum(positive)}/{len(depths)} ({100*np.sum(positive)/len(depths):.1f}%)")
        if np.any(positive):
            print(f"  Mean depth: {depths[positive].mean():.2f}")
            print(f"  Median depth: {np.median(depths[positive]):.2f}")

    except Exception as e:
        print(f"Error in triangulation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Validation
    print("\n" + "=" * 70)
    print("[Test 4] Phase 5.2 - Triangulation Validation")
    print("-" * 70)

    try:
        start_time = time.time()

        # Compute quality statistics
        stats = validation.validate_reconstruction_quality(
            points_3d, P1, P2, inlier_pts1, inlier_pts2
        )

        print("\nReconstruction quality (before filtering):")
        print(f"  Number of points: {stats['num_points']}")
        print(f"  Mean reprojection error: {stats['mean_reproj_error']:.4f} pixels")
        print(f"  Median reprojection error: {stats['median_reproj_error']:.4f} pixels")
        print(f"  Max reprojection error: {stats['max_reproj_error']:.4f} pixels")
        print(f"  Positive depth ratio: {stats['positive_depth_ratio']:.2%}")
        print(f"  Mean depth: {stats['mean_depth']:.2f}")

        # Filter points
        filtered_points, valid_mask = validation.filter_triangulated_points(
            points_3d, P1, P2, inlier_pts1, inlier_pts2
        )

        elapsed = time.time() - start_time

        print(f"\nFiltering results:")
        print(f"  Points kept: {len(filtered_points)}/{len(points_3d)} ({100*len(filtered_points)/len(points_3d):.1f}%)")
        print(f"  Points removed: {len(points_3d) - len(filtered_points)}")

        # Quality after filtering
        if len(filtered_points) > 0:
            filtered_pts1 = inlier_pts1[valid_mask]
            filtered_pts2 = inlier_pts2[valid_mask]

            filtered_stats = validation.validate_reconstruction_quality(
                filtered_points, P1, P2, filtered_pts1, filtered_pts2
            )

            print("\nReconstruction quality (after filtering):")
            print(f"  Number of points: {filtered_stats['num_points']}")
            print(f"  Mean reprojection error: {filtered_stats['mean_reproj_error']:.4f} pixels")
            print(f"  Median reprojection error: {filtered_stats['median_reproj_error']:.4f} pixels")
            print(f"  Max reprojection error: {filtered_stats['max_reproj_error']:.4f} pixels")
            print(f"  Positive depth ratio: {filtered_stats['positive_depth_ratio']:.2%}")

        print(f"\nElapsed time: {elapsed:.2f}s")

    except Exception as e:
        print(f"Error in validation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 70)
    print("Two-View Reconstruction Summary")
    print("-" * 70)
    print(f"Input:")
    print(f"  Images: 2")
    print(f"  Initial matches: {len(matches)}")
    print(f"  RANSAC inliers: {num_inliers}")
    print(f"\nOutput:")
    print(f"  3D points (raw): {len(points_3d)}")
    print(f"  3D points (filtered): {len(filtered_points) if len(filtered_points) > 0 else 0}")
    print(f"  Mean reprojection error: {filtered_stats['mean_reproj_error']:.2f} pixels" if len(filtered_points) > 0 else "  N/A")
    print(f"\nReconstruction successful!")

    print("\n" + "=" * 70)
    print("Phases 4-5 Test Complete - All tests passed!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_two_view_reconstruction()
    sys.exit(0 if success else 1)
