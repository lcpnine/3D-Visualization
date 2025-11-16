"""
Phase 5 - Step 5.2: Triangulation Quality Validation
Filters poorly triangulated points based on reprojection error and geometric constraints.
"""

import numpy as np
from typing import Tuple
from config import cfg


def compute_reprojection_error(X: np.ndarray, P: np.ndarray, observed_p: np.ndarray) -> float:
    """
    Compute reprojection error for a 3D point

    Args:
        X: 3D point (3,)
        P: Camera projection matrix (3, 4)
        observed_p: Observed 2D point (2,)

    Returns:
        Reprojection error in pixels
    """
    # Project 3D point
    X_h = np.append(X, 1)  # Homogeneous coordinates
    projected = P @ X_h

    # Convert to Euclidean
    if abs(projected[2]) < 1e-10:
        return float('inf')

    projected_2d = projected[:2] / projected[2]

    # Compute error
    error = np.linalg.norm(projected_2d - observed_p)

    return error


def compute_all_reprojection_errors(points_3d: np.ndarray, P: np.ndarray,
                                    observed_points: np.ndarray) -> np.ndarray:
    """
    Compute reprojection errors for all points

    Args:
        points_3d: 3D points (N, 3)
        P: Camera projection matrix (3, 4)
        observed_points: Observed 2D points (N, 2)

    Returns:
        errors: Reprojection errors (N,)
    """
    N = len(points_3d)
    errors = np.zeros(N)

    for i in range(N):
        errors[i] = compute_reprojection_error(points_3d[i], P, observed_points[i])

    return errors


def check_parallax_angle(X: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> float:
    """
    Compute parallax angle between two camera centers viewing a 3D point

    Args:
        X: 3D point (3,)
        C1: Camera center 1 (3,)
        C2: Camera center 2 (3,)

    Returns:
        Parallax angle in degrees
    """
    # Vectors from cameras to point
    v1 = X - C1
    v2 = X - C2

    # Normalize
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

    # Angle
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg


def filter_triangulated_points(points_3d: np.ndarray,
                               P1: np.ndarray, P2: np.ndarray,
                               pts1: np.ndarray, pts2: np.ndarray,
                               reproj_threshold: float = None,
                               min_parallax: float = None,
                               min_depth: float = None,
                               max_depth: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter triangulated points based on quality criteria

    Args:
        points_3d: 3D points (N, 3)
        P1: Camera matrix 1 (3, 4)
        P2: Camera matrix 2 (3, 4)
        pts1: Observed 2D points in image 1 (N, 2)
        pts2: Observed 2D points in image 2 (N, 2)
        reproj_threshold: Maximum reprojection error (default: from config)
        min_parallax: Minimum parallax angle in degrees (default: from config)
        min_depth: Minimum depth (default: from config)
        max_depth: Maximum depth (default: from config)

    Returns:
        filtered_points: Filtered 3D points (M, 3)
        valid_mask: Boolean mask of valid points (N,)
    """
    if reproj_threshold is None:
        reproj_threshold = cfg.REPROJ_ERROR_THRESHOLD
    if min_parallax is None:
        min_parallax = cfg.MIN_PARALLAX_ANGLE
    if min_depth is None:
        min_depth = cfg.MIN_DEPTH
    if max_depth is None:
        max_depth = cfg.MAX_DEPTH

    N = len(points_3d)
    valid_mask = np.ones(N, dtype=bool)

    # Compute reprojection errors
    errors1 = compute_all_reprojection_errors(points_3d, P1, pts1)
    errors2 = compute_all_reprojection_errors(points_3d, P2, pts2)

    # Filter by reprojection error
    total_errors = errors1 + errors2
    valid_mask &= (total_errors < reproj_threshold)

    # Filter by depth (positive depth in both cameras)
    # For camera 1 (assumed to be world frame): depth is Z coordinate
    depths1 = points_3d[:, 2]
    valid_mask &= (depths1 > min_depth) & (depths1 < max_depth)

    # For camera 2, check depth using projection matrix
    # Project points and check if they're in front of camera (positive Z after projection)
    X_h = np.hstack([points_3d, np.ones((N, 1))])  # Homogeneous coordinates
    projected = (P2 @ X_h.T).T  # (N, 3)
    depths2 = projected[:, 2]  # Z coordinate in camera 2 frame (before perspective division)
    valid_mask &= (depths2 > 0)  # Must be in front of camera 2

    filtered_points = points_3d[valid_mask]

    return filtered_points, valid_mask


def get_camera_center(P: np.ndarray) -> np.ndarray:
    """
    Extract camera center from projection matrix P = K[R|t]

    Camera center C = -R^T * t

    Args:
        P: Projection matrix (3, 4)

    Returns:
        C: Camera center (3,)
    """
    # Decompose P = [M | p4] where M is 3x3
    M = P[:, :3]
    p4 = P[:, 3]

    # Camera center: C = -M^(-1) * p4
    try:
        C = -np.linalg.inv(M) @ p4
    except:
        # Singular matrix, return origin
        C = np.zeros(3)

    return C


def validate_reconstruction_quality(points_3d: np.ndarray, P1: np.ndarray, P2: np.ndarray,
                                    pts1: np.ndarray, pts2: np.ndarray) -> dict:
    """
    Compute quality statistics for reconstructed points

    Args:
        points_3d: 3D points (N, 3)
        P1: Camera matrix 1 (3, 4)
        P2: Camera matrix 2 (3, 4)
        pts1: Observed 2D points in image 1 (N, 2)
        pts2: Observed 2D points in image 2 (N, 2)

    Returns:
        Dictionary containing quality statistics
    """
    errors1 = compute_all_reprojection_errors(points_3d, P1, pts1)
    errors2 = compute_all_reprojection_errors(points_3d, P2, pts2)
    total_errors = errors1 + errors2

    depths = points_3d[:, 2]
    positive_depths = depths > 0

    stats = {
        'num_points': len(points_3d),
        'mean_reproj_error': total_errors.mean(),
        'max_reproj_error': total_errors.max(),
        'median_reproj_error': np.median(total_errors),
        'positive_depth_ratio': np.sum(positive_depths) / len(points_3d),
        'mean_depth': depths[positive_depths].mean() if np.any(positive_depths) else 0,
        'median_depth': np.median(depths[positive_depths]) if np.any(positive_depths) else 0,
    }

    return stats


if __name__ == "__main__":
    # Test triangulation validation
    import sys
    sys.path.insert(0, '.')

    from src.preprocessing.image_loader import load_images
    from src.preprocessing.camera_calibration import estimate_intrinsic_matrix
    from src.features.harris_detector import detect_harris_corners
    from src.features.descriptor import compute_descriptors
    from src.matching.matcher import match_descriptors, get_matched_points
    from src.matching.ransac import estimate_fundamental_matrix_ransac
    from src.geometry.essential_matrix import fundamental_to_essential
    from src.geometry.pose_recovery import decompose_essential_matrix
    from src.triangulation.triangulate import triangulate_points

    print("Testing Triangulation Validation")
    print("=" * 50)

    # Setup (same as triangulate.py test)
    images, metadata = load_images("data/scene1")
    width, height = metadata[0]['width'], metadata[0]['height']
    K = estimate_intrinsic_matrix(width, height)

    img1, img2 = images[0], images[1]
    corners1 = detect_harris_corners(img1)
    corners2 = detect_harris_corners(img2)
    desc1 = compute_descriptors(img1, corners1)
    desc2 = compute_descriptors(img2, corners2)
    matches, _ = match_descriptors(desc1, desc2)
    points1, points2 = get_matched_points(corners1, corners2, matches)

    F, inlier_mask = estimate_fundamental_matrix_ransac(points1, points2)
    E = fundamental_to_essential(F, K)
    poses = decompose_essential_matrix(E)
    R, t = poses[0]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    inlier_pts1 = points1[inlier_mask]
    inlier_pts2 = points2[inlier_mask]

    points_3d = triangulate_points(P1, P2, inlier_pts1, inlier_pts2)

    print(f"\nTriangulated {len(points_3d)} points")

    # Test validation
    print("\nComputing quality statistics...")
    stats = validate_reconstruction_quality(points_3d, P1, P2, inlier_pts1, inlier_pts2)

    print("\nReconstruction quality:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Filter points
    print("\nFiltering points...")
    filtered_points, valid_mask = filter_triangulated_points(
        points_3d, P1, P2, inlier_pts1, inlier_pts2
    )

    print(f"Filtered: {len(filtered_points)}/{len(points_3d)} points kept ({100*len(filtered_points)/len(points_3d):.1f}%)")

    # Quality of filtered points
    if len(filtered_points) > 0:
        filtered_pts1 = inlier_pts1[valid_mask]
        filtered_pts2 = inlier_pts2[valid_mask]

        filtered_stats = validate_reconstruction_quality(
            filtered_points, P1, P2, filtered_pts1, filtered_pts2
        )

        print("\nFiltered point quality:")
        for key, value in filtered_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
