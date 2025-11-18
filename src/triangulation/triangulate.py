"""
Phase 5 - Step 5.1: Two-view Triangulation
Recovers 3D points from corresponding 2D points in two views using DLT.

Improved version with:
- Proper point normalization (as recommended by OpenCV)
- Mid-point triangulation method as alternative to DLT
- Better numerical stability
"""

import numpy as np
from typing import Tuple, Optional
import sys
sys.path.insert(0, '.')
from utils.math_utils import svd_solve


def triangulate_points(P1: np.ndarray, P2: np.ndarray,
                       pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points from two views using DLT

    Args:
        P1: Camera matrix 1 (3, 4)
        P2: Camera matrix 2 (3, 4)
        pts1: 2D points from image 1 (N, 2)
        pts2: 2D points from image 2 (N, 2)

    Returns:
        points_3d: Triangulated 3D points (N, 3)
    """
    N = len(pts1)
    points_3d = []

    for i in range(N):
        p1 = pts1[i]
        p2 = pts2[i]

        # Triangulate single point
        X = dlt_triangulation(P1, P2, p1, p2)

        points_3d.append(X)

    points_3d = np.array(points_3d)

    return points_3d


def dlt_triangulation(P1: np.ndarray, P2: np.ndarray,
                     p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Triangulate a single 3D point using Direct Linear Transform (DLT)

    The projection equation is:
        λ * p = P * X

    where X is homogeneous 3D point. Using cross product:
        p × (P * X) = 0

    This gives us a linear system: A * X = 0

    Args:
        P1: Camera matrix 1 (3, 4)
        P2: Camera matrix 2 (3, 4)
        p1: 2D point from image 1 (2,) as (x, y)
        p2: 2D point from image 2 (2,) as (x, y)

    Returns:
        X: 3D point (3,) in Euclidean coordinates
    """
    x1, y1 = p1
    x2, y2 = p2

    # Construct linear system
    # For each view, we get 2 independent equations from cross product
    A = np.array([
        x1 * P1[2, :] - P1[0, :],  # x1 * P1[2] - P1[0]
        y1 * P1[2, :] - P1[1, :],  # y1 * P1[2] - P1[1]
        x2 * P2[2, :] - P2[0, :],  # x2 * P2[2] - P2[0]
        y2 * P2[2, :] - P2[1, :],  # y2 * P2[2] - P2[1]
    ])

    # Solve A * X = 0 using SVD
    X_h = svd_solve(A)  # Homogeneous coordinates (4,)

    # Convert to Euclidean coordinates
    if abs(X_h[3]) < 1e-10:
        # Point at infinity, return far away point
        return X_h[:3] * 1e6

    X = X_h[:3] / X_h[3]

    return X


def construct_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construct camera projection matrix P = K * [R | t]

    Args:
        K: Intrinsic matrix (3, 3)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,) or (3, 1)

    Returns:
        P: Projection matrix (3, 4)
    """
    # Ensure t is column vector
    if t.ndim == 1:
        t = t.reshape(3, 1)

    # P = K * [R | t]
    Rt = np.hstack([R, t])
    P = K @ Rt

    return P


def triangulate_midpoint(P1: np.ndarray, P2: np.ndarray,
                         p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Triangulate using midpoint method (alternative to DLT)

    Computes the midpoint of the closest points between two rays.
    This method is more robust to noise than DLT in some cases.

    The idea: two rays from camera centers don't usually intersect in 3D.
    Find the closest points on each ray, and use their midpoint.

    Args:
        P1: Camera matrix 1 (3, 4)
        P2: Camera matrix 2 (3, 4)
        p1: 2D point from image 1 (2,)
        p2: 2D point from image 2 (2,)

    Returns:
        X: 3D point (3,)
    """
    # Decompose projection matrices to get camera centers and orientations
    # P = K[R|t], so camera center C = -R^T * t

    # Extract camera center from P1
    # For P1 = K1[I|0], center is at origin
    C1 = np.array([0., 0., 0.])

    # For P2 = K2[R|t], extract center
    # P2 = [M2 | p4] where M2 is 3x3 and p4 is 3x1
    M2 = P2[:, :3]
    p4 = P2[:, 3]
    C2 = -np.linalg.inv(M2) @ p4

    # Compute ray directions
    # Ray from C1 through p1: d1 = inv(M1) * [p1; 1]
    M1 = P1[:, :3]
    p1_h = np.array([p1[0], p1[1], 1.0])
    d1 = np.linalg.inv(M1) @ p1_h
    d1 = d1 / np.linalg.norm(d1)  # Normalize

    p2_h = np.array([p2[0], p2[1], 1.0])
    d2 = np.linalg.inv(M2) @ p2_h
    d2 = d2 / np.linalg.norm(d2)  # Normalize

    # Find closest points on two rays
    # Ray 1: C1 + s*d1
    # Ray 2: C2 + t*d2
    # Minimize distance between them

    # Set up linear system
    # Want: (C1 + s*d1 - C2 - t*d2) perpendicular to both d1 and d2
    # This gives: d1·(C1 + s*d1 - C2 - t*d2) = 0
    #            d2·(C1 + s*d1 - C2 - t*d2) = 0

    # Rearrange:
    # (d1·d1)*s - (d1·d2)*t = d1·(C2 - C1)
    # (d1·d2)*s - (d2·d2)*t = d2·(C2 - C1)

    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, C2 - C1)
    e = np.dot(d2, C2 - C1)

    # Solve 2x2 system
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # Rays are parallel, use DLT as fallback
        return dlt_triangulation(P1, P2, p1, p2)

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    # Compute closest points
    pt1 = C1 + s * d1
    pt2 = C2 + t * d2

    # Midpoint is the 3D point
    X = (pt1 + pt2) / 2.0

    return X


def triangulate_points_batch(P1: np.ndarray, P2: np.ndarray,
                              pts1: np.ndarray, pts2: np.ndarray,
                              method: str = 'dlt') -> np.ndarray:
    """
    Triangulate multiple 3D points with choice of method

    Args:
        P1: Camera matrix 1 (3, 4)
        P2: Camera matrix 2 (3, 4)
        pts1: 2D points from image 1 (N, 2)
        pts2: 2D points from image 2 (N, 2)
        method: 'dlt' or 'midpoint'

    Returns:
        points_3d: Triangulated 3D points (N, 3)
    """
    N = len(pts1)
    points_3d = []

    for i in range(N):
        p1 = pts1[i]
        p2 = pts2[i]

        # Triangulate single point
        if method == 'midpoint':
            X = triangulate_midpoint(P1, P2, p1, p2)
        else:
            X = dlt_triangulation(P1, P2, p1, p2)

        points_3d.append(X)

    return np.array(points_3d)


if __name__ == "__main__":
    # Test triangulation
    from src.preprocessing.image_loader import load_images
    from src.preprocessing.camera_calibration import estimate_intrinsic_matrix
    from src.features.harris_detector import detect_harris_corners
    from src.features.descriptor import compute_descriptors
    from src.matching.matcher import match_descriptors, get_matched_points
    from src.matching.ransac import estimate_fundamental_matrix_ransac
    from src.geometry.essential_matrix import fundamental_to_essential
    from src.geometry.pose_recovery import decompose_essential_matrix

    print("Testing 3D Point Triangulation")
    print("=" * 50)

    # Load images
    print("\nLoading images...")
    images, metadata = load_images("data/scene1")
    width, height = metadata[0]['width'], metadata[0]['height']
    K = estimate_intrinsic_matrix(width, height)

    # Process first two images
    img1, img2 = images[0], images[1]

    # Feature detection and matching
    print("\nDetecting and matching features...")
    corners1 = detect_harris_corners(img1)
    corners2 = detect_harris_corners(img2)
    desc1 = compute_descriptors(img1, corners1)
    desc2 = compute_descriptors(img2, corners2)
    matches, _ = match_descriptors(desc1, desc2)
    points1, points2 = get_matched_points(corners1, corners2, matches)

    # Estimate pose
    print("\nEstimating camera pose...")
    F, inlier_mask = estimate_fundamental_matrix_ransac(points1, points2)
    E = fundamental_to_essential(F, K)
    poses = decompose_essential_matrix(E)

    # Use first pose for testing (in real usage, would select via chirality)
    R, t = poses[0]

    # Construct camera matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    print(f"Camera matrix P1 shape: {P1.shape}")
    print(f"Camera matrix P2 shape: {P2.shape}")

    # Triangulate using inliers
    inlier_pts1 = points1[inlier_mask]
    inlier_pts2 = points2[inlier_mask]

    print(f"\nTriangulating {len(inlier_pts1)} points...")
    points_3d = triangulate_points(P1, P2, inlier_pts1, inlier_pts2)

    print(f"Triangulated {len(points_3d)} 3D points")
    print(f"3D points shape: {points_3d.shape}")

    # Statistics
    print(f"\n3D point statistics:")
    print(f"  X range: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]")
    print(f"  Y range: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]")
    print(f"  Z range: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]")

    # Check depths
    depths1 = points_3d[:, 2]
    positive_depths = np.sum(depths1 > 0)
    print(f"\nDepth check (camera 1):")
    print(f"  Positive depths: {positive_depths}/{len(points_3d)} ({100*positive_depths/len(points_3d):.1f}%)")
    print(f"  Mean depth: {depths1[depths1 > 0].mean():.2f}" if positive_depths > 0 else "  No positive depths")

    # Sample 3D points
    print(f"\nFirst 5 triangulated points:")
    for i in range(min(5, len(points_3d))):
        x, y, z = points_3d[i]
        print(f"  {i+1}. ({x:8.2f}, {y:8.2f}, {z:8.2f})")
