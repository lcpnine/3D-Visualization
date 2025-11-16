"""
Phase 4 - Step 4.2: Camera Pose Recovery
Extracts rotation and translation from the essential matrix using chirality check.
"""

import numpy as np
from typing import List, Tuple
import sys
sys.path.insert(0, '.')
from utils.math_utils import is_rotation_matrix


def decompose_essential_matrix(E: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Decompose essential matrix into 4 possible (R, t) combinations

    Args:
        E: Essential matrix (3, 3)

    Returns:
        List of 4 (R, t) tuples
    """
    # SVD decomposition
    U, S, Vt = np.linalg.svd(E)

    # Ensure proper rotation (det(U*Vt) = 1)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1

    # W matrix for rotation extraction
    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float64)

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Ensure determinant is +1 (proper rotation)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Translation (up to scale, from last column of U)
    t = U[:, 2]

    # Four possible combinations: (R1, t), (R1, -t), (R2, t), (R2, -t)
    poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    return poses


def select_valid_pose(poses: List[Tuple[np.ndarray, np.ndarray]],
                     points1: np.ndarray, points2: np.ndarray,
                     K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select the correct pose from 4 candidates using chirality check

    The correct pose is the one where most points are in front of both cameras

    Args:
        poses: List of 4 (R, t) candidates
        points1: Points from image 1 (N, 2)
        points2: Points from image 2 (N, 2)
        K: Intrinsic matrix (3, 3)

    Returns:
        R: Selected rotation matrix (3, 3)
        t: Selected translation vector (3,)
    """
    best_pose = None
    best_count = 0

    # First camera at origin
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    for R, t in poses:
        # Second camera
        P2 = K @ np.hstack([R, t.reshape(3, 1)])

        # Count points with positive depth in both cameras
        count = check_chirality(R, t, points1, points2, K)

        if count > best_count:
            best_count = count
            best_pose = (R, t)

    if best_pose is None:
        raise RuntimeError("No valid pose found (all chirality checks failed)")

    R, t = best_pose

    # Check if majority of points are in front
    total_points = len(points1)
    if best_count < 0.75 * total_points:
        print(f"Warning: Only {best_count}/{total_points} ({100*best_count/total_points:.1f}%) "
              f"points have positive depth. This may indicate poor pose estimation.")

    return R, t


def check_chirality(R: np.ndarray, t: np.ndarray,
                   points1: np.ndarray, points2: np.ndarray,
                   K: np.ndarray) -> int:
    """
    Count number of points with positive depth in both cameras (chirality check)

    Args:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        points1: Points from image 1 (N, 2)
        points2: Points from image 2 (N, 2)
        K: Intrinsic matrix (3, 3)

    Returns:
        Number of points with positive depth in both cameras
    """
    # Import here to avoid circular dependency
    from src.triangulation.triangulate import triangulate_points

    # Camera matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    # Triangulate points
    try:
        points_3d = triangulate_points(P1, P2, points1, points2)
    except:
        return 0

    if len(points_3d) == 0:
        return 0

    # Check depth in camera 1 (z > 0)
    depths1 = points_3d[:, 2]
    valid1 = depths1 > 0

    # Transform to camera 2 frame
    points_3d_h = np.column_stack([points_3d, np.ones(len(points_3d))])
    T2 = np.eye(4)
    T2[:3, :3] = R
    T2[:3, 3] = t

    points_3d_cam2 = (T2 @ points_3d_h.T).T
    depths2 = points_3d_cam2[:, 2]
    valid2 = depths2 > 0

    # Count points valid in both cameras
    count = np.sum(valid1 & valid2)

    return count


def normalize_translation(t: np.ndarray) -> np.ndarray:
    """
    Normalize translation to unit vector

    Args:
        t: Translation vector (3,)

    Returns:
        Normalized translation vector
    """
    norm = np.linalg.norm(t)
    if norm < 1e-10:
        raise ValueError("Translation vector is too small to normalize")
    return t / norm


if __name__ == "__main__":
    # Test pose recovery
    from src.preprocessing.image_loader import load_images
    from src.preprocessing.camera_calibration import estimate_intrinsic_matrix
    from src.features.harris_detector import detect_harris_corners
    from src.features.descriptor import compute_descriptors
    from src.matching.matcher import match_descriptors, get_matched_points
    from src.matching.ransac import estimate_fundamental_matrix_ransac
    from src.geometry.essential_matrix import fundamental_to_essential

    print("Testing Camera Pose Recovery")
    print("=" * 50)

    # Load images and compute intrinsics
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

    # Estimate F and E
    print("\nEstimating fundamental and essential matrices...")
    F, inlier_mask = estimate_fundamental_matrix_ransac(points1, points2)
    E = fundamental_to_essential(F, K)

    # Use only inliers
    inlier_pts1 = points1[inlier_mask]
    inlier_pts2 = points2[inlier_mask]
    print(f"Using {len(inlier_pts1)} inlier points")

    # Decompose essential matrix
    print("\nDecomposing essential matrix...")
    poses = decompose_essential_matrix(E)
    print(f"Generated {len(poses)} pose candidates")

    # Test each pose
    print("\nTesting each pose:")
    for i, (R, t) in enumerate(poses):
        count = check_chirality(R, t, inlier_pts1, inlier_pts2, K)
        print(f"  Pose {i+1}: {count}/{len(inlier_pts1)} points in front ({100*count/len(inlier_pts1):.1f}%)")
        print(f"    det(R) = {np.linalg.det(R):.6f}")
        print(f"    ||t|| = {np.linalg.norm(t):.6f}")

    # Select best pose
    print("\nSelecting best pose...")
    R, t = select_valid_pose(poses, inlier_pts1, inlier_pts2, K)

    print(f"\nSelected camera pose:")
    print(f"Rotation R:")
    print(R)
    print(f"\nTranslation t:")
    print(t)

    # Validate rotation
    print(f"\nValidation:")
    print(f"  Is valid rotation matrix: {is_rotation_matrix(R)}")
    print(f"  det(R) = {np.linalg.det(R):.10f} (should be 1)")
    print(f"  R^T @ R =")
    print(R.T @ R)
    print(f"  (should be identity)")

    # Normalize translation
    t_norm = normalize_translation(t)
    print(f"\nNormalized translation:")
    print(f"  ||t|| = {np.linalg.norm(t_norm):.6f} (should be 1)")
