"""
Phase 3 - Step 3.2: RANSAC and Fundamental Matrix Estimation
Removes outlier matches and estimates the fundamental matrix.
"""

import numpy as np
from typing import Tuple
from config import cfg
import sys
sys.path.insert(0, '.')
from utils.math_utils import svd_solve, normalize_homogeneous


def estimate_fundamental_matrix_ransac(points1: np.ndarray, points2: np.ndarray,
                                      iterations: int = None,
                                      threshold: float = None,
                                      random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate fundamental matrix using RANSAC

    Args:
        points1: Points from image 1 (N, 2)
        points2: Points from image 2 (N, 2)
        iterations: Number of RANSAC iterations (default: from config)
        threshold: Inlier threshold in pixels (default: from config)
        random_seed: Random seed for reproducibility (default: from config)

    Returns:
        F: Fundamental matrix (3, 3)
        inlier_mask: Boolean mask of inliers (N,)
    """
    if iterations is None:
        iterations = cfg.RANSAC_ITERATIONS
    if threshold is None:
        threshold = cfg.RANSAC_THRESHOLD
    if random_seed is None:
        random_seed = cfg.RANDOM_SEED

    N = len(points1)
    if N < 8:
        raise ValueError(f"Need at least 8 points for fundamental matrix estimation, got {N}")

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    best_F = None
    best_inliers = None
    best_num_inliers = 0

    for iter in range(iterations):
        # Randomly sample 8 points
        sample_indices = np.random.choice(N, 8, replace=False)
        sample_pts1 = points1[sample_indices]
        sample_pts2 = points2[sample_indices]

        # Estimate F from 8 points
        try:
            F = eight_point_algorithm(sample_pts1, sample_pts2)
        except:
            continue

        # Compute inliers
        distances = compute_sampson_distance(F, points1, points2)
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        # Update best model
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_F = F

            if cfg.VERBOSE and (iter % 500 == 0 or num_inliers > 0.8 * N):
                print(f"  RANSAC iter {iter}: {num_inliers}/{N} inliers ({100*num_inliers/N:.1f}%)")

    if best_F is None:
        raise RuntimeError("RANSAC failed to find a valid fundamental matrix")

    # Refine F using all inliers
    inlier_pts1 = points1[best_inliers]
    inlier_pts2 = points2[best_inliers]

    F_refined = eight_point_algorithm(inlier_pts1, inlier_pts2)

    # Recompute inliers with refined F
    distances = compute_sampson_distance(F_refined, points1, points2)
    final_inliers = distances < threshold

    if cfg.VERBOSE:
        print(f"  Final: {np.sum(final_inliers)}/{N} inliers ({100*np.sum(final_inliers)/N:.1f}%)")

    return F_refined, final_inliers


def eight_point_algorithm(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Estimate fundamental matrix using the 8-point algorithm with normalization

    Args:
        pts1: Points from image 1 (N, 2) where N >= 8
        pts2: Points from image 2 (N, 2)

    Returns:
        F: Fundamental matrix (3, 3)
    """
    N = len(pts1)
    if N < 8:
        raise ValueError(f"Need at least 8 points, got {N}")

    # Normalize points (critical for numerical stability!)
    norm_pts1, T1 = normalize_homogeneous(pts1)
    norm_pts2, T2 = normalize_homogeneous(pts2)

    # Construct constraint matrix A
    # Each point pair contributes one row: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    A = np.zeros((N, 9))

    for i in range(N):
        x1, y1 = norm_pts1[i]
        x2, y2 = norm_pts2[i]

        A[i] = [
            x2 * x1, x2 * y1, x2,
            y2 * x1, y2 * y1, y2,
            x1, y1, 1
        ]

    # Solve Af = 0 using SVD
    f = svd_solve(A)

    # Reshape to 3x3 matrix
    F = f.reshape(3, 3)

    # Enforce rank-2 constraint
    F = enforce_rank2_constraint(F)

    # Denormalize: F_denorm = T2^T * F_norm * T1
    F = T2.T @ F @ T1

    # Normalize so that ||F|| = 1
    F = F / np.linalg.norm(F)

    return F


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize 2D points for improved numerical stability (Hartley normalization)

    Shift centroid to origin and scale so average distance to origin is sqrt(2)

    Args:
        points: 2D points (N, 2)

    Returns:
        normalized_points: Normalized points (N, 2)
        T: 3x3 transformation matrix
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Shift to origin
    shifted = points - centroid

    # Compute average distance to origin
    avg_dist = np.mean(np.linalg.norm(shifted, axis=1))

    if avg_dist < 1e-10:
        # Points are too close, return as-is
        return points, np.eye(3)

    # Scale factor
    scale = np.sqrt(2) / avg_dist

    # Normalized points
    normalized = shifted * scale

    # Transformation matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    return normalized, T


def enforce_rank2_constraint(F: np.ndarray) -> np.ndarray:
    """
    Enforce rank-2 constraint on fundamental matrix

    Set the smallest singular value to zero

    Args:
        F: Fundamental matrix (3, 3)

    Returns:
        F_rank2: Rank-2 fundamental matrix (3, 3)
    """
    U, S, Vt = np.linalg.svd(F)

    # Set smallest singular value to zero
    S[2] = 0

    # Reconstruct F
    F_rank2 = U @ np.diag(S) @ Vt

    return F_rank2


def compute_sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Compute Sampson distance (first-order geometric error)

    Sampson distance is an approximation to the geometric error and is more
    accurate than the algebraic error (p2^T * F * p1)

    Args:
        F: Fundamental matrix (3, 3)
        pts1: Points from image 1 (N, 2)
        pts2: Points from image 2 (N, 2)

    Returns:
        distances: Sampson distances (N,)
    """
    # Convert to homogeneous coordinates
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])  # (N, 3)
    pts2_h = np.column_stack([pts2, np.ones(len(pts2))])  # (N, 3)

    # Compute Fp1 and F^T p2
    Fp1 = (F @ pts1_h.T).T  # (N, 3)
    FTp2 = (F.T @ pts2_h.T).T  # (N, 3)

    # Algebraic error: p2^T * F * p1
    algebraic_error = np.sum(pts2_h * Fp1, axis=1)  # (N,)

    # Sampson distance denominator
    denom = (
        Fp1[:, 0] ** 2 +
        Fp1[:, 1] ** 2 +
        FTp2[:, 0] ** 2 +
        FTp2[:, 1] ** 2
    )

    # Sampson distance
    distances = np.abs(algebraic_error) / np.sqrt(denom + 1e-10)

    return distances


def compute_epipolar_error(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Compute algebraic epipolar error: |p2^T * F * p1|

    Args:
        F: Fundamental matrix (3, 3)
        pts1: Points from image 1 (N, 2)
        pts2: Points from image 2 (N, 2)

    Returns:
        errors: Algebraic errors (N,)
    """
    # Convert to homogeneous coordinates
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])
    pts2_h = np.column_stack([pts2, np.ones(len(pts2))])

    # Compute p2^T * F * p1
    errors = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))

    return errors


if __name__ == "__main__":
    # Test RANSAC and fundamental matrix estimation
    from src.preprocessing.image_loader import load_images
    from src.features.harris_detector import detect_harris_corners
    from src.features.descriptor import compute_descriptors
    from src.matching.matcher import match_descriptors, get_matched_points

    print("Testing RANSAC and Fundamental Matrix Estimation")
    print("=" * 50)

    # Load images
    print("\nLoading images...")
    images, metadata = load_images("data/scene1")

    # Process first two images
    img1, img2 = images[0], images[1]
    print(f"Image 1: {metadata[0]['filename']}")
    print(f"Image 2: {metadata[1]['filename']}")

    # Detect and match features
    print("\nDetecting and matching features...")
    corners1 = detect_harris_corners(img1)
    corners2 = detect_harris_corners(img2)
    desc1 = compute_descriptors(img1, corners1)
    desc2 = compute_descriptors(img2, corners2)
    matches, _ = match_descriptors(desc1, desc2)

    print(f"Initial matches: {len(matches)}")

    # Get matched points
    points1, points2 = get_matched_points(corners1, corners2, matches)

    # Estimate fundamental matrix with RANSAC
    print("\nEstimating fundamental matrix with RANSAC...")
    F, inlier_mask = estimate_fundamental_matrix_ransac(points1, points2)

    print(f"\nFundamental matrix F:")
    print(F)

    print(f"\nInliers: {np.sum(inlier_mask)}/{len(points1)} ({100*np.sum(inlier_mask)/len(points1):.1f}%)")

    # Compute errors
    inlier_pts1 = points1[inlier_mask]
    inlier_pts2 = points2[inlier_mask]

    sampson_dist = compute_sampson_distance(F, inlier_pts1, inlier_pts2)
    epipolar_err = compute_epipolar_error(F, inlier_pts1, inlier_pts2)

    print(f"\nInlier error statistics:")
    print(f"  Sampson distance - mean: {sampson_dist.mean():.4f}, max: {sampson_dist.max():.4f}")
    print(f"  Epipolar error - mean: {epipolar_err.mean():.6f}, max: {epipolar_err.max():.6f}")

    # Check rank
    rank = np.linalg.matrix_rank(F)
    print(f"\nFundamental matrix rank: {rank} (should be 2)")

    # Check det(F) = 0
    det_F = np.linalg.det(F)
    print(f"det(F) = {det_F:.10f} (should be ≈ 0)")
