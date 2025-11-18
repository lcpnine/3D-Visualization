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
                                      random_seed: int = None,
                                      confidence: float = 0.99,
                                      use_adaptive: bool = True,
                                      use_7point: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate fundamental matrix using RANSAC with OpenCV-style improvements

    Improvements over basic RANSAC:
    - 7-point algorithm for minimal sets (more efficient than 8-point)
    - Adaptive iteration count based on inlier ratio
    - Symmetric epipolar distance for error computation
    - Local optimization after finding best model

    Args:
        points1: Points from image 1 (N, 2)
        points2: Points from image 2 (N, 2)
        iterations: Maximum number of RANSAC iterations (default: from config)
        threshold: Inlier threshold in pixels (default: from config)
        random_seed: Random seed for reproducibility (default: from config)
        confidence: Desired confidence level (default: 0.99)
        use_adaptive: Use adaptive RANSAC (default: True)
        use_7point: Use 7-point algorithm instead of 8-point (default: True)

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
    min_points = 7 if use_7point else 8

    if N < min_points:
        raise ValueError(f"Need at least {min_points} points for fundamental matrix estimation, got {N}")

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    best_F = None
    best_inliers = None
    best_num_inliers = 0

    # Adaptive RANSAC: update max iterations based on inlier ratio
    max_iterations = iterations
    iterations_done = 0

    for iter in range(max_iterations):
        iterations_done = iter + 1

        # Randomly sample minimal set (7 or 8 points)
        sample_indices = np.random.choice(N, min_points, replace=False)
        sample_pts1 = points1[sample_indices]
        sample_pts2 = points2[sample_indices]

        # Estimate F from minimal set
        try:
            if use_7point:
                # 7-point algorithm returns 1 or 3 solutions
                F_candidates = seven_point_algorithm(sample_pts1, sample_pts2)
            else:
                F_candidates = [eight_point_algorithm(sample_pts1, sample_pts2)]
        except:
            continue

        # Test all candidate solutions (7-point can give up to 3 solutions)
        for F in F_candidates:
            if F is None:
                continue

            # Compute inliers using symmetric epipolar distance
            distances = compute_symmetric_epipolar_distance(F, points1, points2)
            inliers = distances < threshold
            num_inliers = np.sum(inliers)

            # Update best model
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers
                best_F = F

                # Adaptive RANSAC: update iteration count based on inlier ratio
                if use_adaptive and num_inliers > min_points:
                    inlier_ratio = num_inliers / N
                    # Avoid log(0) by clamping inlier ratio
                    inlier_ratio = max(min(inlier_ratio, 0.99), 0.01)

                    # Compute adaptive iteration count
                    # Formula: k = log(1-confidence) / log(1-inlier_ratio^s)
                    # where s is the sample size
                    denominator = np.log(1.0 - inlier_ratio ** min_points)
                    if denominator < -1e-10:  # Avoid division by zero
                        k_adaptive = int(np.log(1.0 - confidence) / denominator)
                        max_iterations = min(k_adaptive, iterations)

                if cfg.VERBOSE and (iter % 500 == 0 or num_inliers > 0.8 * N):
                    print(f"  RANSAC iter {iter}: {num_inliers}/{N} inliers ({100*num_inliers/N:.1f}%), max_iter={max_iterations}")

    if best_F is None:
        raise RuntimeError("RANSAC failed to find a valid fundamental matrix")

    if cfg.VERBOSE:
        print(f"  RANSAC completed in {iterations_done} iterations (max was {max_iterations})")

    # Refine F using all inliers (local optimization)
    inlier_pts1 = points1[best_inliers]
    inlier_pts2 = points2[best_inliers]

    F_refined = eight_point_algorithm(inlier_pts1, inlier_pts2)

    # Recompute inliers with refined F
    distances = compute_symmetric_epipolar_distance(F_refined, points1, points2)
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


def seven_point_algorithm(pts1: np.ndarray, pts2: np.ndarray) -> List[np.ndarray]:
    """
    Estimate fundamental matrix using the 7-point algorithm
    This is what OpenCV uses internally for RANSAC minimal sets

    The 7-point algorithm can return 1 or 3 valid solutions (solving a cubic equation)

    Args:
        pts1: Points from image 1 (7, 2)
        pts2: Points from image 2 (7, 2)

    Returns:
        List of fundamental matrices (1 to 3 solutions)
    """
    N = len(pts1)
    if N != 7:
        raise ValueError(f"7-point algorithm requires exactly 7 points, got {N}")

    # Normalize points (critical for numerical stability!)
    norm_pts1, T1 = normalize_homogeneous(pts1)
    norm_pts2, T2 = normalize_homogeneous(pts2)

    # Construct constraint matrix A
    A = np.zeros((7, 9))

    for i in range(7):
        x1, y1 = norm_pts1[i]
        x2, y2 = norm_pts2[i]

        A[i] = [
            x2 * x1, x2 * y1, x2,
            y2 * x1, y2 * y1, y2,
            x1, y1, 1
        ]

    # Solve for null space (2D) using SVD
    U, S, Vt = np.linalg.svd(A)

    # The null space is spanned by the last two rows of Vt
    f1 = Vt[-1, :].reshape(3, 3)
    f2 = Vt[-2, :].reshape(3, 3)

    # F = alpha * f1 + (1 - alpha) * f2 must have det(F) = 0
    # This gives a cubic equation in alpha:
    # det(alpha * f1 + (1-alpha) * f2) = 0

    # Expand: det(f2 + alpha * (f1 - f2)) = 0
    # Let F_diff = f1 - f2
    # det(f2 + alpha * F_diff) = cubic in alpha

    # Compute coefficients of the cubic polynomial
    # det(f2 + alpha * F_diff) = a0 + a1*alpha + a2*alpha^2 + a3*alpha^3
    F_diff = f1 - f2

    # Compute polynomial coefficients using determinant expansion
    # This is computationally intensive but necessary
    a0 = np.linalg.det(f2)
    a1 = (np.linalg.det(np.column_stack([F_diff[:, 0], f2[:, 1], f2[:, 2]])) +
          np.linalg.det(np.column_stack([f2[:, 0], F_diff[:, 1], f2[:, 2]])) +
          np.linalg.det(np.column_stack([f2[:, 0], f2[:, 1], F_diff[:, 2]])))
    a2 = (np.linalg.det(np.column_stack([F_diff[:, 0], F_diff[:, 1], f2[:, 2]])) +
          np.linalg.det(np.column_stack([F_diff[:, 0], f2[:, 1], F_diff[:, 2]])) +
          np.linalg.det(np.column_stack([f2[:, 0], F_diff[:, 1], F_diff[:, 2]])))
    a3 = np.linalg.det(F_diff)

    # Solve cubic equation
    coeffs = [a3, a2, a1, a0]
    roots = np.roots(coeffs)

    # Extract real roots (ignore complex roots with large imaginary parts)
    real_roots = []
    for root in roots:
        if np.abs(np.imag(root)) < 1e-6:
            real_roots.append(np.real(root))

    if len(real_roots) == 0:
        # No valid solution
        return []

    # Construct fundamental matrices from valid roots
    F_matrices = []
    for alpha in real_roots:
        F = alpha * f1 + (1 - alpha) * f2

        # Denormalize: F_denorm = T2^T * F_norm * T1
        F = T2.T @ F @ T1

        # Normalize so that ||F|| = 1
        F = F / np.linalg.norm(F)

        F_matrices.append(F)

    return F_matrices


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


def compute_symmetric_epipolar_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Compute symmetric epipolar distance (OpenCV's default for findFundamentalMat)

    This is more robust than Sampson distance for RANSAC as it measures
    the maximum of distances in both directions

    The symmetric epipolar distance is:
    max(d(p2, F*p1), d(p1, F^T*p2))

    where d(p, l) is the distance from point p to line l

    Args:
        F: Fundamental matrix (3, 3)
        pts1: Points from image 1 (N, 2)
        pts2: Points from image 2 (N, 2)

    Returns:
        distances: Symmetric epipolar distances (N,)
    """
    # Convert to homogeneous coordinates
    pts1_h = np.column_stack([pts1, np.ones(len(pts1))])  # (N, 3)
    pts2_h = np.column_stack([pts2, np.ones(len(pts2))])  # (N, 3)

    # Compute epipolar lines
    # l2 = F * p1 (epipolar line in image 2)
    # l1 = F^T * p2 (epipolar line in image 1)
    l2 = (F @ pts1_h.T).T  # (N, 3) - line in image 2
    l1 = (F.T @ pts2_h.T).T  # (N, 3) - line in image 1

    # Distance from point to line: |ax + by + c| / sqrt(a^2 + b^2)
    # For p2 to l2: |l2[0]*p2[0] + l2[1]*p2[1] + l2[2]| / sqrt(l2[0]^2 + l2[1]^2)
    dist_p2_to_l2 = np.abs(np.sum(pts2_h * l2, axis=1)) / np.sqrt(l2[:, 0]**2 + l2[:, 1]**2 + 1e-10)

    # For p1 to l1
    dist_p1_to_l1 = np.abs(np.sum(pts1_h * l1, axis=1)) / np.sqrt(l1[:, 0]**2 + l1[:, 1]**2 + 1e-10)

    # Symmetric distance: maximum of both
    distances = np.maximum(dist_p2_to_l2, dist_p1_to_l1)

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
