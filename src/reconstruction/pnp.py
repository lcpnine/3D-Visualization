"""
Phase 6 - Step 6.2: PnP (Perspective-n-Point) Solver
Estimates camera pose from 2D-3D correspondences using RANSAC-based PnP.
"""

import numpy as np
from typing import Tuple
import sys
sys.path.insert(0, '.')
from config import cfg
from utils.math_utils import svd_solve, enforce_orthogonal


def solve_pnp_ransac(points_3d: np.ndarray, points_2d: np.ndarray, K: np.ndarray,
                     iterations: int = None, threshold: float = None,
                     random_seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve PnP using RANSAC

    Args:
        points_3d: 3D points in world coordinates (N, 3)
        points_2d: Corresponding 2D points in image (N, 2)
        K: Intrinsic matrix (3, 3)
        iterations: Number of RANSAC iterations (default: from config)
        threshold: Inlier threshold in pixels (default: from config)
        random_seed: Random seed for reproducibility

    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
        inlier_mask: Boolean mask of inliers (N,)
    """
    if iterations is None:
        iterations = cfg.PNP_RANSAC_ITERATIONS
    if threshold is None:
        threshold = cfg.PNP_RANSAC_THRESHOLD
    if random_seed is None:
        random_seed = cfg.RANDOM_SEED

    N = len(points_3d)
    if N < 6:
        raise ValueError(f"Need at least 6 points for PnP, got {N}")

    np.random.seed(random_seed)

    best_R = None
    best_t = None
    best_inliers = None
    best_num_inliers = 0

    for iter in range(iterations):
        # Randomly sample 6 points
        sample_indices = np.random.choice(N, 6, replace=False)
        sample_3d = points_3d[sample_indices]
        sample_2d = points_2d[sample_indices]

        # Solve PnP with 6 points
        try:
            R, t = dlt_pnp(sample_3d, sample_2d, K)
        except:
            continue

        # Count inliers
        errors = compute_reprojection_errors(points_3d, points_2d, K, R, t)
        inliers = errors < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inliers
            best_R = R
            best_t = t

            if cfg.VERBOSE and (iter % 200 == 0 or num_inliers > 0.8 * N):
                print(f"  PnP RANSAC iter {iter}: {num_inliers}/{N} inliers ({100*num_inliers/N:.1f}%)")

    if best_R is None:
        raise RuntimeError("PnP RANSAC failed to find a valid pose")

    # Refine using all inliers
    inlier_3d = points_3d[best_inliers]
    inlier_2d = points_2d[best_inliers]

    try:
        R_refined, t_refined = dlt_pnp(inlier_3d, inlier_2d, K)

        # Recompute inliers
        errors = compute_reprojection_errors(points_3d, points_2d, K, R_refined, t_refined)
        final_inliers = errors < threshold

        if cfg.VERBOSE:
            print(f"  Final PnP: {np.sum(final_inliers)}/{N} inliers ({100*np.sum(final_inliers)/N:.1f}%)")

        return R_refined, t_refined, final_inliers
    except:
        return best_R, best_t, best_inliers


def dlt_pnp(points_3d: np.ndarray, points_2d: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve PnP using Direct Linear Transform

    Args:
        points_3d: 3D points in world coordinates (N, 3) where N >= 6
        points_2d: Corresponding 2D points (N, 2)
        K: Intrinsic matrix (3, 3)

    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    """
    N = len(points_3d)
    if N < 6:
        raise ValueError(f"Need at least 6 points, got {N}")

    # Normalize 2D points using inverse of K
    K_inv = np.linalg.inv(K)
    points_2d_h = np.column_stack([points_2d, np.ones(N)])
    points_2d_normalized = (K_inv @ points_2d_h.T).T  # (N, 3)

    # Construct DLT matrix
    A = []
    for i in range(N):
        X, Y, Z = points_3d[i]
        u, v, w = points_2d_normalized[i]

        # Two equations per point
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    A = np.array(A)

    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1, :].reshape(3, 4)

    # Extract R and t from P = [R|t]
    R = P[:, :3]
    t = P[:, 3]

    # Project R to valid rotation matrix
    R = project_to_rotation(R)

    return R, t


def project_to_rotation(R_approx: np.ndarray) -> np.ndarray:
    """
    Project approximate rotation matrix to nearest valid rotation matrix

    Uses SVD: R = U * V^T

    Args:
        R_approx: Approximate rotation matrix (3, 3)

    Returns:
        R: Valid rotation matrix (3, 3)
    """
    U, S, Vt = np.linalg.svd(R_approx)
    R = U @ Vt

    # Ensure det(R) = +1 (proper rotation, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    return R


def compute_reprojection_errors(points_3d: np.ndarray, points_2d: np.ndarray,
                                K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute reprojection errors for all points

    Args:
        points_3d: 3D points (N, 3)
        points_2d: 2D points (N, 2)
        K: Intrinsic matrix (3, 3)
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)

    Returns:
        errors: Reprojection errors in pixels (N,)
    """
    N = len(points_3d)

    # Project 3D points
    P = K @ np.hstack([R, t.reshape(3, 1)])

    points_3d_h = np.column_stack([points_3d, np.ones(N)])
    projected_h = (P @ points_3d_h.T).T  # (N, 3)

    # Convert to 2D
    projected_2d = projected_h[:, :2] / projected_h[:, 2:3]

    # Compute errors
    errors = np.linalg.norm(projected_2d - points_2d, axis=1)

    return errors


if __name__ == "__main__":
    # Test PnP solver
    print("Testing PnP Solver")
    print("=" * 50)

    # Generate synthetic data for testing
    print("\nGenerating synthetic test data...")

    # True camera pose
    from utils.math_utils import axis_angle_to_rotation_matrix
    axis_angle = np.array([0.1, 0.2, 0.1])
    R_true = axis_angle_to_rotation_matrix(axis_angle)
    t_true = np.array([1.0, 0.5, 0.2])

    # Intrinsic matrix (simplified)
    K = np.array([
        [1000, 0, 500],
        [0, 1000, 400],
        [0, 0, 1]
    ], dtype=np.float64)

    # Generate 3D points
    N = 50
    points_3d = np.random.randn(N, 3) * 2
    points_3d[:, 2] += 10  # Move points in front of camera

    # Project to get 2D points
    P_true = K @ np.hstack([R_true, t_true.reshape(3, 1)])
    points_3d_h = np.column_stack([points_3d, np.ones(N)])
    projected_h = (P_true @ points_3d_h.T).T
    points_2d = projected_h[:, :2] / projected_h[:, 2:3]

    # Add some noise
    points_2d += np.random.randn(N, 2) * 0.5

    print(f"Generated {N} 3D points")
    print(f"True rotation R:")
    print(R_true)
    print(f"True translation t: {t_true}")

    # Solve PnP
    print("\nSolving PnP with RANSAC...")
    R_est, t_est, inliers = solve_pnp_ransac(points_3d, points_2d, K)

    print(f"\nEstimated rotation R:")
    print(R_est)
    print(f"Estimated translation t: {t_est}")

    print(f"\nInliers: {np.sum(inliers)}/{N}")

    # Compare with ground truth
    print(f"\nComparison with ground truth:")
    print(f"Rotation error (Frobenius norm): {np.linalg.norm(R_est - R_true):.6f}")
    print(f"Translation error (L2 norm): {np.linalg.norm(t_est - t_true):.6f}")

    # Compute final reprojection errors
    errors = compute_reprojection_errors(points_3d[inliers], points_2d[inliers], K, R_est, t_est)
    print(f"\nReprojection errors on inliers:")
    print(f"  Mean: {errors.mean():.4f} pixels")
    print(f"  Max: {errors.max():.4f} pixels")
