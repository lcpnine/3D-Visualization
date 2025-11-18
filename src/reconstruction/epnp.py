"""
EPnP (Efficient Perspective-n-Point) Algorithm
Based on the paper: "EPnP: An Accurate O(n) Solution to the PnP Problem"
by Lepetit, Moreno-Noguer, and Fua (IJCV 2009)

This is what OpenCV uses internally for PnP estimation.
"""

import numpy as np
from typing import Tuple


def solve_epnp(points_3d: np.ndarray, points_2d: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve PnP using EPnP algorithm

    EPnP expresses the n 3D points as a weighted sum of four virtual control points.
    The problem then reduces to estimating the coordinates of these control points
    in the camera coordinate system, which is done efficiently.

    Args:
        points_3d: 3D points in world coordinates (N, 3) where N >= 4
        points_2d: Corresponding 2D points (N, 2)
        K: Intrinsic matrix (3, 3)

    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    """
    N = len(points_3d)
    if N < 4:
        raise ValueError(f"EPnP requires at least 4 points, got {N}")

    # Step 1: Choose control points in world coordinate system
    # Use centroid + 3 PCA directions
    cws = choose_control_points(points_3d)

    # Step 2: Compute barycentric coordinates
    # Express each 3D point as: X_i = sum(alpha_ij * C_j) for j=0..3
    alphas = compute_barycentric_coordinates(points_3d, cws)

    # Step 3: Compute M matrix (2N x 12)
    # This relates the control points in camera coords to image measurements
    M = compute_M_matrix(alphas, points_2d, K)

    # Step 4: Find control points in camera coordinate system
    # Solve using SVD - the solution lies in the null space of M
    _, _, Vt = np.linalg.svd(M)

    # The null space is spanned by the last few rows of Vt
    # We can have 1, 2, 3, or 4 solutions (betas)
    # For simplicity, we use the last row (rank-deficient case)

    # Get null space vectors
    null_vectors = Vt[-4:, :]  # Last 4 rows

    # For simplicity, use case N=1 (single solution)
    # In full EPnP, you would solve for betas using Gauss-Newton
    ccs = null_vectors[-1, :].reshape(4, 3)

    # Step 5: Estimate R and t from control points
    # We have: ccs = R * cws + t
    # This is an absolute orientation problem (Procrustes)
    R, t = compute_pose_from_control_points(cws, ccs)

    return R, t


def choose_control_points(points_3d: np.ndarray) -> np.ndarray:
    """
    Choose 4 control points in world coordinate system

    Control points:
    - C0: centroid of all points
    - C1, C2, C3: directions of principal components (PCA)

    Args:
        points_3d: 3D points (N, 3)

    Returns:
        cws: Control points in world coords (4, 3)
    """
    N = len(points_3d)

    # C0 = centroid
    c0 = np.mean(points_3d, axis=0)

    # PCA for other control points
    centered = points_3d - c0

    # Covariance matrix
    cov = (centered.T @ centered) / N

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # Control points: centroid + scaled eigenvectors
    # Scale by sqrt of eigenvalue for better numerical stability
    c1 = c0 + np.sqrt(eigenvalues[0]) * eigenvectors[:, 0]
    c2 = c0 + np.sqrt(eigenvalues[1]) * eigenvectors[:, 1]
    c3 = c0 + np.sqrt(eigenvalues[2]) * eigenvectors[:, 2]

    cws = np.array([c0, c1, c2, c3])

    return cws


def compute_barycentric_coordinates(points_3d: np.ndarray, cws: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates for each point with respect to control points

    Each point X_i = sum(alpha_ij * C_j) for j=0..3
    where sum(alpha_ij) = 1

    Args:
        points_3d: 3D points (N, 3)
        cws: Control points in world coords (4, 3)

    Returns:
        alphas: Barycentric coordinates (N, 4)
    """
    N = len(points_3d)

    # Build linear system: X = C @ alpha
    # where C = [c0, c1, c2, c3]^T (4x3)
    # and alpha = [alpha_0, alpha_1, alpha_2, alpha_3]^T (4x1)

    # Add constraint: sum(alpha) = 1
    # Use first control point as reference

    alphas = np.zeros((N, 4))

    for i in range(N):
        # Solve for alpha such that: points_3d[i] = C @ alpha
        # with constraint: sum(alpha) = 1

        # Build augmented system with constraint
        C_aug = np.vstack([cws.T, np.ones(4)])  # (4, 4)
        b_aug = np.append(points_3d[i], 1)  # (4,)

        # Solve: C_aug^T @ alpha = b_aug
        alpha = np.linalg.solve(C_aug, b_aug)

        alphas[i] = alpha

    return alphas


def compute_M_matrix(alphas: np.ndarray, points_2d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute M matrix that relates control points to image measurements

    For each point i:
    u_i = (sum(alpha_ij * fx * x_cj) + alpha_i0 * cx) / (sum(alpha_ij * z_cj))
    v_i = (sum(alpha_ij * fy * y_cj) + alpha_i0 * cy) / (sum(alpha_ij * z_cj))

    where (x_cj, y_cj, z_cj) are control point coords in camera frame

    Args:
        alphas: Barycentric coordinates (N, 4)
        points_2d: 2D points (N, 2)
        K: Intrinsic matrix (3, 3)

    Returns:
        M: Matrix (2N, 12)
    """
    N = len(points_2d)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    M = np.zeros((2 * N, 12))

    for i in range(N):
        u, v = points_2d[i]
        alpha = alphas[i]  # (4,)

        # Two equations per point
        # Row for u coordinate
        for j in range(4):
            # Contribution to x coordinate
            M[2*i, 3*j + 0] = fx * alpha[j]
            # Contribution to z coordinate (in denominator)
            M[2*i, 3*j + 2] = alpha[j] * (cx - u)

        # Row for v coordinate
        for j in range(4):
            # Contribution to y coordinate
            M[2*i + 1, 3*j + 1] = fy * alpha[j]
            # Contribution to z coordinate (in denominator)
            M[2*i + 1, 3*j + 2] = alpha[j] * (cy - v)

    return M


def compute_pose_from_control_points(cws: np.ndarray, ccs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute camera pose from corresponding control points

    This is the absolute orientation problem (Procrustes analysis)

    Args:
        cws: Control points in world coords (4, 3)
        ccs: Control points in camera coords (4, 3)

    Returns:
        R: Rotation matrix (3, 3)
        t: Translation vector (3,)
    """
    # Compute centroids
    cw_mean = np.mean(cws, axis=0)
    cc_mean = np.mean(ccs, axis=0)

    # Center the points
    cws_centered = cws - cw_mean
    ccs_centered = ccs - cc_mean

    # Compute cross-covariance matrix
    H = cws_centered.T @ ccs_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = cc_mean - R @ cw_mean

    return R, t
