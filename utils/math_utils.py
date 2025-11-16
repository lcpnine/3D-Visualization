"""
Common mathematical utility functions for Structure from Motion
"""

import numpy as np
from typing import Tuple


# ==================== Linear Algebra ====================

def svd_solve(A: np.ndarray) -> np.ndarray:
    """
    Solve homogeneous linear system Ax = 0 via SVD
    Returns the last column of V (corresponding to smallest singular value)

    Args:
        A: Matrix of shape (m, n)

    Returns:
        x: Solution vector (last column of V)
    """
    U, S, Vt = np.linalg.svd(A)
    # Last row of Vt = last column of V
    x = Vt[-1, :]
    return x


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length

    Args:
        v: Input vector

    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector

    [v]_× = [ 0   -v3   v2]
            [ v3   0   -v1]
            [-v2   v1   0 ]

    Used for cross product: [v]_× @ w = v × w

    Args:
        v: 3D vector

    Returns:
        3x3 skew-symmetric matrix
    """
    if v.shape != (3,):
        raise ValueError(f"Expected 3D vector, got shape {v.shape}")

    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)


# ==================== Coordinate Transformations ====================

def homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinates

    Args:
        points: Points of shape (N, d) where d=2 or d=3

    Returns:
        Homogeneous points of shape (N, d+1)
    """
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=points.dtype)
    return np.hstack([points, ones])


def euclidean(points: np.ndarray) -> np.ndarray:
    """
    Convert homogeneous coordinates to Euclidean

    Args:
        points: Homogeneous points of shape (N, d)

    Returns:
        Euclidean points of shape (N, d-1)
    """
    return points[:, :-1] / points[:, -1:]


def euclidean_single(point: np.ndarray) -> np.ndarray:
    """
    Convert single homogeneous point to Euclidean

    Args:
        point: Homogeneous point of shape (d,)

    Returns:
        Euclidean point of shape (d-1,)
    """
    return point[:-1] / point[-1]


# ==================== Rotation Utilities ====================

def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to axis-angle representation

    Args:
        R: 3x3 rotation matrix

    Returns:
        axis_angle: 3D vector, magnitude is angle in radians
    """
    # Using Rodrigues formula
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if theta < 1e-10:
        # Near identity, return zero vector
        return np.zeros(3, dtype=np.float64)

    # Extract axis
    axis = (1 / (2 * np.sin(theta))) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])

    # axis_angle = theta * axis
    return theta * axis


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix using Rodrigues formula

    R = I + (sin θ) K + (1 - cos θ) K²
    where K is the skew-symmetric matrix of the unit axis

    Args:
        axis_angle: 3D vector, magnitude is angle in radians

    Returns:
        R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(axis_angle)

    if theta < 1e-10:
        # Near zero, return identity
        return np.eye(3, dtype=np.float64)

    # Unit axis
    axis = axis_angle / theta

    # Skew-symmetric matrix
    K = skew_symmetric(axis)

    # Rodrigues formula
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return R


def enforce_orthogonal(R: np.ndarray) -> np.ndarray:
    """
    Project matrix to nearest orthogonal matrix (SO(3)) using SVD

    R_ortho = U * V^T

    Args:
        R: Approximate rotation matrix (3x3)

    Returns:
        R_ortho: Nearest orthogonal matrix with det = +1
    """
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt

    # Ensure det = +1 (rotation, not reflection)
    if np.linalg.det(R_ortho) < 0:
        Vt[-1, :] *= -1
        R_ortho = U @ Vt

    return R_ortho


def rotation_from_two_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that rotates v1 to v2

    Args:
        v1: Source 3D vector
        v2: Target 3D vector

    Returns:
        R: 3x3 rotation matrix such that R @ v1 ≈ v2
    """
    # Normalize vectors
    v1 = normalize_vector(v1)
    v2 = normalize_vector(v2)

    # Rotation axis (cross product)
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)

    # Angle
    cos_angle = np.dot(v1, v2)
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    if axis_norm < 1e-10:
        # Vectors are parallel or anti-parallel
        if cos_angle > 0:
            return np.eye(3)
        else:
            # 180 degree rotation around any perpendicular axis
            axis = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
            axis = normalize_vector(np.cross(v1, axis))
            angle = np.pi

    axis = axis / axis_norm
    axis_angle = angle * axis

    return axis_angle_to_rotation_matrix(axis_angle)


# ==================== Matrix Operations ====================

def condition_number(A: np.ndarray) -> float:
    """
    Compute condition number of matrix

    Args:
        A: Input matrix

    Returns:
        Condition number
    """
    return np.linalg.cond(A)


def is_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if matrix is a valid rotation matrix

    Args:
        R: 3x3 matrix
        tol: Tolerance for checks

    Returns:
        True if valid rotation matrix
    """
    if R.shape != (3, 3):
        return False

    # Check orthogonality: R^T @ R = I
    should_be_identity = R.T @ R
    identity = np.eye(3)
    if not np.allclose(should_be_identity, identity, atol=tol):
        return False

    # Check determinant = 1
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=tol):
        return False

    return True


def make_homogeneous_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Create 4x4 homogeneous transformation matrix from R and t

    T = [R  t]
        [0  1]

    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector

    Returns:
        T: 4x4 homogeneous matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


# ==================== Statistical Functions ====================

def compute_median_absolute_deviation(data: np.ndarray) -> float:
    """
    Compute Median Absolute Deviation (MAD)

    MAD = median(|x - median(x)|)

    Args:
        data: Input array

    Returns:
        MAD value
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad


def robust_mean(data: np.ndarray, sigma: float = 2.0) -> float:
    """
    Compute robust mean by excluding outliers

    Args:
        data: Input array
        sigma: Number of standard deviations for outlier threshold

    Returns:
        Robust mean
    """
    mean = np.mean(data)
    std = np.std(data)
    mask = np.abs(data - mean) < sigma * std
    if np.sum(mask) == 0:
        return mean
    return np.mean(data[mask])


# ==================== Numerical Stability ====================

def normalize_homogeneous(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize 2D points for improved numerical stability (Hartley normalization)

    Shift centroid to origin and scale so average distance to origin is sqrt(2)

    Args:
        points: 2D points of shape (N, 2)

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


if __name__ == "__main__":
    # Test mathematical utilities
    print("Testing Mathematical Utilities")
    print("=" * 50)

    # Test rotation conversion
    print("\n1. Rotation matrix conversion:")
    axis_angle = np.array([0.1, 0.2, 0.3])
    R = axis_angle_to_rotation_matrix(axis_angle)
    print(f"Axis-angle: {axis_angle}")
    print(f"Rotation matrix:\n{R}")
    print(f"Is valid rotation: {is_rotation_matrix(R)}")

    # Convert back
    aa_back = rotation_matrix_to_axis_angle(R)
    print(f"Converted back: {aa_back}")
    print(f"Difference: {np.linalg.norm(axis_angle - aa_back):.10f}")

    # Test normalization
    print("\n2. Point normalization:")
    points = np.random.randn(10, 2) * 100 + 50
    norm_points, T = normalize_homogeneous(points)
    print(f"Original centroid: {np.mean(points, axis=0)}")
    print(f"Normalized centroid: {np.mean(norm_points, axis=0)}")
    print(f"Normalized avg distance: {np.mean(np.linalg.norm(norm_points, axis=1)):.4f}")
    print(f"Expected: {np.sqrt(2):.4f}")
