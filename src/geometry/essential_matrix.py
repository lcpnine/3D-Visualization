"""
Phase 4 - Step 4.1: Essential Matrix Recovery
Computes the essential matrix from the fundamental matrix and camera intrinsics.
"""

import numpy as np
from config import cfg


def fundamental_to_essential(F: np.ndarray, K1: np.ndarray, K2: np.ndarray = None) -> np.ndarray:
    """
    Convert fundamental matrix to essential matrix

    E = K2^T * F * K1

    For same camera: E = K^T * F * K

    Args:
        F: Fundamental matrix (3, 3)
        K1: Intrinsic matrix of camera 1 (3, 3)
        K2: Intrinsic matrix of camera 2 (3, 3), optional (defaults to K1)

    Returns:
        E: Essential matrix (3, 3)
    """
    if K2 is None:
        K2 = K1

    # E = K2^T * F * K1
    E = K2.T @ F @ K1

    # Enforce essential matrix constraints
    E = enforce_essential_constraints(E)

    return E


def enforce_essential_constraints(E: np.ndarray) -> np.ndarray:
    """
    Enforce essential matrix constraints via SVD

    Essential matrix has:
    - Rank 2
    - Two equal singular values and one zero singular value
    - Typically normalized so the two non-zero singular values are 1

    Args:
        E: Essential matrix (3, 3)

    Returns:
        E_corrected: Corrected essential matrix
    """
    # SVD decomposition
    U, S, Vt = np.linalg.svd(E)

    # Set singular values to (1, 1, 0)
    S_corrected = np.array([1.0, 1.0, 0.0])

    # Reconstruct E
    E_corrected = U @ np.diag(S_corrected) @ Vt

    return E_corrected


def check_essential_matrix(E: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if matrix satisfies essential matrix constraints

    Args:
        E: Essential matrix (3, 3)
        tol: Tolerance for checks

    Returns:
        True if valid essential matrix
    """
    # Check rank = 2
    rank = np.linalg.matrix_rank(E, tol=tol)
    if rank != 2:
        return False

    # Check singular values
    U, S, Vt = np.linalg.svd(E)

    # Two singular values should be equal (or close to equal)
    if not np.isclose(S[0], S[1], atol=tol):
        return False

    # Third singular value should be zero
    if not np.isclose(S[2], 0, atol=tol):
        return False

    return True


if __name__ == "__main__":
    # Test essential matrix recovery
    import sys
    sys.path.insert(0, '.')

    from src.preprocessing.image_loader import load_images
    from src.preprocessing.camera_calibration import estimate_intrinsic_matrix
    from src.features.harris_detector import detect_harris_corners
    from src.features.descriptor import compute_descriptors
    from src.matching.matcher import match_descriptors, get_matched_points
    from src.matching.ransac import estimate_fundamental_matrix_ransac

    print("Testing Essential Matrix Recovery")
    print("=" * 50)

    # Load images
    print("\nLoading images...")
    images, metadata = load_images("data/scene1")

    # Get intrinsic matrix
    width = metadata[0]['width']
    height = metadata[0]['height']
    K = estimate_intrinsic_matrix(width, height)

    print(f"\nIntrinsic matrix K:")
    print(K)

    # Process first two images
    img1, img2 = images[0], images[1]

    # Detect and match features
    print("\nDetecting and matching features...")
    corners1 = detect_harris_corners(img1)
    corners2 = detect_harris_corners(img2)
    desc1 = compute_descriptors(img1, corners1)
    desc2 = compute_descriptors(img2, corners2)
    matches, _ = match_descriptors(desc1, desc2)
    points1, points2 = get_matched_points(corners1, corners2, matches)

    # Estimate fundamental matrix
    print("\nEstimating fundamental matrix...")
    F, inlier_mask = estimate_fundamental_matrix_ransac(points1, points2)
    print(f"Inliers: {np.sum(inlier_mask)}/{len(points1)}")

    # Convert to essential matrix
    print("\nConverting to essential matrix...")
    E = fundamental_to_essential(F, K)

    print(f"\nEssential matrix E:")
    print(E)

    # Check properties
    print(f"\nEssential matrix properties:")
    rank = np.linalg.matrix_rank(E)
    print(f"  Rank: {rank} (should be 2)")

    U, S, Vt = np.linalg.svd(E)
    print(f"  Singular values: {S}")
    print(f"  Expected: [1.0, 1.0, 0.0]")

    det_E = np.linalg.det(E)
    print(f"  Determinant: {det_E:.10f} (should be ≈ 0)")

    is_valid = check_essential_matrix(E)
    print(f"\nEssential matrix is valid: {is_valid}")
