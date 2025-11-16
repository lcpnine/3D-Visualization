"""
Phase 1 - Step 1.2: Camera Intrinsic Parameter Estimation
Computes the camera intrinsic matrix from image dimensions and field of view.
"""

import numpy as np
from typing import Tuple
from config import cfg


def estimate_intrinsic_matrix(image_width: int, image_height: int,
                             fov_degrees: float = None) -> np.ndarray:
    """
    Estimate the 3x3 camera intrinsic matrix K

    K = [fx  0  cx]
        [0  fy  cy]
        [0   0   1]

    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_degrees: Field of view in degrees (default: from config)

    Returns:
        K: 3x3 intrinsic matrix
    """
    if fov_degrees is None:
        fov_degrees = cfg.DEFAULT_FOV

    # Compute focal length from FOV
    fx, fy = compute_focal_length(image_width, image_height, fov_degrees)

    # Compute principal point (assume at image center)
    cx, cy = get_principal_point(image_width, image_height)

    # Construct intrinsic matrix
    K = np.array([
        [fx,  0,  cx],
        [0,  fy,  cy],
        [0,   0,   1]
    ], dtype=np.float64)

    return K


def compute_focal_length(width: int, height: int, fov_degrees: float) -> Tuple[float, float]:
    """
    Compute focal length in pixels from field of view

    Mathematical formula:
    For horizontal FOV: fx = width / (2 * tan(FOV/2))
    For vertical FOV: fy = height / (2 * tan(FOV/2))

    We assume square pixels (fx = fy) and use the horizontal FOV

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_degrees: Horizontal field of view in degrees

    Returns:
        fx, fy: Focal lengths in pixels (both equal for square pixels)
    """
    # Convert FOV to radians
    fov_radians = np.deg2rad(fov_degrees)

    # Compute focal length from horizontal FOV
    fx = width / (2.0 * np.tan(fov_radians / 2.0))

    # Assume square pixels
    fy = fx

    return fx, fy


def get_principal_point(width: int, height: int) -> Tuple[float, float]:
    """
    Compute principal point coordinates (assume at image center)

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        cx, cy: Principal point coordinates in pixels
    """
    cx = width / 2.0
    cy = height / 2.0

    return cx, cy


def get_fov_from_focal_length(focal_length: float, image_width: int) -> float:
    """
    Compute field of view from focal length (inverse operation)

    Args:
        focal_length: Focal length in pixels
        image_width: Image width in pixels

    Returns:
        fov_degrees: Horizontal field of view in degrees
    """
    fov_radians = 2.0 * np.arctan(image_width / (2.0 * focal_length))
    fov_degrees = np.rad2deg(fov_radians)

    return fov_degrees


def camera_params_from_intrinsic(K: np.ndarray) -> dict:
    """
    Extract camera parameters from intrinsic matrix

    Args:
        K: 3x3 intrinsic matrix

    Returns:
        Dictionary containing fx, fy, cx, cy
    """
    return {
        'fx': K[0, 0],
        'fy': K[1, 1],
        'cx': K[0, 2],
        'cy': K[1, 2]
    }


def validate_intrinsic_matrix(K: np.ndarray) -> bool:
    """
    Validate intrinsic matrix structure

    Args:
        K: 3x3 intrinsic matrix

    Returns:
        True if valid, False otherwise
    """
    if K.shape != (3, 3):
        return False

    # Check structure
    if K[2, 0] != 0 or K[2, 1] != 0 or K[2, 2] != 1:
        return False

    if K[1, 0] != 0:
        return False

    # Check positive focal lengths
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        return False

    return True


if __name__ == "__main__":
    # Test camera calibration
    print("Testing Camera Intrinsic Estimation")
    print("=" * 50)

    # Typical iPhone image resolution (4032 x 3024)
    width = 4032
    height = 3024
    fov = 60.0  # degrees

    print(f"Image size: {width} x {height}")
    print(f"Field of View: {fov} degrees")

    K = estimate_intrinsic_matrix(width, height, fov)

    print("\nIntrinsic Matrix K:")
    print(K)

    params = camera_params_from_intrinsic(K)
    print("\nCamera Parameters:")
    print(f"  fx = {params['fx']:.2f} pixels")
    print(f"  fy = {params['fy']:.2f} pixels")
    print(f"  cx = {params['cx']:.2f} pixels")
    print(f"  cy = {params['cy']:.2f} pixels")

    # Verify
    fov_computed = get_fov_from_focal_length(params['fx'], width)
    print(f"\nVerification:")
    print(f"  Computed FOV from fx: {fov_computed:.2f} degrees")
    print(f"  Original FOV: {fov:.2f} degrees")

    # Validate
    is_valid = validate_intrinsic_matrix(K)
    print(f"\nIntrinsic matrix is valid: {is_valid}")
