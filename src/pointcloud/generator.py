"""
Phase 8 - Step 8.1: Point Cloud Generation
Extracts point cloud from reconstruction with colors.
"""

import numpy as np
from typing import List, Dict
import sys
sys.path.insert(0, '.')
from utils import io_utils


def extract_point_cloud(points_3d: np.ndarray, cameras: List[Dict],
                       images: List[np.ndarray], observations: List[Dict] = None) -> Dict:
    """
    Extract point cloud with colors from reconstruction

    Args:
        points_3d: 3D points (N, 3)
        cameras: List of camera dictionaries
        images: Original images (grayscale normalized)
        observations: Optional observation data for color extraction

    Returns:
        Dictionary with 'points' and 'colors'
    """
    N = len(points_3d)

    # For now, use default gray color (color extraction would require observation data)
    colors = np.ones((N, 3)) * 200  # Light gray

    return {
        'points': points_3d,
        'colors': colors.astype(np.uint8)
    }


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, filepath: str):
    """
    Save point cloud to PLY file

    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3) with values 0-255
        filepath: Output file path
    """
    io_utils.save_ply(points, colors, filepath)
    print(f"Saved point cloud to {filepath}")


def save_point_cloud_npy(points: np.ndarray, colors: np.ndarray, filepath: str):
    """
    Save point cloud to NumPy file

    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3)
        filepath: Output file path
    """
    np.savez_compressed(filepath, points=points, colors=colors)
    print(f"Saved point cloud to {filepath}")
