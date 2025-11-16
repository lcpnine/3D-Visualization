"""
Phase 8 - Step 8.2: Statistical Outlier Removal
Removes noise points from point cloud.
"""

import numpy as np
from config import cfg


def statistical_outlier_removal(points: np.ndarray, k_neighbors: int = None,
                                std_ratio: float = None) -> np.ndarray:
    """
    Remove statistical outliers from point cloud

    Args:
        points: Point cloud (N, 3)
        k_neighbors: Number of neighbors to consider (default: from config)
        std_ratio: Standard deviation ratio threshold (default: from config)

    Returns:
        filtered_points: Filtered point cloud (M, 3) where M <= N
    """
    if k_neighbors is None:
        k_neighbors = cfg.KNN_NEIGHBORS
    if std_ratio is None:
        std_ratio = cfg.OUTLIER_STD_RATIO

    if len(points) < k_neighbors:
        return points

    # Compute k-nearest neighbor distances
    distances = compute_knn_distances(points, k_neighbors)

    # Compute statistics
    mean_dist = distances.mean()
    std_dist = distances.std()

    # Outliers are points with mean distance > mean + std_ratio * std
    threshold = mean_dist + std_ratio * std_dist
    inliers = distances < threshold

    filtered_points = points[inliers]

    print(f"Statistical outlier removal: kept {len(filtered_points)}/{len(points)} points")

    return filtered_points


def compute_knn_distances(points: np.ndarray, k: int) -> np.ndarray:
    """
    Compute mean distance to k nearest neighbors for each point

    Args:
        points: Point cloud (N, 3)
        k: Number of neighbors

    Returns:
        mean_distances: Mean k-NN distance for each point (N,)
    """
    N = len(points)

    # Compute pairwise distances (simplified, uses brute force)
    dist_matrix = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)

    # For each point, find k+1 nearest (including itself)
    # Then exclude itself and compute mean of k nearest
    sorted_indices = np.argsort(dist_matrix, axis=1)

    # Take k+1 nearest (first is always itself with distance 0)
    nearest_k_indices = sorted_indices[:, 1:k+1]  # Skip first (self)

    # Get distances to these neighbors
    mean_distances = np.zeros(N)
    for i in range(N):
        neighbor_distances = dist_matrix[i, nearest_k_indices[i]]
        mean_distances[i] = neighbor_distances.mean()

    return mean_distances
