"""
Phase 8 - Step 8.3: Visualization
Visualizes reconstruction results.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize_point_cloud(points: np.ndarray, colors: np.ndarray = None,
                          save_path: str = None, title: str = "3D Point Cloud"):
    """
    Visualize 3D point cloud

    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3), optional
        save_path: Path to save figure, optional
        title: Plot title
    """
    if len(points) == 0:
        print("Warning: No points to visualize")
        return
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None:
        # Normalize colors to 0-1 range
        colors_normalized = colors / 255.0
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors_normalized, marker='.', s=1)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='blue', marker='.', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close()


def plot_cameras(cameras: List[Dict], points: np.ndarray = None,
                save_path: str = None, title: str = "Camera Trajectory"):
    """
    Plot camera positions and orientations

    Args:
        cameras: List of camera dictionaries with R, t
        points: Optional 3D points to show (N, 3)
        save_path: Path to save figure, optional
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cameras
    camera_positions = []
    for cam in cameras:
        R = cam['R']
        t = cam['t']

        # Camera center in world coordinates: C = -R^T * t
        C = -R.T @ t
        camera_positions.append(C)

    camera_positions = np.array(camera_positions)

    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
              c='red', marker='o', s=100, label='Cameras')

    # Plot trajectory
    ax.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
           'r--', linewidth=1, alpha=0.5)

    # Plot points if provided
    if points is not None:
        # Subsample for visualization
        if len(points) > 1000:
            indices = np.random.choice(len(points), 1000, replace=False)
            points_vis = points[indices]
        else:
            points_vis = points

        ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2],
                  c='blue', marker='.', s=1, alpha=0.3, label='3D Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved camera plot to {save_path}")

    plt.close()
