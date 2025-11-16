"""
Comprehensive debugging visualizations for Structure from Motion pipeline
Provides visual debugging at each stage to understand reconstruction quality
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class DebugVisualizer:
    """Handles all debugging visualizations for SfM pipeline"""

    def __init__(self, output_dir: str = "output/debug"):
        """
        Initialize debug visualizer

        Args:
            output_dir: Directory to save debug visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_features(self, image: np.ndarray, keypoints: np.ndarray,
                          img_idx: int, save_name: str = None):
        """
        Visualize detected features on an image

        Args:
            image: Grayscale image (H, W)
            keypoints: Detected keypoints (N, 2) in [x, y] format
            img_idx: Image index
            save_name: Optional custom save name
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.imshow(image, cmap='gray')

        # Plot keypoints
        if len(keypoints) > 0:
            ax.scatter(keypoints[:, 0], keypoints[:, 1],
                      c='red', s=20, alpha=0.6, marker='+')

        ax.set_title(f'Feature Detection - Image {img_idx}\n{len(keypoints)} features detected',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add text with statistics
        textstr = f'Total features: {len(keypoints)}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=props)

        if save_name is None:
            save_name = f"features_img_{img_idx:03d}.png"

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray,
                         kp1: np.ndarray, kp2: np.ndarray,
                         matches: np.ndarray,
                         inlier_mask: np.ndarray = None,
                         idx1: int = 0, idx2: int = 1,
                         save_name: str = None):
        """
        Visualize feature matches between two images

        Args:
            img1, img2: Grayscale images
            kp1, kp2: Keypoints for each image (N, 2)
            matches: Match indices (M, 2) where matches[i] = [idx_in_kp1, idx_in_kp2]
            inlier_mask: Boolean mask indicating inliers (M,)
            idx1, idx2: Image indices for naming
            save_name: Optional custom save name
        """
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        h_max = max(h1, h2)

        # Create side-by-side image
        combined = np.zeros((h_max, w1 + w2), dtype=img1.dtype)
        combined[:h1, :w1] = img1
        combined[:h2, w1:] = img2

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.imshow(combined, cmap='gray')

        if len(matches) > 0:
            # Separate inliers and outliers
            if inlier_mask is not None:
                inliers = matches[inlier_mask]
                outliers = matches[~inlier_mask]
            else:
                inliers = matches
                outliers = np.array([])

            # Draw outliers in red (thin, transparent)
            for match in outliers:
                pt1 = kp1[match[0]]
                pt2 = kp2[match[1]] + np.array([w1, 0])

                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                       'r-', linewidth=0.5, alpha=0.3)

            # Draw inliers in green (thicker, more visible)
            for match in inliers:
                pt1 = kp1[match[0]]
                pt2 = kp2[match[1]] + np.array([w1, 0])

                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],
                       'g-', linewidth=1, alpha=0.6)

            # Plot keypoints
            if len(inliers) > 0:
                inlier_kp1 = kp1[inliers[:, 0]]
                inlier_kp2 = kp2[inliers[:, 1]] + np.array([w1, 0])
                ax.scatter(inlier_kp1[:, 0], inlier_kp1[:, 1],
                          c='lime', s=30, marker='o', edgecolors='green', linewidths=1)
                ax.scatter(inlier_kp2[:, 0], inlier_kp2[:, 1],
                          c='lime', s=30, marker='o', edgecolors='green', linewidths=1)

            if len(outliers) > 0:
                outlier_kp1 = kp1[outliers[:, 0]]
                outlier_kp2 = kp2[outliers[:, 1]] + np.array([w1, 0])
                ax.scatter(outlier_kp1[:, 0], outlier_kp1[:, 1],
                          c='red', s=20, marker='x', alpha=0.5)
                ax.scatter(outlier_kp2[:, 0], outlier_kp2[:, 1],
                          c='red', s=20, marker='x', alpha=0.5)

            n_inliers = len(inliers)
            n_outliers = len(outliers)
            inlier_ratio = n_inliers / len(matches) if len(matches) > 0 else 0
        else:
            n_inliers = 0
            n_outliers = 0
            inlier_ratio = 0

        ax.set_title(f'Feature Matching: Image {idx1} ↔ Image {idx2}\n'
                    f'Inliers: {n_inliers} (green) | Outliers: {n_outliers} (red) | '
                    f'Ratio: {inlier_ratio:.1%}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add statistics
        textstr = f'Total matches: {len(matches)}\nInliers: {n_inliers}\nOutliers: {n_outliers}\nInlier ratio: {inlier_ratio:.1%}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)

        if save_name is None:
            save_name = f"matches_{idx1:03d}_{idx2:03d}.png"

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_reprojection_errors(self, points_3d: np.ndarray,
                                     points_2d: np.ndarray,
                                     P: np.ndarray,
                                     img: np.ndarray,
                                     img_idx: int,
                                     save_name: str = None):
        """
        Visualize reprojection errors on an image

        Args:
            points_3d: 3D points (N, 3)
            points_2d: Corresponding 2D points (N, 2)
            P: Camera projection matrix (3, 4)
            img: Image to overlay on
            img_idx: Image index
            save_name: Optional save name
        """
        if len(points_3d) == 0:
            return

        # Reproject 3D points
        points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        projected = (P @ points_3d_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        # Compute errors
        errors = np.linalg.norm(projected - points_2d, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Image with reprojection visualization
        ax1.imshow(img, cmap='gray')

        # Color-code points by error
        scatter = ax1.scatter(points_2d[:, 0], points_2d[:, 1],
                            c=errors, cmap='jet', s=50, alpha=0.7,
                            vmin=0, vmax=np.percentile(errors, 95))

        # Draw error vectors
        for i in range(len(points_2d)):
            ax1.plot([points_2d[i, 0], projected[i, 0]],
                    [points_2d[i, 1], projected[i, 1]],
                    'r-', linewidth=0.5, alpha=0.5)

        ax1.scatter(projected[:, 0], projected[:, 1],
                   c='lime', s=20, marker='+', alpha=0.8)

        ax1.set_title(f'Reprojection Errors - Image {img_idx}\n'
                     f'Mean: {np.mean(errors):.2f} px | Median: {np.median(errors):.2f} px',
                     fontsize=12, fontweight='bold')
        ax1.axis('off')

        plt.colorbar(scatter, ax=ax1, label='Error (pixels)')

        # Right: Error histogram
        ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(errors), color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(errors):.2f} px')
        ax2.axvline(np.median(errors), color='g', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(errors):.2f} px')
        ax2.set_xlabel('Reprojection Error (pixels)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Statistics text
        textstr = (f'Points: {len(points_3d)}\n'
                  f'Mean: {np.mean(errors):.2f} px\n'
                  f'Median: {np.median(errors):.2f} px\n'
                  f'Std: {np.std(errors):.2f} px\n'
                  f'Max: {np.max(errors):.2f} px')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.65, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        if save_name is None:
            save_name = f"reprojection_img_{img_idx:03d}.png"

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_2d_3d_correspondences(self, img: np.ndarray,
                                       points_2d: np.ndarray,
                                       points_3d: np.ndarray,
                                       inlier_mask: np.ndarray = None,
                                       img_idx: int = 0,
                                       save_name: str = None):
        """
        Visualize 2D-3D correspondences for PnP

        Args:
            img: Image
            points_2d: 2D points (N, 2)
            points_3d: Corresponding 3D points (N, 3)
            inlier_mask: Boolean mask for inliers after RANSAC
            img_idx: Image index
            save_name: Optional save name
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.imshow(img, cmap='gray')

        if inlier_mask is not None and len(inlier_mask) > 0:
            # Separate inliers and outliers
            inliers = points_2d[inlier_mask]
            outliers = points_2d[~inlier_mask]

            if len(outliers) > 0:
                ax.scatter(outliers[:, 0], outliers[:, 1],
                          c='red', s=50, marker='x', alpha=0.6, label='Outliers')

            if len(inliers) > 0:
                ax.scatter(inliers[:, 0], inliers[:, 1],
                          c='lime', s=60, marker='o', edgecolors='green',
                          linewidths=2, alpha=0.8, label='Inliers')

            n_inliers = np.sum(inlier_mask)
            n_outliers = len(inlier_mask) - n_inliers
        else:
            ax.scatter(points_2d[:, 0], points_2d[:, 1],
                      c='cyan', s=50, marker='o', alpha=0.7)
            n_inliers = len(points_2d)
            n_outliers = 0

        ax.set_title(f'2D-3D Correspondences - Image {img_idx}\n'
                    f'Total: {len(points_2d)} | Inliers: {n_inliers} | Outliers: {n_outliers}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=11)

        # Add statistics
        textstr = (f'Total correspondences: {len(points_2d)}\n'
                  f'Inliers: {n_inliers}\n'
                  f'Outliers: {n_outliers}\n'
                  f'Inlier ratio: {n_inliers/len(points_2d) if len(points_2d) > 0 else 0:.1%}')
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)

        if save_name is None:
            save_name = f"2d_3d_correspondences_img_{img_idx:03d}.png"

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_point_cloud_growth(self, points_3d_history: List[np.ndarray],
                                    camera_indices: List[int],
                                    save_name: str = "point_cloud_growth.png"):
        """
        Visualize how point cloud grows as cameras are added

        Args:
            points_3d_history: List of point clouds at each stage
            camera_indices: List of camera indices added at each stage
            save_name: Save filename
        """
        n_stages = len(points_3d_history)

        if n_stages == 0:
            return

        # Create grid of subplots
        n_cols = min(4, n_stages)
        n_rows = (n_stages + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

        for i, (points, cam_idx) in enumerate(zip(points_3d_history, camera_indices)):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c='blue', s=10, alpha=0.6)

                # Set equal aspect ratio
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

            ax.set_title(f'After adding camera {cam_idx}\n{len(points)} points',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        plt.suptitle('Point Cloud Growth During Reconstruction',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_camera_frustums(self, cameras: List[dict],
                                 points_3d: np.ndarray,
                                 frustum_size: float = 0.3,
                                 save_name: str = "camera_frustums.png"):
        """
        Visualize cameras with frustums and point cloud

        Args:
            cameras: List of camera dicts with 'R', 't', 'index'
            points_3d: 3D points (N, 3)
            frustum_size: Size of camera frustum visualization
            save_name: Save filename
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        if len(points_3d) > 0:
            ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                      c='gray', s=5, alpha=0.4, label='3D Points')

        # Plot cameras
        camera_positions = []
        for i, cam in enumerate(cameras):
            R = cam['R']
            t = cam['t']

            # Camera center in world coordinates
            C = -R.T @ t
            camera_positions.append(C)

            # Camera coordinate axes
            axis_length = frustum_size
            axes = np.eye(3) * axis_length
            axes_world = R.T @ axes

            # Draw axes
            colors = ['r', 'g', 'b']
            labels = ['X', 'Y', 'Z']
            for j, (axis, color, label) in enumerate(zip(axes_world.T, colors, labels)):
                ax.plot([C[0], C[0] + axis[0]],
                       [C[1], C[1] + axis[1]],
                       [C[2], C[2] + axis[2]],
                       color=color, linewidth=2, alpha=0.8)

            # Draw frustum
            # Define frustum corners in camera coordinates
            f = frustum_size * 0.8
            corners_cam = np.array([
                [f, f, f],
                [f, -f, f],
                [-f, -f, f],
                [-f, f, f]
            ])

            # Transform to world coordinates
            corners_world = (R.T @ corners_cam.T).T + C

            # Draw frustum edges
            for j in range(4):
                next_j = (j + 1) % 4
                ax.plot([C[0], corners_world[j, 0]],
                       [C[1], corners_world[j, 1]],
                       [C[2], corners_world[j, 2]],
                       'c-', linewidth=1, alpha=0.6)
                ax.plot([corners_world[j, 0], corners_world[next_j, 0]],
                       [corners_world[j, 1], corners_world[next_j, 1]],
                       [corners_world[j, 2], corners_world[next_j, 2]],
                       'c-', linewidth=1, alpha=0.6)

            # Label camera
            ax.text(C[0], C[1], C[2], f'  Cam {cam["index"]}',
                   fontsize=10, fontweight='bold', color='black')

        # Draw camera trajectory
        if len(camera_positions) > 1:
            camera_positions = np.array(camera_positions)
            ax.plot(camera_positions[:, 0],
                   camera_positions[:, 1],
                   camera_positions[:, 2],
                   'mo-', linewidth=2, markersize=8, label='Camera Trajectory')

        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.set_zlabel('Z', fontsize=11)
        ax.set_title(f'3D Reconstruction with Camera Frustums\n'
                    f'{len(cameras)} cameras, {len(points_3d)} points',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)

        # Set equal aspect ratio
        if len(points_3d) > 0:
            max_range = np.array([
                points_3d[:, 0].max() - points_3d[:, 0].min(),
                points_3d[:, 1].max() - points_3d[:, 1].min(),
                points_3d[:, 2].max() - points_3d[:, 2].min()
            ]).max() / 2.0

            mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
            mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
            mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()

    def save_reconstruction_report(self, stats: dict, save_name: str = "reconstruction_report.txt"):
        """
        Save text report with reconstruction statistics

        Args:
            stats: Dictionary with reconstruction statistics
            save_name: Save filename
        """
        report_lines = [
            "=" * 70,
            "RECONSTRUCTION DEBUG REPORT",
            "=" * 70,
            "",
            f"Total images: {stats.get('n_images', 'N/A')}",
            f"Reconstructed cameras: {stats.get('n_cameras', 'N/A')}",
            f"Success rate: {stats.get('n_cameras', 0) / max(stats.get('n_images', 1), 1) * 100:.1f}%",
            "",
            f"Total 3D points: {stats.get('n_points', 'N/A')}",
            f"Points per camera: {stats.get('n_points', 0) / max(stats.get('n_cameras', 1), 1):.1f}",
            "",
            "=" * 70,
            "STAGE-BY-STAGE BREAKDOWN",
            "=" * 70,
        ]

        if 'stages' in stats:
            for stage in stats['stages']:
                report_lines.extend([
                    "",
                    f"Stage: {stage.get('name', 'Unknown')}",
                    f"  Camera added: {stage.get('camera_idx', 'N/A')}",
                    f"  Points before: {stage.get('points_before', 'N/A')}",
                    f"  Points after: {stage.get('points_after', 'N/A')}",
                    f"  New points: {stage.get('new_points', 'N/A')}",
                    f"  Status: {stage.get('status', 'N/A')}",
                ])

                if 'error' in stage:
                    report_lines.append(f"  Error: {stage['error']}")

        report_lines.extend([
            "",
            "=" * 70,
        ])

        report_path = self.output_dir / save_name
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Debug report saved to: {report_path}")
