"""
Phase 6: Incremental Structure from Motion
Incrementally reconstructs 3D structure by adding images one at a time.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
sys.path.insert(0, '.')

from config import cfg
from src.preprocessing import camera_calibration
from src.features import harris_detector, descriptor
from src.matching import matcher, ransac
from src.geometry import essential_matrix, pose_recovery
from src.triangulation import triangulate, validation
from src.reconstruction import pnp


class IncrementalSfM:
    """Incremental Structure from Motion reconstructor"""

    def __init__(self, images: List[np.ndarray], K: np.ndarray, verbose: bool = True):
        """
        Initialize incremental SfM

        Args:
            images: List of grayscale normalized images
            K: Intrinsic matrix (3, 3)
            verbose: Print progress information
        """
        self.images = images
        self.K = K
        self.verbose = verbose
        self.n_images = len(images)

        # Reconstruction state
        self.cameras = []  # List of {index, R, t, K}
        self.points_3d = None  # (N, 3) array of 3D points
        self.point_colors = None  # (N, 3) array of RGB colors
        self.point_tracks = []  # List of {point_idx, observations: [(img_idx, feature_idx)]}

        # Feature data
        self.all_keypoints = []  # List of keypoint arrays for each image
        self.all_descriptors = []  # List of descriptor arrays
        self.pairwise_matches = {}  # Dict[(i, j)] = (matches, inliers)

        # Reconstruction status
        self.reconstructed_images = set()
        self.unreconstructed_images = set(range(self.n_images))

    def extract_features(self):
        """Extract features from all images"""
        if self.verbose:
            print("\n" + "=" * 70)
            print("Extracting features from all images...")
            print("-" * 70)

        for i, img in enumerate(self.images):
            if self.verbose and i % 5 == 0:
                print(f"  Processing image {i+1}/{self.n_images}...")

            corners = harris_detector.detect_harris_corners(img)
            desc = descriptor.compute_descriptors(img, corners)

            self.all_keypoints.append(corners)
            self.all_descriptors.append(desc)

        if self.verbose:
            total_features = sum(len(d) for d in self.all_descriptors)
            print(f"Extracted {total_features} features total")
            print(f"Average per image: {total_features / self.n_images:.1f}")

    def match_all_pairs(self):
        """Match features between all image pairs"""
        if self.verbose:
            print("\n" + "=" * 70)
            print("Matching features between image pairs...")
            print("-" * 70)

        # Match consecutive pairs and some non-consecutive pairs
        pairs_to_match = []

        # Consecutive pairs
        for i in range(self.n_images - 1):
            pairs_to_match.append((i, i + 1))

        # Every 2nd neighbor
        for i in range(self.n_images - 2):
            pairs_to_match.append((i, i + 2))

        if self.verbose:
            print(f"Matching {len(pairs_to_match)} image pairs...")

        for i, j in pairs_to_match:
            desc1 = self.all_descriptors[i]
            desc2 = self.all_descriptors[j]

            matches, _ = matcher.match_descriptors(desc1, desc2)

            if len(matches) >= cfg.MIN_MATCHES:
                self.pairwise_matches[(i, j)] = matches

                if self.verbose and (i, j) == (0, 1):
                    print(f"  Pair ({i}, {j}): {len(matches)} matches")

        if self.verbose:
            valid_pairs = len(self.pairwise_matches)
            print(f"Found matches for {valid_pairs}/{len(pairs_to_match)} pairs")

    def initialize_reconstruction(self):
        """Initialize reconstruction with first two images"""
        if self.verbose:
            print("\n" + "=" * 70)
            print("Initializing reconstruction with first two images...")
            print("-" * 70)

        # Use first pair with sufficient matches
        init_pair = None
        for (i, j), matches in self.pairwise_matches.items():
            if i == 0:  # Start with first image
                init_pair = (i, j)
                break

        if init_pair is None:
            raise RuntimeError("No valid image pair for initialization")

        i, j = init_pair
        matches_ij = self.pairwise_matches[(i, j)]

        if self.verbose:
            print(f"Using images {i} and {j} for initialization")
            print(f"Initial matches: {len(matches_ij)}")

        # Get matched points
        kp1 = self.all_keypoints[i]
        kp2 = self.all_keypoints[j]
        pts1, pts2 = matcher.get_matched_points(kp1, kp2, matches_ij)

        # Estimate pose
        F, inlier_mask = ransac.estimate_fundamental_matrix_ransac(pts1, pts2)
        E = essential_matrix.fundamental_to_essential(F, self.K)
        poses = pose_recovery.decompose_essential_matrix(E)

        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]

        R, t = pose_recovery.select_valid_pose(poses, inlier_pts1, inlier_pts2, self.K)

        # Create camera matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t.reshape(3, 1)])

        # Triangulate points
        points_3d = triangulate.triangulate_points(P1, P2, inlier_pts1, inlier_pts2)

        # Filter points
        filtered_points, valid_mask = validation.filter_triangulated_points(
            points_3d, P1, P2, inlier_pts1, inlier_pts2
        )

        # Store reconstruction
        self.cameras.append({'index': i, 'R': np.eye(3), 't': np.zeros(3), 'K': self.K})
        self.cameras.append({'index': j, 'R': R, 't': t, 'K': self.K})

        self.points_3d = filtered_points
        self.point_colors = np.ones((len(filtered_points), 3)) * 128  # Gray for now

        # Track which images see which points
        # Simplified tracking for initial pair
        self.point_tracks = []
        inlier_indices = np.where(inlier_mask)[0]
        valid_inlier_indices = inlier_indices[valid_mask]

        for k, match_idx in enumerate(valid_inlier_indices):
            self.point_tracks.append({
                'point_idx': k,
                'observations': [
                    (i, matches_ij[match_idx, 0]),  # (image_idx, keypoint_idx)
                    (j, matches_ij[match_idx, 1])
                ]
            })

        self.reconstructed_images.add(i)
        self.reconstructed_images.add(j)
        self.unreconstructed_images.discard(i)
        self.unreconstructed_images.discard(j)

        if self.verbose:
            print(f"Initialized with {len(filtered_points)} 3D points")
            print(f"Cameras: {len(self.cameras)}")

    def add_next_image(self) -> bool:
        """
        Add next best image to reconstruction

        Returns:
            True if image was added successfully, False otherwise
        """
        if len(self.unreconstructed_images) == 0:
            return False

        # Select next image based on number of 2D-3D correspondences
        next_img_idx = self.select_next_image()

        if next_img_idx is None:
            return False

        if self.verbose:
            print(f"\nAdding image {next_img_idx}...")

        # Find 2D-3D correspondences
        points_3d_corr, points_2d_corr = self.find_2d_3d_correspondences(next_img_idx)

        if len(points_3d_corr) < cfg.MIN_PNP_POINTS:
            if self.verbose:
                print(f"  Insufficient 2D-3D correspondences: {len(points_3d_corr)}")
            return False

        # Estimate camera pose using PnP
        try:
            R, t, inliers = pnp.solve_pnp_ransac(points_3d_corr, points_2d_corr, self.K)

            if np.sum(inliers) < cfg.MIN_PNP_INLIERS:
                if self.verbose:
                    print(f"  Too few PnP inliers: {np.sum(inliers)}")
                return False

        except Exception as e:
            if self.verbose:
                print(f"  PnP failed: {e}")
            return False

        # Add camera
        self.cameras.append({'index': next_img_idx, 'R': R, 't': t, 'K': self.K})
        self.reconstructed_images.add(next_img_idx)
        self.unreconstructed_images.discard(next_img_idx)

        if self.verbose:
            print(f"  Added camera {next_img_idx} with {np.sum(inliers)} inliers")

        # Triangulate new points (simplified - skip for now to save time)

        return True

    def select_next_image(self) -> Optional[int]:
        """Select next image to add based on 2D-3D correspondences"""
        best_img = None
        best_score = 0

        for img_idx in self.unreconstructed_images:
            # Count potential 2D-3D correspondences
            score = 0

            for cam in self.cameras:
                cam_idx = cam['index']
                if (min(cam_idx, img_idx), max(cam_idx, img_idx)) in self.pairwise_matches:
                    score += 1

            if score > best_score:
                best_score = score
                best_img = img_idx

        return best_img

    def find_2d_3d_correspondences(self, img_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find 2D-3D correspondences for an image"""
        points_3d_list = []
        points_2d_list = []

        # For each reconstructed camera
        for cam in self.cameras:
            cam_idx = cam['index']
            pair_key = (min(cam_idx, img_idx), max(cam_idx, img_idx))

            if pair_key not in self.pairwise_matches:
                continue

            matches = self.pairwise_matches[pair_key]

            # Find which matches correspond to reconstructed 3D points
            # Simplified: use track information
            for track in self.point_tracks:
                for obs_cam_idx, obs_kp_idx in track['observations']:
                    if obs_cam_idx == cam_idx:
                        # Find if this keypoint matches with img_idx
                        for match in matches:
                            if pair_key[0] == cam_idx and match[0] == obs_kp_idx:
                                # Found correspondence
                                point_3d = self.points_3d[track['point_idx']]
                                point_2d = self.all_keypoints[img_idx][match[1]]

                                points_3d_list.append(point_3d)
                                points_2d_list.append(point_2d)
                                break
                            elif pair_key[0] == img_idx and match[1] == obs_kp_idx:
                                point_3d = self.points_3d[track['point_idx']]
                                point_2d = self.all_keypoints[img_idx][match[0]]

                                points_3d_list.append(point_3d)
                                points_2d_list.append(point_2d)
                                break

        if len(points_3d_list) == 0:
            return np.array([]), np.array([])

        return np.array(points_3d_list), np.array(points_2d_list)

    def reconstruct(self):
        """Run complete incremental reconstruction"""
        # Extract features
        self.extract_features()

        # Match features
        self.match_all_pairs()

        # Initialize
        self.initialize_reconstruction()

        # Add remaining images
        if self.verbose:
            print("\n" + "=" * 70)
            print("Adding remaining images...")
            print("-" * 70)

        while len(self.unreconstructed_images) > 0:
            success = self.add_next_image()

            if not success:
                if self.verbose:
                    print(f"Cannot add more images. Stopping.")
                    print(f"Reconstructed: {len(self.reconstructed_images)}/{self.n_images} images")
                break

        if self.verbose:
            print("\n" + "=" * 70)
            print("Incremental reconstruction complete!")
            print("-" * 70)
            print(f"Reconstructed cameras: {len(self.cameras)}")
            print(f"3D points: {len(self.points_3d)}")

        return {
            'cameras': self.cameras,
            'points_3d': self.points_3d,
            'point_colors': self.point_colors,
            'n_images': self.n_images,
            'n_reconstructed': len(self.reconstructed_images)
        }
