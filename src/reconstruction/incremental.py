"""
Phase 6: Incremental Structure from Motion
Incrementally reconstructs 3D structure by adding images one at a time.
"""

import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, '.')

from config import cfg
from src.debug.visualizer import DebugVisualizer
from src.features import descriptor, harris_detector
from src.features.feature_factory import detect_and_compute_features
from src.geometry import essential_matrix, pose_recovery
from src.matching import matcher, ransac
from src.preprocessing import camera_calibration
from src.reconstruction import pnp
from src.triangulation import triangulate, validation


class IncrementalSfM:
    """Incremental Structure from Motion reconstructor"""

    def __init__(self, images: List[np.ndarray], K: np.ndarray, verbose: bool = True,
                 enable_debug: bool = False, debug_dir: str = "output/debug"):
        """
        Initialize incremental SfM

        Args:
            images: List of grayscale normalized images
            K: Intrinsic matrix (3, 3)
            verbose: Print progress information
            enable_debug: Enable visual debugging
            debug_dir: Directory for debug visualizations
        """
        self.images = images
        self.K = K
        self.verbose = verbose
        self.n_images = len(images)

        # Debug mode
        self.enable_debug = enable_debug
        self.debug_viz = DebugVisualizer(debug_dir) if enable_debug else None
        self.reconstruction_history = []  # Track point cloud growth

        # Reconstruction state
        self.cameras = []  # List of {index, R, t, K}
        self.points_3d = None  # (N, 3) array of 3D points
        self.point_colors = None  # (N, 3) array of RGB colors
        self.point_tracks = []  # List of {point_idx, observations: [(img_idx, feature_idx)]}

        # Feature data
        self.all_keypoints = []  # List of keypoint arrays for each image
        self.all_descriptors = []  # List of descriptor arrays
        self.descriptor_type = None  # Type of descriptors: 'float' or 'binary'
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

        total_start_time = time.time()

        for i, img in enumerate(self.images):
            if self.verbose:
                print(f"  Processing image {i+1}/{self.n_images}...")

            img_start_time = time.time()

            # Detect features and compute descriptors using factory
            keypoints, descriptors, desc_type = detect_and_compute_features(img)

            # Store descriptor type (should be same for all images)
            if self.descriptor_type is None:
                self.descriptor_type = desc_type
                if self.verbose:
                    print(f"  Descriptor type: {desc_type}")

            img_time = time.time() - img_start_time

            self.all_keypoints.append(keypoints)
            self.all_descriptors.append(descriptors)

            if self.verbose:
                print(f"    Detected {len(keypoints)} features")
                print(f"    Computed {len(descriptors)} descriptors")
                print(f"    Total time for image: {img_time:.2f}s")

            # Debug: Visualize features
            if self.enable_debug:
                self.debug_viz.visualize_features(img, keypoints, i)

        total_time = time.time() - total_start_time

        if self.verbose:
            total_features = sum(len(d) for d in self.all_descriptors)
            print(f"\nTotal extraction time: {total_time:.2f}s")
            print(f"Extracted {total_features} features total")
            print(f"Average per image: {total_features / self.n_images:.1f}")

            # Diagnostic: Check descriptor statistics for first image
            if len(self.all_descriptors) > 0 and len(self.all_descriptors[0]) > 0:
                desc_sample = self.all_descriptors[0]
                print(f"  Descriptor diagnostics (image 0):")
                print(f"    Shape: {desc_sample.shape}")
                print(f"    Type: {self.descriptor_type}")

                if self.descriptor_type == 'float':
                    desc_norms = np.linalg.norm(desc_sample, axis=1)
                    print(f"    L2 norms - mean: {desc_norms.mean():.4f}, std: {desc_norms.std():.4f}")
                    print(f"    Values - mean: {desc_sample.mean():.4f}, std: {desc_sample.std():.4f}")
                elif self.descriptor_type == 'binary':
                    # For binary descriptors, show bit statistics
                    total_bits = desc_sample.shape[1] * 8
                    bits_per_desc = np.unpackbits(desc_sample, axis=1).sum(axis=1)
                    print(f"    Bits per descriptor: {total_bits}")
                    print(f"    Set bits - mean: {bits_per_desc.mean():.1f}, std: {bits_per_desc.std():.1f}")

        if self.enable_debug and self.verbose:
            print(f"  Debug: Feature visualizations saved to {self.debug_viz.output_dir}")

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

        # Diagnostic counters
        total_raw_matches = 0
        pairs_with_any_matches = 0

        for i, j in pairs_to_match:
            desc1 = self.all_descriptors[i]
            desc2 = self.all_descriptors[j]

            # Check if descriptors are valid
            if len(desc1) == 0 or len(desc2) == 0:
                if self.verbose and (i, j) in [(0, 1), (1, 2)]:
                    print(f"  Pair ({i}, {j}): Empty descriptors (desc1={len(desc1)}, desc2={len(desc2)})")
                continue

            # Enable debug output for first pair
            debug_this_pair = self.verbose and (i, j) == (0, 1)
            if debug_this_pair:
                print(f"  Debugging pair ({i}, {j}):")

            matches, _ = matcher.match_descriptors(desc1, desc2, debug=debug_this_pair,
                                                   descriptor_type=self.descriptor_type)

            # Diagnostic output for first few pairs
            if self.verbose and (i, j) in [(0, 1), (1, 2), (0, 2)]:
                print(f"  Pair ({i}, {j}): {len(matches)} matches (desc1={len(desc1)}, desc2={len(desc2)})")

            if len(matches) > 0:
                pairs_with_any_matches += 1
                total_raw_matches += len(matches)

            if len(matches) >= cfg.MIN_MATCHES:
                self.pairwise_matches[(i, j)] = matches

        if self.verbose:
            valid_pairs = len(self.pairwise_matches)
            print(f"Found matches for {valid_pairs}/{len(pairs_to_match)} pairs")
            print(f"  Pairs with any matches: {pairs_with_any_matches}")
            print(f"  Total raw matches: {total_raw_matches}")
            print(f"  MIN_MATCHES threshold: {cfg.MIN_MATCHES}")

        if self.enable_debug and self.verbose:
            print(f"  Debug: Match visualizations saved to {self.debug_viz.output_dir}")

    def initialize_reconstruction(self):
        """Initialize reconstruction with first two images"""
        if self.verbose:
            print("\n" + "=" * 70)
            print("Initializing reconstruction with first two images...")
            print("-" * 70)

        init_pair = None
        
        # 5도 간격으로 찍었으므로, 5~8 프레임 떨어진 쌍을 찾는다.
        # (예: 5*6=30도, 5*8=40도)
        # 이 범위가 3D 복원에 가장 이상적인 'Wide Baseline'이다.
        min_frame_gap = 5
        max_frame_gap = 10 

        best_pair = None
        best_matches = 0

        # 0번 이미지를 기준으로 가장 매치가 많은 '적절한' 쌍을 찾는다
        i = 0
        for j in range(i + min_frame_gap, min(i + max_frame_gap + 1, self.n_images)):
            pair_key = (i, j)
            if pair_key in self.pairwise_matches:
                current_matches = len(self.pairwise_matches[pair_key])
                if current_matches > best_matches:
                    best_matches = current_matches
                    best_pair = pair_key

        init_pair = best_pair
        
        # 만약 (0, 5~10) 쌍을 못 찾으면, 그냥 원래 로직대로 (0, 1)이든 (0, 2)든 시도한다.
        if init_pair is None:
            if self.verbose:
                print(f"  Warning: No ideal wide-baseline pair found (e.g., 0 vs 5-10).")
                print(f"  Falling back to closest sequential pair...")
            for (i, j), matches in self.pairwise_matches.items():
                if i == 0:  # Start with first image
                    init_pair = (i, j)
                    break
        else:
             if self.verbose:
                print(f"  Selected wide-baseline pair {init_pair} with {best_matches} matches for initialization.")

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

        if self.verbose:
            print(f"Inlier point coordinates (pixels):")
            print(f"  Image 1: x range [{inlier_pts1[:, 0].min():.1f}, {inlier_pts1[:, 0].max():.1f}], y range [{inlier_pts1[:, 1].min():.1f}, {inlier_pts1[:, 1].max():.1f}]")
            print(f"  Image 2: x range [{inlier_pts2[:, 0].min():.1f}, {inlier_pts2[:, 0].max():.1f}], y range [{inlier_pts2[:, 1].min():.1f}, {inlier_pts2[:, 1].max():.1f}]")

        R, t = pose_recovery.select_valid_pose(poses, inlier_pts1, inlier_pts2, self.K)

        # Debug: Visualize matches with inliers/outliers
        if self.enable_debug:
            self.debug_viz.visualize_matches(
                self.images[i], self.images[j], kp1, kp2, matches_ij, inlier_mask, i, j,
                save_name=f"init_matches_{i:03d}_{j:03d}.png"
            )

        # Create camera matrices
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t.reshape(3, 1)])

        if self.verbose:
            print(f"Camera matrices:")
            print(f"  K (intrinsics): focal length = {self.K[0,0]:.2f}, principal point = ({self.K[0,2]:.2f}, {self.K[1,2]:.2f})")
            print(f"  P1 shape: {P1.shape}, P2 shape: {P2.shape}")

        # Triangulate points
        points_3d = triangulate.triangulate_points(P1, P2, inlier_pts1, inlier_pts2)

        # Sanity check: manually project first 3D point
        if self.verbose and len(points_3d) > 0:
            X_test = np.append(points_3d[0], 1)  # Homogeneous
            proj1 = P1 @ X_test
            proj2 = P2 @ X_test
            proj1_2d = proj1[:2] / proj1[2]
            proj2_2d = proj2[:2] / proj2[2]
            print(f"  Sanity check (first point):")
            print(f"    3D point: {points_3d[0]}")
            print(f"    Original 2D (img1): {inlier_pts1[0]}, Projected: {proj1_2d}, Error: {np.linalg.norm(proj1_2d - inlier_pts1[0]):.2f} px")
            print(f"    Original 2D (img2): {inlier_pts2[0]}, Projected: {proj2_2d}, Error: {np.linalg.norm(proj2_2d - inlier_pts2[0]):.2f} px")

        if self.verbose:
            print(f"Triangulated {len(points_3d)} points before filtering")

        # Filter points
        filtered_points, valid_mask = validation.filter_triangulated_points(
            points_3d, P1, P2, inlier_pts1, inlier_pts2
        )

        if self.verbose:
            # Debug filtering
            errors1 = validation.compute_all_reprojection_errors(points_3d, P1, inlier_pts1)
            errors2 = validation.compute_all_reprojection_errors(points_3d, P2, inlier_pts2)
            total_errors = errors1 + errors2

            print(f"  Reprojection errors: min={total_errors.min():.2f}, max={total_errors.max():.2f}, mean={total_errors.mean():.2f}")
            print(f"  Points passing reproj filter: {np.sum(total_errors < cfg.REPROJ_ERROR_THRESHOLD)}/{len(points_3d)}")

            depths1 = points_3d[:, 2]
            print(f"  Depths (cam1): min={depths1.min():.2f}, max={depths1.max():.2f}, mean={depths1.mean():.2f}")
            print(f"  Points passing depth filter (cam1): {np.sum((depths1 > cfg.MIN_DEPTH) & (depths1 < cfg.MAX_DEPTH))}/{len(points_3d)}")

            print(f"  Final: {len(filtered_points)}/{len(points_3d)} points passed all filters")

        # Debug: Visualize reprojection errors for both cameras
        if self.enable_debug:
            valid_inlier_pts1 = inlier_pts1[valid_mask]
            valid_inlier_pts2 = inlier_pts2[valid_mask]

            self.debug_viz.visualize_reprojection_errors(
                filtered_points, valid_inlier_pts1, P1, self.images[i], i,
                save_name=f"init_reprojection_img_{i:03d}.png"
            )
            self.debug_viz.visualize_reprojection_errors(
                filtered_points, valid_inlier_pts2, P2, self.images[j], j,
                save_name=f"init_reprojection_img_{j:03d}.png"
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

        # Track reconstruction history for debugging
        if self.enable_debug:
            self.reconstruction_history.append((j, self.points_3d.copy()))

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
            if self.verbose:
                print(f"  No suitable next image found")
            return False

        if self.verbose:
            print(f"\nAdding image {next_img_idx}...")

        # Find 2D-3D correspondences
        points_3d_corr, points_2d_corr = self.find_2d_3d_correspondences(next_img_idx)

        if len(points_3d_corr) < cfg.MIN_PNP_POINTS:
            if self.verbose:
                print(f"  Insufficient 2D-3D correspondences: {len(points_3d_corr)}")
            return False

        if self.verbose:
            print(f"  Found {len(points_3d_corr)} 2D-3D correspondences")

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

        # Debug: Visualize 2D-3D correspondences
        if self.enable_debug:
            self.debug_viz.visualize_2d_3d_correspondences(
                self.images[next_img_idx], points_2d_corr, points_3d_corr,
                inliers, next_img_idx,
                save_name=f"pnp_correspondences_img_{next_img_idx:03d}.png"
            )

        # Add camera
        self.cameras.append({'index': next_img_idx, 'R': R, 't': t, 'K': self.K})
        self.reconstructed_images.add(next_img_idx)
        self.unreconstructed_images.discard(next_img_idx)

        n_points_before = len(self.points_3d)

        if self.verbose:
            print(f"  Added camera {next_img_idx} with {np.sum(inliers)} PnP inliers")

        # CRITICAL FIX: Triangulate new points with existing cameras
        self.triangulate_new_points(next_img_idx)

        n_points_after = len(self.points_3d)
        new_points = n_points_after - n_points_before

        if self.verbose:
            print(f"  Triangulated {new_points} new points (total: {n_points_after})")

        # Debug: Visualize reprojection errors for new camera
        if self.enable_debug:
            P = self.K @ np.hstack([R, t.reshape(3, 1)])
            inlier_3d = points_3d_corr[inliers]
            inlier_2d = points_2d_corr[inliers]

            self.debug_viz.visualize_reprojection_errors(
                inlier_3d, inlier_2d, P, self.images[next_img_idx], next_img_idx,
                save_name=f"reprojection_img_{next_img_idx:03d}.png"
            )

            # Track reconstruction history
            self.reconstruction_history.append((next_img_idx, self.points_3d.copy()))

        return True

    def triangulate_new_points(self, new_img_idx: int):
        """
        Triangulate new 3D points between the newly added camera and existing cameras

        Args:
            new_img_idx: Index of newly added image
        """
        new_cam = self.cameras[-1]  # Just added
        new_R = new_cam['R']
        new_t = new_cam['t']
        new_P = self.K @ np.hstack([new_R, new_t.reshape(3, 1)])

        # Get existing point indices that are already triangulated
        existing_point_kps = set()  # Set of (img_idx, kp_idx) tuples
        for track in self.point_tracks:
            for obs_img_idx, obs_kp_idx in track['observations']:
                existing_point_kps.add((obs_img_idx, obs_kp_idx))

        new_points_list = []
        new_colors_list = []
        new_tracks_list = []

        # Try to triangulate with all reconstructed cameras
        for ref_cam in self.cameras[:-1]:  # All cameras except the newly added one
            ref_idx = ref_cam['index']
            pair_key = (min(ref_idx, new_img_idx), max(ref_idx, new_img_idx))

            if pair_key not in self.pairwise_matches:
                continue

            matches = self.pairwise_matches[pair_key]

            # Create projection matrix for reference camera
            ref_R = ref_cam['R']
            ref_t = ref_cam['t']
            ref_P = self.K @ np.hstack([ref_R, ref_t.reshape(3, 1)])

            # Find matches that haven't been triangulated yet
            new_matches = []
            for match in matches:
                if pair_key[0] == ref_idx:
                    ref_kp_idx, new_kp_idx = match[0], match[1]
                else:
                    new_kp_idx, ref_kp_idx = match[0], match[1]

                # Check if either keypoint is already part of a 3D point
                if (ref_idx, ref_kp_idx) in existing_point_kps:
                    continue
                if (new_img_idx, new_kp_idx) in existing_point_kps:
                    continue

                new_matches.append((ref_kp_idx, new_kp_idx))

            if len(new_matches) == 0:
                continue

            # Get keypoints
            new_matches = np.array(new_matches)
            ref_kps = self.all_keypoints[ref_idx][new_matches[:, 0]]
            new_kps = self.all_keypoints[new_img_idx][new_matches[:, 1]]

            # Triangulate
            points_3d = triangulate.triangulate_points(ref_P, new_P, ref_kps, new_kps)

            # Filter by reprojection error and depth
            filtered_points, valid_mask = validation.filter_triangulated_points(
                points_3d, ref_P, new_P, ref_kps, new_kps
            )

            if len(filtered_points) == 0:
                continue

            # Store new points and tracks
            current_n_points = len(self.points_3d) + len(new_points_list)
            valid_matches = new_matches[valid_mask]

            for i, (ref_kp_idx, new_kp_idx) in enumerate(valid_matches):
                point_idx = current_n_points + i

                new_points_list.append(filtered_points[i])
                new_colors_list.append([128, 128, 128])  # Gray for now

                # Create track
                new_tracks_list.append({
                    'point_idx': point_idx,
                    'observations': [
                        (ref_idx, ref_kp_idx),
                        (new_img_idx, new_kp_idx)
                    ]
                })

                # Mark as triangulated
                existing_point_kps.add((ref_idx, ref_kp_idx))
                existing_point_kps.add((new_img_idx, new_kp_idx))

        # Add new points to reconstruction
        if len(new_points_list) > 0:
            new_points = np.array(new_points_list)
            new_colors = np.array(new_colors_list)

            self.points_3d = np.vstack([self.points_3d, new_points])
            self.point_colors = np.vstack([self.point_colors, new_colors])
            self.point_tracks.extend(new_tracks_list)

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

        # Debug: Visualize matches for key pairs
        if self.enable_debug:
            if self.verbose:
                print("\n  Generating match visualizations...")

            for (i, j), matches in list(self.pairwise_matches.items())[:5]:  # First 5 pairs
                kp1 = self.all_keypoints[i]
                kp2 = self.all_keypoints[j]

                # Get matched points for RANSAC
                pts1, pts2 = matcher.get_matched_points(kp1, kp2, matches)
                _, inlier_mask = ransac.estimate_fundamental_matrix_ransac(pts1, pts2)

                self.debug_viz.visualize_matches(
                    self.images[i], self.images[j], kp1, kp2, matches, inlier_mask, i, j
                )

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

        # Debug: Generate final visualizations
        if self.enable_debug:
            if self.verbose:
                print("\n  Generating debug visualizations...")

            # Point cloud growth over time
            if len(self.reconstruction_history) > 0:
                camera_indices = [idx for idx, _ in self.reconstruction_history]
                point_clouds = [pc for _, pc in self.reconstruction_history]

                self.debug_viz.visualize_point_cloud_growth(
                    point_clouds, camera_indices
                )

            # Camera frustums
            self.debug_viz.visualize_camera_frustums(
                self.cameras, self.points_3d
            )

            # Save reconstruction report
            stats = {
                'n_images': self.n_images,
                'n_cameras': len(self.cameras),
                'n_points': len(self.points_3d),
                'stages': []
            }

            for i, (cam_idx, pc) in enumerate(self.reconstruction_history):
                prev_points = self.reconstruction_history[i-1][1] if i > 0 else np.array([])
                stats['stages'].append({
                    'name': f'Camera {cam_idx}',
                    'camera_idx': cam_idx,
                    'points_before': len(prev_points),
                    'points_after': len(pc),
                    'new_points': len(pc) - len(prev_points),
                    'status': 'Success'
                })

            self.debug_viz.save_reconstruction_report(stats)

            if self.verbose:
                print(f"  Debug visualizations saved to: {self.debug_viz.output_dir}")

        return {
            'cameras': self.cameras,
            'points_3d': self.points_3d,
            'point_colors': self.point_colors,
            'n_images': self.n_images,
            'n_reconstructed': len(self.reconstructed_images)
        }
