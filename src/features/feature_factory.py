"""
Feature detector factory for selecting and creating different feature detectors
"""

import numpy as np
from .harris_detector import detect_harris_corners
from .sift_detector import detect_sift_features
from .orb_detector import detect_orb_features
from .descriptor import compute_descriptors
from config import cfg


class FeatureDetectorFactory:
    """
    Factory class for creating and using different feature detectors
    """

    @staticmethod
    def detect_and_compute(image, detector_type=None):
        """
        Detect features and compute descriptors using specified detector

        Args:
            image: Input grayscale image (H, W) normalized to [0, 1]
            detector_type: Type of detector ('harris', 'sift', 'orb')
                          If None, uses cfg.FEATURE_DETECTOR

        Returns:
            keypoints: Array of (x, y) coordinates, shape (N, 2)
            descriptors: Array of descriptors, shape (N, D)
            descriptor_type: String indicating descriptor type ('float' or 'binary')
        """
        if detector_type is None:
            detector_type = cfg.FEATURE_DETECTOR.lower()

        print(f"  Using {detector_type.upper()} feature detector")

        if detector_type == 'harris':
            # Use Harris corner detector with patch descriptors
            corners = detect_harris_corners(
                image,
                k=cfg.HARRIS_K,
                threshold=cfg.HARRIS_THRESHOLD,
                window_size=cfg.HARRIS_WINDOW_SIZE,
                sigma=cfg.HARRIS_SIGMA,
                nms_size=cfg.NMS_WINDOW_SIZE,
                max_corners=cfg.TARGET_CORNERS_MAX,
                use_multiscale=cfg.USE_MULTISCALE_DETECTION,
                use_adaptive_threshold=cfg.USE_ADAPTIVE_THRESHOLD
            )

            if len(corners) == 0:
                print("  Warning: No Harris corners detected!")
                return np.array([]), np.array([]), 'float'

            # Compute patch descriptors
            # Note: compute_descriptors already filters keypoints near borders
            # and returns both filtered keypoints and descriptors
            H, W = image.shape
            half_patch = cfg.PATCH_SIZE // 2
            border = cfg.DESCRIPTOR_BORDER

            # Filter keypoints that are too close to borders
            valid_mask = (
                (corners[:, 0] >= border + half_patch) &
                (corners[:, 0] < W - border - half_patch) &
                (corners[:, 1] >= border + half_patch) &
                (corners[:, 1] < H - border - half_patch)
            )
            valid_corners = corners[valid_mask]

            if len(valid_corners) == 0:
                print("  Warning: No valid Harris corners after border filtering!")
                return np.array([]), np.array([]), 'float'

            # Compute descriptors for valid corners
            descriptors = compute_descriptors(
                image,
                valid_corners,
                patch_size=cfg.PATCH_SIZE,
                border=0  # Already filtered, so no need for border
            )

            # Sanity check
            if len(descriptors) != len(valid_corners):
                print(f"  Warning: Descriptor count mismatch! corners={len(valid_corners)}, descriptors={len(descriptors)}")
                # Take minimum
                min_len = min(len(valid_corners), len(descriptors))
                keypoints = valid_corners[:min_len]
                descriptors = descriptors[:min_len]
            else:
                keypoints = valid_corners

            if len(keypoints) == 0:
                print("  Warning: No valid Harris descriptors computed!")
                return np.array([]), np.array([]), 'float'

            # Limit to target number
            if len(keypoints) > cfg.TARGET_FEATURES:
                keypoints = keypoints[:cfg.TARGET_FEATURES]
                descriptors = descriptors[:cfg.TARGET_FEATURES]

            return keypoints, descriptors, 'float'

        elif detector_type == 'sift':
            # Use SIFT detector
            keypoints, descriptors = detect_sift_features(
                image,
                n_octaves=cfg.SIFT_N_OCTAVES,
                contrast_threshold=cfg.SIFT_CONTRAST_THRESHOLD,
                edge_threshold=cfg.SIFT_EDGE_THRESHOLD,
                sigma=cfg.SIFT_SIGMA,
                max_features=cfg.TARGET_FEATURES
            )

            if len(keypoints) == 0:
                print("  Warning: No SIFT features detected!")
                return np.array([]), np.array([]), 'float'

            return keypoints, descriptors, 'float'

        elif detector_type == 'orb':
            # Use ORB detector
            keypoints, descriptors = detect_orb_features(
                image,
                n_keypoints=cfg.ORB_N_KEYPOINTS,
                scale_factor=cfg.ORB_SCALE_FACTOR,
                n_levels=cfg.ORB_N_LEVELS,
                edge_threshold=cfg.ORB_EDGE_THRESHOLD,
                fast_threshold=cfg.ORB_FAST_THRESHOLD
            )

            if len(keypoints) == 0:
                print("  Warning: No ORB features detected!")
                return np.array([]), np.array([]), 'binary'

            return keypoints, descriptors, 'binary'

        else:
            raise ValueError(f"Unknown detector type: {detector_type}. "
                           f"Must be 'harris', 'sift', or 'orb'")

    @staticmethod
    def get_descriptor_type(detector_type=None):
        """
        Get descriptor type for a given detector

        Args:
            detector_type: Type of detector ('harris', 'sift', 'orb')
                          If None, uses cfg.FEATURE_DETECTOR

        Returns:
            descriptor_type: 'float' or 'binary'
        """
        if detector_type is None:
            detector_type = cfg.FEATURE_DETECTOR.lower()

        if detector_type in ['harris', 'sift']:
            return 'float'
        elif detector_type == 'orb':
            return 'binary'
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")


def detect_and_compute_features(image, detector_type=None):
    """
    Convenience function to detect features and compute descriptors

    Args:
        image: Input grayscale image (H, W) normalized to [0, 1]
        detector_type: Type of detector ('harris', 'sift', 'orb')
                      If None, uses cfg.FEATURE_DETECTOR

    Returns:
        keypoints: Array of (x, y) coordinates, shape (N, 2)
        descriptors: Array of descriptors, shape (N, D)
        descriptor_type: String indicating descriptor type ('float' or 'binary')
    """
    return FeatureDetectorFactory.detect_and_compute(image, detector_type)
