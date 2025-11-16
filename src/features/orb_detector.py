"""
ORB (Oriented FAST and Rotated BRIEF) detector implementation

ORB is a fast and efficient alternative to SIFT:
- Much faster than SIFT
- Good rotation invariance
- Binary descriptors (256-bit = 32 bytes)
- Works well for real-time applications

ORB combines:
- FAST keypoint detector
- Harris corner measure for ranking
- Orientation from intensity centroid
- Rotated BRIEF descriptors
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class ORBDetector:
    """
    ORB feature detector implementation
    """

    def __init__(self, n_keypoints=2000, scale_factor=1.2, n_levels=8,
                 edge_threshold=31, fast_threshold=20):
        """
        Initialize ORB detector

        Args:
            n_keypoints: Maximum number of keypoints to retain
            scale_factor: Pyramid decimation ratio (>1)
            n_levels: Number of pyramid levels
            edge_threshold: Border size where features are not detected
            fast_threshold: Threshold for FAST detector
        """
        self.n_keypoints = n_keypoints
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.edge_threshold = edge_threshold
        self.fast_threshold = fast_threshold

        # Precompute BRIEF sampling pattern (256 pairs)
        self._init_brief_pattern()

    def _init_brief_pattern(self):
        """Initialize random sampling pattern for BRIEF descriptor"""
        np.random.seed(42)  # For reproducibility
        # Sample 256 pairs within a 31x31 patch
        self.pattern_size = 31
        self.n_pairs = 256

        # Generate random sampling points
        self.point_pairs = []
        for _ in range(self.n_pairs):
            x1 = np.random.randint(-15, 16)
            y1 = np.random.randint(-15, 16)
            x2 = np.random.randint(-15, 16)
            y2 = np.random.randint(-15, 16)
            self.point_pairs.append(((x1, y1), (x2, y2)))

    def detect_and_compute(self, image, max_features=None):
        """
        Detect ORB keypoints and compute descriptors

        Args:
            image: Input grayscale image (H, W) in range [0, 1]
            max_features: Maximum number of features (overrides n_keypoints if provided)

        Returns:
            keypoints: Array of shape (N, 2) with (x, y) coordinates
            descriptors: Array of shape (N, 32) with binary descriptors (256 bits)
        """
        if max_features is None:
            max_features = self.n_keypoints

        if image.ndim == 3:
            # Convert to grayscale
            image = np.mean(image, axis=2)

        # Normalize to [0, 255] uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Build image pyramid
        pyramid = self._build_pyramid(image)

        # Detect FAST keypoints at each level
        all_keypoints = []
        for level, img in enumerate(pyramid):
            # Detect FAST corners
            corners = self._detect_fast_corners(img, self.fast_threshold)

            if len(corners) == 0:
                continue

            # Compute Harris response for ranking
            responses = self._compute_harris_responses(img, corners)

            # Add scale information
            scale = self.scale_factor ** level
            for i, (x, y) in enumerate(corners):
                all_keypoints.append([
                    x * scale,  # x in original image
                    y * scale,  # y in original image
                    responses[i],  # Harris response
                    level  # Pyramid level
                ])

        if len(all_keypoints) == 0:
            print("  Warning: No ORB keypoints detected!")
            return np.array([]), np.array([])

        all_keypoints = np.array(all_keypoints)

        # Select top keypoints by Harris response
        if len(all_keypoints) > max_features:
            top_indices = np.argsort(all_keypoints[:, 2])[::-1][:max_features]
            all_keypoints = all_keypoints[top_indices]

        # Compute orientation and descriptors
        descriptors = []
        valid_keypoints = []

        for kp in all_keypoints:
            x, y, response, level = kp
            level = int(level)

            # Get image at appropriate scale
            img = pyramid[level]
            scale = self.scale_factor ** level
            x_scaled = int(x / scale)
            y_scaled = int(y / scale)

            # Compute orientation
            angle = self._compute_orientation(img, x_scaled, y_scaled)

            # Compute BRIEF descriptor
            desc = self._compute_brief_descriptor(img, x_scaled, y_scaled, angle)

            if desc is not None:
                descriptors.append(desc)
                valid_keypoints.append([x, y])

        if len(valid_keypoints) == 0:
            print("  Warning: No valid ORB descriptors computed!")
            return np.array([]), np.array([])

        keypoints = np.array(valid_keypoints)
        descriptors = np.array(descriptors)

        return keypoints, descriptors

    def _build_pyramid(self, image):
        """Build image pyramid"""
        pyramid = [image]

        for level in range(1, self.n_levels):
            # Compute scale
            scale = self.scale_factor ** level
            new_width = int(image.shape[1] / scale)
            new_height = int(image.shape[0] / scale)

            if new_width < 32 or new_height < 32:
                break

            # Downsample using simple decimation
            # First smooth to avoid aliasing
            smoothed = gaussian_filter(pyramid[-1].astype(np.float32), sigma=1.0)
            downsampled = smoothed[::2, ::2].astype(np.uint8)
            pyramid.append(downsampled)

        return pyramid

    def _detect_fast_corners(self, image, threshold):
        """
        Detect FAST (Features from Accelerated Segment Test) corners

        Simplified version: checks if 9 contiguous pixels in a circle are
        all brighter or darker than center
        """
        height, width = image.shape
        corners = []

        # FAST circle pattern (16 pixels around center at radius 3)
        circle_offsets = [
            (0, -3), (1, -3), (2, -2), (3, -1),
            (3, 0), (3, 1), (2, 2), (1, 3),
            (0, 3), (-1, 3), (-2, 2), (-3, 1),
            (-3, 0), (-3, -1), (-2, -2), (-1, -3)
        ]

        border = self.edge_threshold
        for y in range(border, height - border):
            for x in range(border, width - border):
                center_val = int(image[y, x])

                # Check circle pixels
                circle_vals = []
                valid = True
                for dx, dy in circle_offsets:
                    cx, cy = x + dx, y + dy
                    if 0 <= cx < width and 0 <= cy < height:
                        circle_vals.append(int(image[cy, cx]))
                    else:
                        valid = False
                        break

                if not valid:
                    continue

                circle_vals = np.array(circle_vals)

                # Check if 9 contiguous pixels are all brighter or darker
                brighter = (circle_vals > center_val + threshold).astype(int)
                darker = (circle_vals < center_val - threshold).astype(int)

                # Check for 9 contiguous pixels (with wrapping)
                extended_brighter = np.concatenate([brighter, brighter[:9]])
                extended_darker = np.concatenate([darker, darker[:9]])

                max_contiguous_brighter = 0
                max_contiguous_darker = 0

                for i in range(16):
                    if np.all(extended_brighter[i:i+9]):
                        max_contiguous_brighter = max(max_contiguous_brighter, 9)
                    if np.all(extended_darker[i:i+9]):
                        max_contiguous_darker = max(max_contiguous_darker, 9)

                if max_contiguous_brighter >= 9 or max_contiguous_darker >= 9:
                    corners.append([x, y])

        return np.array(corners) if corners else np.array([])

    def _compute_harris_responses(self, image, corners):
        """Compute Harris corner response for ranking keypoints"""
        responses = []

        # Compute gradients
        image_float = image.astype(np.float32)
        dy, dx = np.gradient(image_float)

        for x, y in corners:
            # Get local patch
            patch_size = 5
            half = patch_size // 2

            if (y < half or y >= image.shape[0] - half or
                x < half or x >= image.shape[1] - half):
                responses.append(0)
                continue

            dx_patch = dx[y-half:y+half+1, x-half:x+half+1]
            dy_patch = dy[y-half:y+half+1, x-half:x+half+1]

            # Structure tensor
            Ixx = np.sum(dx_patch * dx_patch)
            Iyy = np.sum(dy_patch * dy_patch)
            Ixy = np.sum(dx_patch * dy_patch)

            # Harris response
            det = Ixx * Iyy - Ixy * Ixy
            trace = Ixx + Iyy
            k = 0.04

            if trace > 0:
                response = det - k * trace * trace
            else:
                response = 0

            responses.append(response)

        return np.array(responses)

    def _compute_orientation(self, image, x, y, patch_radius=15):
        """
        Compute orientation using intensity centroid method

        Returns angle in radians
        """
        height, width = image.shape

        if (x < patch_radius or x >= width - patch_radius or
            y < patch_radius or y >= height - patch_radius):
            return 0.0

        # Extract patch
        patch = image[y - patch_radius:y + patch_radius + 1,
                     x - patch_radius:x + patch_radius + 1].astype(np.float32)

        # Compute moments
        m10 = 0.0
        m01 = 0.0

        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                intensity = patch[i, j]
                m10 += (j - patch_radius) * intensity
                m01 += (i - patch_radius) * intensity

        # Compute angle
        angle = np.arctan2(m01, m10)
        return angle

    def _compute_brief_descriptor(self, image, x, y, angle):
        """
        Compute rotated BRIEF descriptor (256 bits = 32 bytes)

        Returns:
            descriptor: Array of 32 uint8 values (256 bits packed)
        """
        height, width = image.shape
        patch_radius = 15

        if (x < patch_radius or x >= width - patch_radius or
            y < patch_radius or y >= height - patch_radius):
            return None

        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Extract smoothed patch for stability
        patch = image[y - patch_radius:y + patch_radius + 1,
                     x - patch_radius:x + patch_radius + 1].astype(np.float32)
        patch = gaussian_filter(patch, sigma=2.0)

        # Compute binary descriptor
        bits = []
        for (p1, p2) in self.point_pairs:
            # Rotate sampling points
            x1_rot = int(p1[0] * cos_a - p1[1] * sin_a + patch_radius)
            y1_rot = int(p1[0] * sin_a + p1[1] * cos_a + patch_radius)
            x2_rot = int(p2[0] * cos_a - p2[1] * sin_a + patch_radius)
            y2_rot = int(p2[0] * sin_a + p2[1] * cos_a + patch_radius)

            # Check bounds
            if (0 <= x1_rot < patch.shape[1] and 0 <= y1_rot < patch.shape[0] and
                0 <= x2_rot < patch.shape[1] and 0 <= y2_rot < patch.shape[0]):

                val1 = patch[y1_rot, x1_rot]
                val2 = patch[y2_rot, x2_rot]
                bits.append(1 if val1 < val2 else 0)
            else:
                bits.append(0)

        # Pack bits into bytes
        descriptor = np.zeros(32, dtype=np.uint8)
        for i in range(32):
            byte_val = 0
            for j in range(8):
                bit_idx = i * 8 + j
                if bit_idx < len(bits):
                    byte_val |= (bits[bit_idx] << j)
            descriptor[i] = byte_val

        return descriptor


def detect_orb_features(image, n_keypoints=2000, scale_factor=1.2, n_levels=8,
                       edge_threshold=31, fast_threshold=20):
    """
    Convenience function to detect ORB features

    Args:
        image: Input grayscale image
        n_keypoints: Maximum number of keypoints
        scale_factor: Pyramid scale factor
        n_levels: Number of pyramid levels
        edge_threshold: Border threshold
        fast_threshold: FAST detector threshold

    Returns:
        keypoints: Array of (x, y) coordinates
        descriptors: Array of binary descriptors (32 bytes each)
    """
    detector = ORBDetector(n_keypoints, scale_factor, n_levels,
                          edge_threshold, fast_threshold)
    return detector.detect_and_compute(image)
