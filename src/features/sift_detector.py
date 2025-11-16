"""
SIFT (Scale-Invariant Feature Transform) detector implementation

SIFT is one of the most robust feature detectors, invariant to:
- Scale changes
- Rotation
- Illumination changes
- Affine distortion

This implementation provides much better feature detection than Harris corners,
especially for objects with uniform textures or smooth surfaces.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class SIFTDetector:
    """
    SIFT feature detector implementation
    """

    def __init__(self, n_octaves=4, n_scales=5, contrast_threshold=0.04,
                 edge_threshold=10, sigma=1.6):
        """
        Initialize SIFT detector

        Args:
            n_octaves: Number of octaves in scale space
            n_scales: Number of scales per octave
            contrast_threshold: Threshold for filtering low-contrast keypoints
            edge_threshold: Threshold for filtering edge responses
            sigma: Base sigma for Gaussian smoothing
        """
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        self.k = 2 ** (1.0 / (n_scales - 3))  # Scale factor between scales

    def detect_and_compute(self, image, max_features=2000):
        """
        Detect SIFT keypoints and compute descriptors

        Args:
            image: Input grayscale image (H, W) in range [0, 1]
            max_features: Maximum number of features to return

        Returns:
            keypoints: Array of shape (N, 2) with (x, y) coordinates
            descriptors: Array of shape (N, 128) with SIFT descriptors
        """
        if image.ndim == 3:
            # Convert to grayscale if needed
            image = np.mean(image, axis=2)

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Build scale space
        gaussian_pyramid, dog_pyramid = self._build_scale_space(image)

        # Detect keypoints in DoG pyramid
        keypoints = self._detect_keypoints(dog_pyramid, gaussian_pyramid)

        if len(keypoints) == 0:
            print("  Warning: No SIFT keypoints detected!")
            return np.array([]), np.array([])

        # Compute descriptors
        descriptors = self._compute_descriptors(gaussian_pyramid, keypoints)

        # Filter out keypoints without descriptors
        valid_mask = ~np.isnan(descriptors[:, 0])
        keypoints = keypoints[valid_mask]
        descriptors = descriptors[valid_mask]

        if len(keypoints) == 0:
            print("  Warning: No valid SIFT descriptors computed!")
            return np.array([]), np.array([])

        # Select top features by response
        if len(keypoints) > max_features:
            # Use absolute DoG response as strength
            responses = np.abs(keypoints[:, 2])
            top_indices = np.argsort(responses)[::-1][:max_features]
            keypoints = keypoints[top_indices]
            descriptors = descriptors[top_indices]

        # Return only (x, y) coordinates
        return keypoints[:, :2], descriptors

    def _build_scale_space(self, image):
        """
        Build Gaussian and Difference-of-Gaussian pyramids

        Returns:
            gaussian_pyramid: List of octaves, each containing n_scales+3 Gaussian images
            dog_pyramid: List of octaves, each containing n_scales+2 DoG images
        """
        gaussian_pyramid = []
        dog_pyramid = []

        # Initial image is upsampled for better detection
        current_image = image

        for octave in range(self.n_octaves):
            # Build Gaussian images for this octave
            gaussian_octave = []
            sigma = self.sigma

            # First image of octave
            gaussian_octave.append(gaussian_filter(current_image, sigma))

            # Generate remaining scales
            for scale in range(1, self.n_scales + 3):
                sigma = self.sigma * (self.k ** scale)
                blurred = gaussian_filter(current_image, sigma)
                gaussian_octave.append(blurred)

            gaussian_pyramid.append(gaussian_octave)

            # Build DoG images for this octave
            dog_octave = []
            for i in range(len(gaussian_octave) - 1):
                dog = gaussian_octave[i + 1] - gaussian_octave[i]
                dog_octave.append(dog)

            dog_pyramid.append(dog_octave)

            # Downsample for next octave
            if octave < self.n_octaves - 1:
                # Use the n_scales-th Gaussian image for next octave
                current_image = gaussian_octave[self.n_scales]
                current_image = current_image[::2, ::2]  # Downsample by 2

        return gaussian_pyramid, dog_pyramid

    def _detect_keypoints(self, dog_pyramid, gaussian_pyramid):
        """
        Detect keypoints as local extrema in DoG pyramid

        Returns:
            keypoints: Array of shape (N, 4) with (x, y, response, octave_scale)
        """
        keypoints = []

        for octave_idx, dog_octave in enumerate(dog_pyramid):
            # Check scales 1 to n_scales (skip first and last)
            for scale_idx in range(1, len(dog_octave) - 1):
                # Get three consecutive DoG images
                prev_dog = dog_octave[scale_idx - 1]
                curr_dog = dog_octave[scale_idx]
                next_dog = dog_octave[scale_idx + 1]

                # Find local extrema
                height, width = curr_dog.shape

                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        # Get 3x3x3 neighborhood
                        val = curr_dog[y, x]

                        # Check if it's an extremum
                        neighborhood = [
                            prev_dog[y-1:y+2, x-1:x+2],
                            curr_dog[y-1:y+2, x-1:x+2],
                            next_dog[y-1:y+2, x-1:x+2]
                        ]
                        neighborhood = np.array(neighborhood)

                        # Check if it's a maximum or minimum
                        if val > 0:
                            is_extremum = (val >= neighborhood).all()
                        else:
                            is_extremum = (val <= neighborhood).all()

                        if is_extremum:
                            # Apply contrast threshold
                            if abs(val) < self.contrast_threshold:
                                continue

                            # Apply edge threshold (Harris corner measure)
                            if not self._is_corner(curr_dog, x, y):
                                continue

                            # Convert to original image coordinates
                            scale_factor = 2 ** octave_idx
                            x_orig = x * scale_factor
                            y_orig = y * scale_factor

                            keypoints.append([x_orig, y_orig, val, octave_idx * 10 + scale_idx])

        return np.array(keypoints) if keypoints else np.array([])

    def _is_corner(self, dog_image, x, y):
        """
        Check if keypoint is a corner (not an edge) using Harris corner measure
        """
        # Compute Hessian
        dxx = dog_image[y, x+1] + dog_image[y, x-1] - 2 * dog_image[y, x]
        dyy = dog_image[y+1, x] + dog_image[y-1, x] - 2 * dog_image[y, x]
        dxy = (dog_image[y+1, x+1] - dog_image[y+1, x-1] -
               dog_image[y-1, x+1] + dog_image[y-1, x-1]) / 4.0

        # Compute trace and determinant
        tr = dxx + dyy
        det = dxx * dyy - dxy * dxy

        # Avoid division by zero
        if abs(det) < 1e-10:
            return False

        # Edge threshold test
        ratio = tr * tr / det
        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold

        return ratio < threshold

    def _compute_descriptors(self, gaussian_pyramid, keypoints):
        """
        Compute 128-dimensional SIFT descriptors for each keypoint

        Args:
            gaussian_pyramid: Gaussian scale space
            keypoints: Array of keypoints (x, y, response, octave_scale)

        Returns:
            descriptors: Array of shape (N, 128)
        """
        descriptors = []

        for kp in keypoints:
            x, y, response, octave_scale = kp

            # Determine octave and scale
            octave_idx = int(octave_scale // 10)
            scale_idx = int(octave_scale % 10)

            # Get image at appropriate scale
            if octave_idx >= len(gaussian_pyramid) or scale_idx >= len(gaussian_pyramid[octave_idx]):
                descriptors.append(np.full(128, np.nan))
                continue

            image = gaussian_pyramid[octave_idx][scale_idx]

            # Convert coordinates to this scale
            scale_factor = 2 ** octave_idx
            x_scaled = int(x / scale_factor)
            y_scaled = int(y / scale_factor)

            # Compute descriptor
            desc = self._compute_single_descriptor(image, x_scaled, y_scaled)
            descriptors.append(desc)

        return np.array(descriptors)

    def _compute_single_descriptor(self, image, x, y, patch_size=16):
        """
        Compute single 128-dimensional SIFT descriptor

        Uses 4x4 grid of 8-bin orientation histograms
        """
        height, width = image.shape
        half_size = patch_size // 2

        # Check bounds
        if x < half_size or x >= width - half_size or y < half_size or y >= height - half_size:
            return np.full(128, np.nan)

        # Extract patch
        patch = image[y - half_size:y + half_size, x - half_size:x + half_size]

        # Compute gradients
        dy, dx = np.gradient(patch.astype(np.float32))
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
        orientation = (orientation + 360) % 360  # Ensure [0, 360)

        # Build 4x4 grid of 8-bin histograms
        descriptor = []
        bins = 8
        bin_width = 360.0 / bins

        for i in range(4):
            for j in range(4):
                # Extract 4x4 sub-region
                y_start = i * 4
                y_end = (i + 1) * 4
                x_start = j * 4
                x_end = (j + 1) * 4

                sub_mag = magnitude[y_start:y_end, x_start:x_end]
                sub_ori = orientation[y_start:y_end, x_start:x_end]

                # Compute histogram
                hist = np.zeros(bins)
                for bin_idx in range(bins):
                    bin_min = bin_idx * bin_width
                    bin_max = (bin_idx + 1) * bin_width

                    mask = (sub_ori >= bin_min) & (sub_ori < bin_max)
                    hist[bin_idx] = np.sum(sub_mag[mask])

                descriptor.extend(hist)

        descriptor = np.array(descriptor)

        # Normalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
            # Clip values and renormalize
            descriptor = np.clip(descriptor, 0, 0.2)
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm

        return descriptor


def detect_sift_features(image, n_octaves=4, contrast_threshold=0.04,
                        edge_threshold=10, sigma=1.6, max_features=2000):
    """
    Convenience function to detect SIFT features

    Args:
        image: Input grayscale image
        n_octaves: Number of octaves in scale space
        contrast_threshold: Contrast threshold for filtering
        edge_threshold: Edge threshold for filtering
        sigma: Base Gaussian sigma
        max_features: Maximum number of features to return

    Returns:
        keypoints: Array of (x, y) coordinates
        descriptors: Array of 128-D descriptors
    """
    detector = SIFTDetector(n_octaves, 5, contrast_threshold, edge_threshold, sigma)
    return detector.detect_and_compute(image, max_features)
