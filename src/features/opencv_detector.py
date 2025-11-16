"""
OpenCV-based feature detectors for SIFT, ORB, and Harris
This replaces custom implementations with optimized OpenCV versions
"""

import numpy as np
import cv2


def detect_sift_features(image, n_octaves=4, contrast_threshold=0.04, edge_threshold=10,
                        sigma=1.6, max_features=2000):
    """
    Detect SIFT features using OpenCV's optimized implementation

    Args:
        image: Input grayscale image (H, W) normalized to [0, 1]
        n_octaves: Number of octaves in the scale pyramid
        contrast_threshold: Threshold to filter weak features
        edge_threshold: Threshold to filter edge-like features
        sigma: Gaussian blur sigma for base image
        max_features: Maximum number of features to return

    Returns:
        keypoints: Array of (x, y) coordinates, shape (N, 2)
        descriptors: Array of 128-D SIFT descriptors, shape (N, 128)
    """
    # Convert to uint8 format expected by OpenCV
    image_uint8 = (image * 255).astype(np.uint8)

    # Create SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=max_features,
        nOctaveLayers=5,  # Standard SIFT uses 5 scales per octave
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )

    # Detect and compute
    kp, desc = sift.detectAndCompute(image_uint8, None)

    # Convert keypoints to array format (x, y)
    if len(kp) == 0:
        return np.array([]), np.array([])

    keypoints = np.array([[k.pt[0], k.pt[1]] for k in kp], dtype=np.float32)

    # Descriptors are already in the right format
    if desc is None:
        return np.array([]), np.array([])

    return keypoints, desc


def detect_orb_features(image, n_keypoints=2000, scale_factor=1.2, n_levels=8,
                       edge_threshold=31, fast_threshold=20):
    """
    Detect ORB features using OpenCV's optimized implementation

    Args:
        image: Input grayscale image (H, W) normalized to [0, 1]
        n_keypoints: Maximum number of keypoints to detect
        scale_factor: Pyramid decimation ratio
        n_levels: Number of pyramid levels
        edge_threshold: Size of border where features are not detected
        fast_threshold: FAST threshold

    Returns:
        keypoints: Array of (x, y) coordinates, shape (N, 2)
        descriptors: Array of 32-byte binary descriptors, shape (N, 32)
    """
    # Convert to uint8 format expected by OpenCV
    image_uint8 = (image * 255).astype(np.uint8)

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=n_keypoints,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=edge_threshold,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=fast_threshold
    )

    # Detect and compute
    kp, desc = orb.detectAndCompute(image_uint8, None)

    # Convert keypoints to array format (x, y)
    if len(kp) == 0:
        return np.array([]), np.array([])

    keypoints = np.array([[k.pt[0], k.pt[1]] for k in kp], dtype=np.float32)

    # Descriptors are already in the right format
    if desc is None:
        return np.array([]), np.array([])

    return keypoints, desc


def detect_harris_corners(image, k=0.04, threshold=0.01, window_size=3, sigma=1.0,
                         nms_size=3, max_corners=2000, use_multiscale=False,
                         use_adaptive_threshold=True):
    """
    Detect Harris corners using OpenCV's optimized implementation

    Args:
        image: Input grayscale image (H, W) normalized to [0, 1]
        k: Harris detector free parameter
        threshold: Threshold for corner response (relative to max response)
        window_size: Window size for computing gradients
        sigma: Gaussian blur sigma
        nms_size: Non-maximum suppression window size
        max_corners: Maximum number of corners to return
        use_multiscale: If True, detect at multiple scales (ignored for OpenCV)
        use_adaptive_threshold: If True, use adaptive thresholding

    Returns:
        corners: Array of (x, y) coordinates, shape (N, 2)
    """
    # Convert to uint8 format expected by OpenCV
    image_uint8 = (image * 255).astype(np.uint8)

    # Compute Harris corner response
    block_size = window_size
    ksize = 3  # Sobel aperture parameter

    # Apply Gaussian blur if sigma > 0
    if sigma > 0:
        ksize_blur = int(np.ceil(sigma * 3) * 2 + 1)  # Ensure odd size
        image_uint8 = cv2.GaussianBlur(image_uint8, (ksize_blur, ksize_blur), sigma)

    # Compute corner response using Harris detector
    dst = cv2.cornerHarris(image_uint8, block_size, ksize, k)

    # Apply adaptive thresholding if requested
    if use_adaptive_threshold:
        # Use percentile-based threshold instead of max-based
        threshold_value = np.percentile(dst[dst > 0], 99.5) if np.any(dst > 0) else 0
    else:
        # Use threshold relative to maximum response
        threshold_value = threshold * dst.max()

    # Find corners above threshold
    corner_mask = dst > threshold_value

    # Get coordinates
    y_coords, x_coords = np.where(corner_mask)
    corner_coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
    corner_responses = dst[corner_mask]

    if len(corner_coords) == 0:
        return np.array([])

    # Apply non-maximum suppression
    if nms_size > 1:
        corners = apply_nms(corner_coords, corner_responses, nms_size)
    else:
        # Sort by response and take top corners
        sorted_indices = np.argsort(corner_responses)[::-1]
        corners = corner_coords[sorted_indices]

    # Limit to max_corners
    if len(corners) > max_corners:
        corners = corners[:max_corners]

    return corners


def apply_nms(coords, responses, window_size):
    """
    Apply non-maximum suppression to corner detections

    Args:
        coords: Array of (x, y) coordinates, shape (N, 2)
        responses: Array of corner responses, shape (N,)
        window_size: Size of suppression window

    Returns:
        suppressed_coords: Array of (x, y) coordinates after NMS
    """
    # Sort by response (strongest first)
    sorted_indices = np.argsort(responses)[::-1]
    sorted_coords = coords[sorted_indices]

    # Keep track of which corners to keep
    keep = []
    suppressed = set()

    half_window = window_size // 2

    for i, coord in enumerate(sorted_coords):
        if i in suppressed:
            continue

        keep.append(coord)

        # Suppress nearby corners
        for j in range(i + 1, len(sorted_coords)):
            if j in suppressed:
                continue

            # Check if within suppression window
            dist = np.abs(sorted_coords[j] - coord).max()
            if dist <= half_window:
                suppressed.add(j)

    return np.array(keep, dtype=np.float32) if keep else np.array([])


def compute_descriptors(image, keypoints, patch_size=11, border=5):
    """
    Compute simple patch descriptors for Harris corners
    Uses normalized patches as descriptors (same as original implementation)

    Args:
        image: Input grayscale image (H, W) normalized to [0, 1]
        keypoints: Array of (x, y) coordinates, shape (N, 2)
        patch_size: Size of patch around each keypoint
        border: Minimum distance from image border

    Returns:
        descriptors: Array of patch descriptors, shape (M, patch_size^2)
                    where M <= N (some keypoints may be filtered)
    """
    if len(keypoints) == 0:
        return np.array([])

    H, W = image.shape
    half_patch = patch_size // 2

    # Filter keypoints too close to borders
    valid_mask = (
        (keypoints[:, 0] >= border + half_patch) &
        (keypoints[:, 0] < W - border - half_patch) &
        (keypoints[:, 1] >= border + half_patch) &
        (keypoints[:, 1] < H - border - half_patch)
    )
    valid_keypoints = keypoints[valid_mask]

    if len(valid_keypoints) == 0:
        return np.array([])

    # Extract patches
    descriptors = []
    for kp in valid_keypoints:
        x, y = int(kp[0]), int(kp[1])
        patch = image[y - half_patch:y + half_patch + 1,
                     x - half_patch:x + half_patch + 1]

        # Normalize patch
        patch_flat = patch.flatten()
        mean = patch_flat.mean()
        std = patch_flat.std()

        if std > 1e-6:  # Avoid division by zero
            patch_norm = (patch_flat - mean) / std
        else:
            patch_norm = patch_flat - mean

        descriptors.append(patch_norm)

    return np.array(descriptors, dtype=np.float32)
