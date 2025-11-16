"""
Phase 2 - Step 2.1: Harris Corner Detection
Detects corner points in images using the Harris corner detector algorithm.
"""

import numpy as np
from scipy import signal
from config import cfg


def detect_harris_corners(image: np.ndarray, k: float = None, threshold: float = None,
                         window_size: int = None, sigma: float = None,
                         nms_size: int = None, max_corners: int = None,
                         use_multiscale: bool = None, use_adaptive_threshold: bool = None) -> np.ndarray:
    """
    Detect Harris corners in an image with enhanced multi-scale and adaptive thresholding

    Args:
        image: Grayscale image (H, W) with values 0-1
        k: Harris corner response parameter (default: from config)
        threshold: Corner response threshold relative to max (default: from config)
        window_size: Window size for structure tensor (default: from config)
        sigma: Gaussian sigma for structure tensor (default: from config)
        nms_size: Non-maximum suppression window size (default: from config)
        max_corners: Maximum number of corners to return (default: from config)
        use_multiscale: Enable multi-scale detection (default: from config)
        use_adaptive_threshold: Enable adaptive thresholding (default: from config)

    Returns:
        corners: Array of corner coordinates (N, 2) as (x, y)
    """
    # Use config defaults if not provided
    if k is None:
        k = cfg.HARRIS_K
    if threshold is None:
        threshold = cfg.HARRIS_THRESHOLD
    if window_size is None:
        window_size = cfg.HARRIS_WINDOW_SIZE
    if sigma is None:
        sigma = cfg.HARRIS_SIGMA
    if nms_size is None:
        nms_size = cfg.NMS_WINDOW_SIZE
    if max_corners is None:
        max_corners = cfg.TARGET_CORNERS_MAX
    if use_multiscale is None:
        use_multiscale = cfg.USE_MULTISCALE_DETECTION
    if use_adaptive_threshold is None:
        use_adaptive_threshold = cfg.USE_ADAPTIVE_THRESHOLD

    if use_multiscale:
        # Multi-scale detection: detects both fine details (labels) and coarse features (bottle outlines)
        corners = detect_multiscale_corners(image, k, threshold, window_size, sigma,
                                           nms_size, max_corners, use_adaptive_threshold)
    else:
        # Original single-scale detection
        # Step 1: Compute image gradients
        Ix, Iy = compute_gradients(image)

        # Step 2: Compute structure tensor components
        M = compute_structure_tensor(Ix, Iy, window_size, sigma)

        # Step 3: Compute corner response
        R = corner_response(M, k)

        # Step 4: Threshold (adaptive or relative)
        if use_adaptive_threshold:
            R_threshold = compute_adaptive_threshold(R, threshold)
        else:
            R_threshold = threshold * R.max()

        corner_mask = R > R_threshold

        # Step 5: Non-maximum suppression
        corner_mask = non_maximum_suppression(R, nms_size) & corner_mask

        # Step 6: Extract corner coordinates
        y_coords, x_coords = np.where(corner_mask)
        corners = np.column_stack([x_coords, y_coords])

        # Step 7: Sort by response and limit number
        if len(corners) > max_corners:
            # Get response values for each corner
            responses = R[y_coords, x_coords]
            # Sort by descending response
            sorted_indices = np.argsort(responses)[::-1]
            corners = corners[sorted_indices[:max_corners]]

    return corners


def compute_gradients(image: np.ndarray) -> tuple:
    """
    Compute image gradients using Sobel operator

    Sobel kernels:
    Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
          [-2, 0, 2],           [ 0,  0,  0],
          [-1, 0, 1]]           [ 1,  2,  1]]

    Args:
        image: Grayscale image (H, W)

    Returns:
        Ix: Gradient in x direction (H, W)
        Iy: Gradient in y direction (H, W)
    """
    # Sobel kernels (manually defined)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # Convolve image with Sobel kernels
    Ix = signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
    Iy = signal.convolve2d(image, sobel_y, mode='same', boundary='symm')

    return Ix.astype(np.float32), Iy.astype(np.float32)


def compute_structure_tensor(Ix: np.ndarray, Iy: np.ndarray,
                             window_size: int = 5, sigma: float = 1.5) -> np.ndarray:
    """
    Compute structure tensor components with Gaussian weighting

    Structure tensor at each pixel:
    M = [Σ(Ix²)    Σ(IxIy)]
        [Σ(IxIy)   Σ(Iy²) ]

    where Σ is Gaussian weighted sum over local window

    Args:
        Ix: Gradient in x direction (H, W)
        Iy: Gradient in y direction (H, W)
        window_size: Window size for Gaussian (must be odd)
        sigma: Gaussian standard deviation

    Returns:
        M: Structure tensor (H, W, 3) where M[:,:,0]=Ixx, M[:,:,1]=Ixy, M[:,:,2]=Iyy
    """
    # Compute products of derivatives
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Create Gaussian window
    gaussian = create_gaussian_kernel(window_size, sigma)

    # Apply Gaussian weighting (convolution)
    Sxx = signal.convolve2d(Ixx, gaussian, mode='same', boundary='symm')
    Sxy = signal.convolve2d(Ixy, gaussian, mode='same', boundary='symm')
    Syy = signal.convolve2d(Iyy, gaussian, mode='same', boundary='symm')

    # Stack into single array
    M = np.stack([Sxx, Sxy, Syy], axis=2)

    return M.astype(np.float32)


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create 2D Gaussian kernel

    G(x, y) = (1 / 2πσ²) * exp(-(x² + y²) / 2σ²)

    Args:
        size: Kernel size (will be made odd)
        sigma: Standard deviation

    Returns:
        Gaussian kernel (size, size) normalized to sum to 1
    """
    # Ensure odd size
    size = size if size % 2 == 1 else size + 1

    # Create coordinate grid
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    # Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize to sum to 1
    kernel = kernel / kernel.sum()

    return kernel.astype(np.float32)


def corner_response(M: np.ndarray, k: float = 0.05) -> np.ndarray:
    """
    Compute Harris corner response

    R = det(M) - k * trace(M)²
    where:
        det(M) = λ1 * λ2 = Sxx * Syy - Sxy²
        trace(M) = λ1 + λ2 = Sxx + Syy

    Args:
        M: Structure tensor (H, W, 3)
        k: Harris parameter (typical: 0.04-0.06)

    Returns:
        R: Corner response (H, W)
    """
    Sxx = M[:, :, 0]
    Sxy = M[:, :, 1]
    Syy = M[:, :, 2]

    # Determinant
    det_M = Sxx * Syy - Sxy * Sxy

    # Trace
    trace_M = Sxx + Syy

    # Harris response
    R = det_M - k * (trace_M ** 2)

    return R.astype(np.float32)


def non_maximum_suppression(response: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply non-maximum suppression to corner response

    Keep only local maxima within window

    Args:
        response: Corner response map (H, W)
        window_size: Window size for NMS

    Returns:
        mask: Boolean mask of local maxima (H, W)
    """
    from scipy.ndimage import maximum_filter

    # Apply maximum filter
    max_response = maximum_filter(response, size=window_size)

    # Local maxima are where response equals max_response
    mask = (response == max_response) & (response > 0)

    return mask


def compute_adaptive_threshold(R: np.ndarray, base_threshold: float) -> float:
    """
    Compute adaptive threshold based on response distribution

    Instead of using a fixed percentage of max response, use statistics
    that are less sensitive to outliers (high-contrast labels)

    Args:
        R: Corner response map (H, W)
        base_threshold: Base threshold multiplier

    Returns:
        Adaptive threshold value
    """
    # Flatten and remove negative values
    responses = R[R > 0].flatten()

    if len(responses) == 0:
        return 0.0

    # Use percentile-based threshold (more robust than max)
    # This prevents high-contrast labels from dominating the threshold
    percentile_95 = np.percentile(responses, 95)
    median = np.median(responses)

    # Adaptive threshold: between median and 95th percentile
    # This captures both strong features (labels) and weaker features (bottle edges)
    adaptive_threshold = median + base_threshold * (percentile_95 - median)

    return adaptive_threshold


def detect_multiscale_corners(image: np.ndarray, k: float, threshold: float,
                               window_size: int, base_sigma: float,
                               nms_size: int, max_corners: int,
                               use_adaptive: bool) -> np.ndarray:
    """
    Detect Harris corners at multiple scales to capture both fine and coarse features

    Multi-scale detection helps detect:
    - Fine scale (sigma=1.0): High-contrast text on labels
    - Medium scale (sigma=1.5): Label edges and bottle cap details
    - Coarse scale (sigma=2.5): Bottle body edges and large structures

    Args:
        image: Grayscale image (H, W)
        k: Harris parameter
        threshold: Threshold value
        window_size: Structure tensor window size
        base_sigma: Base sigma for Gaussian
        nms_size: NMS window size
        max_corners: Maximum corners to return
        use_adaptive: Use adaptive thresholding

    Returns:
        corners: Combined corners from all scales (N, 2)
    """
    # Define multiple scales
    scales = [
        (1.0, 0.3),   # Fine scale: captures text and small details (30% weight)
        (1.5, 0.4),   # Medium scale: captures label edges (40% weight)
        (2.5, 0.3),   # Coarse scale: captures bottle outlines (30% weight)
    ]

    all_corners = []
    all_responses = []

    for sigma, weight in scales:
        # Compute gradients
        Ix, Iy = compute_gradients(image)

        # Compute structure tensor with this sigma
        M = compute_structure_tensor(Ix, Iy, window_size, sigma)

        # Compute corner response
        R = corner_response(M, k)

        # Threshold
        if use_adaptive:
            R_threshold = compute_adaptive_threshold(R, threshold)
        else:
            # For multiscale, use a more permissive threshold per scale
            R_threshold = threshold * 0.5 * R.max()

        corner_mask = R > R_threshold

        # Non-maximum suppression
        corner_mask = non_maximum_suppression(R, nms_size) & corner_mask

        # Extract corners
        y_coords, x_coords = np.where(corner_mask)
        corners = np.column_stack([x_coords, y_coords])

        # Get responses and weight them
        responses = R[y_coords, x_coords] * weight

        all_corners.append(corners)
        all_responses.append(responses)

    # Combine corners from all scales
    if len(all_corners) > 0:
        combined_corners = np.vstack([c for c in all_corners if len(c) > 0])
        combined_responses = np.concatenate([r for r in all_responses if len(r) > 0])
    else:
        return np.array([])

    # Remove duplicates (corners detected at multiple scales)
    unique_corners, unique_responses = remove_duplicate_corners(
        combined_corners, combined_responses, tolerance=3.0
    )

    # Sort by response and limit
    if len(unique_corners) > max_corners:
        sorted_indices = np.argsort(unique_responses)[::-1]
        unique_corners = unique_corners[sorted_indices[:max_corners]]

    return unique_corners


def remove_duplicate_corners(corners: np.ndarray, responses: np.ndarray,
                             tolerance: float = 3.0) -> tuple:
    """
    Remove duplicate corners that are very close to each other
    Keep the one with highest response

    Args:
        corners: Corner coordinates (N, 2)
        responses: Corner responses (N,)
        tolerance: Distance threshold for considering corners as duplicates

    Returns:
        unique_corners: Filtered corners (M, 2)
        unique_responses: Corresponding responses (M,)
    """
    if len(corners) == 0:
        return corners, responses

    # Sort by response (descending)
    sorted_indices = np.argsort(responses)[::-1]
    sorted_corners = corners[sorted_indices]
    sorted_responses = responses[sorted_indices]

    # Keep track of which corners to keep
    keep = np.ones(len(sorted_corners), dtype=bool)

    for i in range(len(sorted_corners)):
        if not keep[i]:
            continue

        # Check all remaining corners
        for j in range(i + 1, len(sorted_corners)):
            if not keep[j]:
                continue

            # Compute distance
            dist = np.linalg.norm(sorted_corners[i] - sorted_corners[j])

            # If too close, remove the weaker one (j, since sorted by response)
            if dist < tolerance:
                keep[j] = False

    return sorted_corners[keep], sorted_responses[keep]


if __name__ == "__main__":
    # Test Harris corner detection
    import sys
    sys.path.insert(0, '.')

    from src.preprocessing.image_loader import load_images
    import matplotlib.pyplot as plt

    print("Testing Harris Corner Detection")
    print("=" * 50)

    # Load test image
    images, metadata = load_images("data/scene1")
    test_image = images[0]

    print(f"\nTest image: {metadata[0]['filename']}")
    print(f"Size: {test_image.shape}")

    # Detect corners
    print("\nDetecting corners...")
    corners = detect_harris_corners(test_image)

    print(f"Detected {len(corners)} corners")
    print(f"First 10 corners (x, y):")
    for i, (x, y) in enumerate(corners[:10]):
        print(f"  {i+1}. ({x:.1f}, {y:.1f})")

    # Visualize (if matplotlib available)
    try:
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.imshow(test_image, cmap='gray')
        ax.plot(corners[:, 0], corners[:, 1], 'r+', markersize=4, markeredgewidth=0.5)
        ax.set_title(f'Harris Corners ({len(corners)} detected)')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('output/visualizations/harris_corners_test.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to output/visualizations/harris_corners_test.png")
    except Exception as e:
        print(f"\nCould not save visualization: {e}")
