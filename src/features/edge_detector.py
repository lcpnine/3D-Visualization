"""
Enhanced Edge Detection Module
Provides Canny edge detection and edge-based corner extraction for detecting
object outlines (e.g., bottle bodies) that have low texture but clear boundaries.
"""

import numpy as np
from scipy import signal, ndimage
from config import cfg


def detect_canny_edges(image: np.ndarray, low_threshold: float = None,
                       high_threshold: float = None, sigma: float = None) -> np.ndarray:
    """
    Detect edges using Canny edge detector

    Canny edge detection is particularly effective for detecting object boundaries
    like bottle outlines, even when the surface has uniform color/texture.

    Args:
        image: Grayscale image (H, W) with values 0-1
        low_threshold: Low threshold for hysteresis (default: from config)
        high_threshold: High threshold for hysteresis (default: from config)
        sigma: Gaussian sigma for smoothing (default: from config)

    Returns:
        edges: Binary edge map (H, W)
    """
    if low_threshold is None:
        low_threshold = cfg.CANNY_LOW_THRESHOLD
    if high_threshold is None:
        high_threshold = cfg.CANNY_HIGH_THRESHOLD
    if sigma is None:
        sigma = cfg.CANNY_SIGMA

    # Step 1: Gaussian smoothing
    smoothed = gaussian_filter(image, sigma)

    # Step 2: Compute gradients
    from src.features.harris_detector import compute_gradients
    Ix, Iy = compute_gradients(smoothed)

    # Step 3: Gradient magnitude and direction
    magnitude = np.sqrt(Ix**2 + Iy**2)
    direction = np.arctan2(Iy, Ix)

    # Step 4: Non-maximum suppression
    suppressed = canny_non_maximum_suppression(magnitude, direction)

    # Step 5: Double threshold
    edges = double_threshold_hysteresis(suppressed, low_threshold, high_threshold)

    return edges


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian filtering to image

    Args:
        image: Input image (H, W)
        sigma: Gaussian standard deviation

    Returns:
        Filtered image (H, W)
    """
    # Create Gaussian kernel
    from src.features.harris_detector import create_gaussian_kernel
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # Cover 3 sigma on each side
    kernel = create_gaussian_kernel(kernel_size, sigma)

    # Apply filter
    filtered = signal.convolve2d(image, kernel, mode='same', boundary='symm')

    return filtered.astype(np.float32)


def canny_non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Non-maximum suppression for Canny edge detection

    Thin edges to single-pixel width by suppressing non-maximum gradients
    perpendicular to edge direction

    Args:
        magnitude: Gradient magnitude (H, W)
        direction: Gradient direction in radians (H, W)

    Returns:
        Suppressed magnitude (H, W)
    """
    H, W = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # Convert angles to 0-180 degrees
    angle = np.rad2deg(direction) % 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            # Get neighboring pixels in gradient direction
            q = 255
            r = 255

            # Angle 0 (horizontal edge)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # Angle 45 (diagonal edge)
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # Angle 90 (vertical edge)
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # Angle 135 (diagonal edge)
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            # Keep only local maxima
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]

    return suppressed


def double_threshold_hysteresis(magnitude: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Apply double threshold and hysteresis for edge detection

    Args:
        magnitude: Edge magnitude (H, W)
        low: Low threshold (relative to max)
        high: High threshold (relative to max)

    Returns:
        Binary edge map (H, W) with values 0 or 1
    """
    max_val = magnitude.max()
    low_threshold = low * max_val
    high_threshold = high * max_val

    # Strong edges
    strong_edges = magnitude >= high_threshold
    # Weak edges
    weak_edges = (magnitude >= low_threshold) & (magnitude < high_threshold)

    # Hysteresis: keep weak edges connected to strong edges
    # Use connected component labeling
    labels, num_labels = ndimage.label(weak_edges | strong_edges)

    # Find which labels are connected to strong edges
    strong_labels = np.unique(labels[strong_edges])

    # Create final edge map
    edges = np.zeros_like(magnitude, dtype=np.uint8)
    for label in strong_labels:
        if label > 0:  # Skip background
            edges[labels == label] = 1

    return edges


def extract_edge_corners(edges: np.ndarray, corner_threshold: int = None) -> np.ndarray:
    """
    Extract corner points from edge map

    Corners are points where multiple edge directions meet,
    useful for detecting bottle corners and outline features

    Args:
        edges: Binary edge map (H, W)
        corner_threshold: Minimum number of edge neighbors (default: from config)

    Returns:
        corners: Corner coordinates (N, 2) as (x, y)
    """
    if corner_threshold is None:
        corner_threshold = cfg.EDGE_CORNER_THRESHOLD

    H, W = edges.shape
    corners = []

    # Use a 3x3 neighborhood to find junctions
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if edges[i, j] == 0:
                continue

            # Count edge pixels in 3x3 neighborhood
            neighborhood = edges[i-1:i+2, j-1:j+2]
            edge_count = neighborhood.sum()

            # Corner if multiple edges meet (3 or more edge pixels including center)
            if edge_count >= corner_threshold:
                corners.append([j, i])  # (x, y)

    return np.array(corners, dtype=np.float32) if corners else np.array([])


def detect_edge_based_features(image: np.ndarray, max_corners: int = None) -> np.ndarray:
    """
    Detect features by combining Canny edges with corner extraction

    This is complementary to Harris corner detection and helps detect
    features on smooth surfaces with clear boundaries (like bottle bodies)

    Args:
        image: Grayscale image (H, W)
        max_corners: Maximum corners to return (default: from config)

    Returns:
        corners: Edge-based corner coordinates (N, 2) as (x, y)
    """
    if max_corners is None:
        max_corners = cfg.EDGE_CORNERS_MAX

    # Detect edges
    edges = detect_canny_edges(image)

    # Extract corners from edges
    corners = extract_edge_corners(edges)

    # Limit number of corners
    if len(corners) > max_corners:
        # Sample uniformly or use spacing
        # For now, just take first N
        corners = corners[:max_corners]

    return corners


def combine_harris_and_edge_features(harris_corners: np.ndarray,
                                     edge_corners: np.ndarray,
                                     max_total: int = None,
                                     min_distance: float = 5.0) -> np.ndarray:
    """
    Combine Harris corners and edge-based corners

    This provides comprehensive coverage:
    - Harris: texture-rich areas (labels with text)
    - Edge: texture-poor areas with boundaries (bottle body outlines)

    Args:
        harris_corners: Harris corner coordinates (N1, 2)
        edge_corners: Edge-based corner coordinates (N2, 2)
        max_total: Maximum total corners (default: from config)
        min_distance: Minimum distance between combined corners

    Returns:
        combined_corners: Combined corner set (M, 2)
    """
    if max_total is None:
        max_total = cfg.TARGET_CORNERS_MAX

    # If either set is empty, return the other
    if len(harris_corners) == 0:
        return edge_corners[:max_total]
    if len(edge_corners) == 0:
        return harris_corners[:max_total]

    # Combine both sets
    all_corners = np.vstack([harris_corners, edge_corners])

    # Remove corners that are too close to each other
    # Give priority to Harris corners (they come first)
    unique_corners = remove_nearby_corners(all_corners, min_distance)

    # Limit total number
    if len(unique_corners) > max_total:
        unique_corners = unique_corners[:max_total]

    return unique_corners


def remove_nearby_corners(corners: np.ndarray, min_distance: float) -> np.ndarray:
    """
    Remove corners that are too close to each other
    Keep earlier corners in the list (priority order)

    Args:
        corners: Corner coordinates (N, 2)
        min_distance: Minimum allowed distance

    Returns:
        filtered_corners: Filtered corners (M, 2)
    """
    if len(corners) == 0:
        return corners

    keep = np.ones(len(corners), dtype=bool)

    for i in range(len(corners)):
        if not keep[i]:
            continue

        # Check all subsequent corners
        for j in range(i + 1, len(corners)):
            if not keep[j]:
                continue

            # Compute distance
            dist = np.linalg.norm(corners[i] - corners[j])

            # If too close, remove the later one
            if dist < min_distance:
                keep[j] = False

    return corners[keep]


if __name__ == "__main__":
    # Test edge detection
    import sys
    sys.path.insert(0, '.')

    from src.preprocessing.image_loader import load_images
    from src.features.harris_detector import detect_harris_corners
    import matplotlib.pyplot as plt

    print("Testing Enhanced Edge Detection")
    print("=" * 50)

    # Load test image
    images, metadata = load_images("data/scene1")
    test_image = images[0]

    print(f"\nTest image: {metadata[0]['filename']}")
    print(f"Size: {test_image.shape}")

    # Detect Canny edges
    print("\nDetecting Canny edges...")
    edges = detect_canny_edges(test_image)
    print(f"Edge pixels: {edges.sum()}")

    # Extract edge corners
    print("\nExtracting edge-based corners...")
    edge_corners = detect_edge_based_features(test_image)
    print(f"Edge corners detected: {len(edge_corners)}")

    # Detect Harris corners for comparison
    print("\nDetecting Harris corners...")
    harris_corners = detect_harris_corners(test_image)
    print(f"Harris corners detected: {len(harris_corners)}")

    # Combine features
    print("\nCombining features...")
    combined = combine_harris_and_edge_features(harris_corners, edge_corners)
    print(f"Combined corners: {len(combined)}")

    # Visualize
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # Original image
        axes[0, 0].imshow(test_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Canny edges
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title(f'Canny Edges ({edges.sum()} pixels)')
        axes[0, 1].axis('off')

        # Harris corners
        axes[1, 0].imshow(test_image, cmap='gray')
        axes[1, 0].plot(harris_corners[:, 0], harris_corners[:, 1], 'r+', markersize=4)
        axes[1, 0].set_title(f'Harris Corners ({len(harris_corners)})')
        axes[1, 0].axis('off')

        # Combined features
        axes[1, 1].imshow(test_image, cmap='gray')
        axes[1, 1].plot(harris_corners[:, 0], harris_corners[:, 1], 'r+', markersize=4, label='Harris')
        axes[1, 1].plot(edge_corners[:, 0], edge_corners[:, 1], 'b+', markersize=4, label='Edge')
        axes[1, 1].set_title(f'Combined Features ({len(combined)})')
        axes[1, 1].legend()
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('output/visualizations/edge_detection_test.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to output/visualizations/edge_detection_test.png")
    except Exception as e:
        print(f"\nCould not save visualization: {e}")
