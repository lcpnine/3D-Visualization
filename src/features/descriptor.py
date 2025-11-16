"""
Phase 2 - Step 2.2: Feature Descriptor Generation
Generates feature descriptors for detected keypoints using normalized patches.
"""

import numpy as np
from scipy.ndimage import map_coordinates
from config import cfg


def compute_descriptors(image: np.ndarray, keypoints: np.ndarray,
                       patch_size: int = None, border: int = None,
                       verbose: bool = False, progress_interval: int = 500) -> np.ndarray:
    """
    Compute descriptors for keypoints using normalized patches

    Args:
        image: Grayscale image (H, W) with values 0-1
        keypoints: Keypoint coordinates (N, 2) as (x, y)
        patch_size: Patch size (default: from config)
        border: Border pixels to exclude (default: from config)
        verbose: Show progress during computation
        progress_interval: Show progress every N keypoints

    Returns:
        descriptors: Descriptor matrix (M, patch_size²) where M <= N
                    (some keypoints may be excluded if near borders)
    """
    if patch_size is None:
        patch_size = cfg.PATCH_SIZE
    if border is None:
        border = cfg.DESCRIPTOR_BORDER

    H, W = image.shape
    half_patch = patch_size // 2

    # Filter keypoints that are too close to borders
    valid_mask = (
        (keypoints[:, 0] >= border + half_patch) &
        (keypoints[:, 0] < W - border - half_patch) &
        (keypoints[:, 1] >= border + half_patch) &
        (keypoints[:, 1] < H - border - half_patch)
    )

    valid_keypoints = keypoints[valid_mask]

    if len(valid_keypoints) == 0:
        return np.array([])

    # Extract and normalize descriptors
    descriptors = []
    total_kps = len(valid_keypoints)

    for i, kp in enumerate(valid_keypoints):
        # Show progress for large sets of keypoints
        if verbose and total_kps >= progress_interval and (i + 1) % progress_interval == 0:
            print(f" [{i+1}/{total_kps} descriptors]", end='', flush=True)

        x, y = kp
        patch = extract_patch(image, x, y, patch_size)
        desc = normalize_descriptor(patch)
        descriptors.append(desc)

    descriptors = np.array(descriptors, dtype=np.float32)

    return descriptors


def extract_patch(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """
    Extract patch centered at (x, y) using bilinear interpolation

    Args:
        image: Grayscale image (H, W)
        x: X coordinate (can be sub-pixel)
        y: Y coordinate (can be sub-pixel)
        size: Patch size (will extract size x size patch)

    Returns:
        patch: Extracted patch (size, size)
    """
    half_size = size // 2

    # Create patch coordinate grid (vectorized)
    patch_x = np.arange(size) - half_size
    patch_y = np.arange(size) - half_size

    # Create 2D meshgrid
    grid_x, grid_y = np.meshgrid(patch_x, patch_y)

    # Map to image coordinates
    img_x = x + grid_x
    img_y = y + grid_y

    # Use scipy's map_coordinates for fast bilinear interpolation
    # map_coordinates expects (row, col) which is (y, x)
    coordinates = np.array([img_y.flatten(), img_x.flatten()])

    # Sample using bilinear interpolation (order=1)
    patch = map_coordinates(image, coordinates, order=1, mode='constant', cval=0.0)

    # Reshape back to 2D patch
    patch = patch.reshape(size, size).astype(np.float32)

    return patch


def bilinear_sample(image: np.ndarray, x: float, y: float) -> float:
    """
    Sample image at (x, y) using bilinear interpolation

    Args:
        image: Grayscale image (H, W)
        x: X coordinate
        y: Y coordinate

    Returns:
        Interpolated pixel value
    """
    H, W = image.shape

    # Clip to valid range
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)

    # Integer and fractional parts
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, W - 1)
    y1 = min(y0 + 1, H - 1)

    fx = x - x0
    fy = y - y0

    # Bilinear interpolation
    value = (
        image[y0, x0] * (1 - fx) * (1 - fy) +
        image[y0, x1] * fx * (1 - fy) +
        image[y1, x0] * (1 - fx) * fy +
        image[y1, x1] * fx * fy
    )

    return value


def normalize_descriptor(patch: np.ndarray) -> np.ndarray:
    """
    Normalize patch descriptor

    Steps:
    1. Flatten patch to 1D vector
    2. Subtract mean
    3. Divide by standard deviation
    4. L2 normalization

    Args:
        patch: Image patch (size, size)

    Returns:
        Normalized descriptor (size²,)
    """
    # Flatten
    descriptor = patch.flatten()

    # Zero mean, unit variance
    mean = descriptor.mean()
    std = descriptor.std()

    if std > 1e-10:
        descriptor = (descriptor - mean) / std
    else:
        # Constant patch, set to zeros
        descriptor = np.zeros_like(descriptor)

    # L2 normalization
    norm = np.linalg.norm(descriptor)
    if norm > 1e-10:
        descriptor = descriptor / norm

    return descriptor.astype(np.float32)


def match_descriptor_distance(desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two descriptors

    Args:
        desc1: First descriptor
        desc2: Second descriptor

    Returns:
        Euclidean distance
    """
    return np.linalg.norm(desc1 - desc2)


if __name__ == "__main__":
    # Test descriptor generation
    import sys
    sys.path.insert(0, '.')

    from src.preprocessing.image_loader import load_images
    from src.features.harris_detector import detect_harris_corners

    print("Testing Feature Descriptor Generation")
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

    # Compute descriptors
    print("\nComputing descriptors...")
    descriptors = compute_descriptors(test_image, corners)

    print(f"Generated {len(descriptors)} descriptors")
    print(f"Descriptor shape: {descriptors.shape}")
    print(f"Descriptor dimension: {descriptors.shape[1]}")

    # Statistics
    print(f"\nDescriptor statistics:")
    print(f"  Mean: {descriptors.mean():.6f}")
    print(f"  Std: {descriptors.std():.6f}")
    print(f"  Min: {descriptors.min():.6f}")
    print(f"  Max: {descriptors.max():.6f}")

    # Check normalization
    norms = np.linalg.norm(descriptors, axis=1)
    print(f"\nL2 norms:")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std: {norms.std():.6f}")
    print(f"  Expected: 1.0")

    # Test descriptor matching
    print(f"\nTest descriptor distance:")
    d1 = descriptors[0]
    d2 = descriptors[1]
    dist = match_descriptor_distance(d1, d2)
    print(f"  Distance between descriptor 0 and 1: {dist:.6f}")

    # Self-distance should be 0
    self_dist = match_descriptor_distance(d1, d1)
    print(f"  Self-distance (should be 0): {self_dist:.10f}")
