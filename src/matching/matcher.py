"""
Phase 3 - Step 3.1: Brute-force Descriptor Matching
Finds feature correspondences between two images using descriptor matching.
"""

import numpy as np
from typing import List, Tuple
from config import cfg


def match_descriptors(desc1: np.ndarray, desc2: np.ndarray,
                     ratio_threshold: float = None,
                     use_symmetric: bool = None,
                     debug: bool = False,
                     descriptor_type: str = 'float',
                     use_crosscheck: bool = True,
                     distance_threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match descriptors between two images using brute-force matching with ratio test
    Improved version based on OpenCV's BFMatcher internal implementation

    Args:
        desc1: Descriptors from image 1 (N1, D)
        desc2: Descriptors from image 2 (N2, D)
        ratio_threshold: Lowe's ratio test threshold (default: from config)
        use_symmetric: Use symmetric matching (default: from config)
        debug: Print debug information (default: False)
        descriptor_type: Type of descriptors ('float' for L2, 'binary' for Hamming)
        use_crosscheck: Use OpenCV-style cross-check (mutual consistency)
        distance_threshold: Maximum allowed match distance (optional)

    Returns:
        matches: Array of matches (M, 2) where each row is [idx1, idx2]
        distances: Array of match distances (M,)
    """
    if ratio_threshold is None:
        ratio_threshold = cfg.RATIO_TEST_THRESHOLD
    if use_symmetric is None:
        use_symmetric = cfg.USE_SYMMETRIC_MATCHING

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(desc1, desc2, descriptor_type)

    if debug:
        print(f"    Distance matrix shape: {dist_matrix.shape}")
        print(f"    Distance stats - min: {dist_matrix.min():.4f}, max: {dist_matrix.max():.4f}, mean: {dist_matrix.mean():.4f}")

    # Forward matching (image 1 to image 2)
    matches_12, distances_12 = match_with_ratio_test(dist_matrix, ratio_threshold, debug=debug)

    if debug:
        print(f"    Forward matches (after ratio test): {len(matches_12)}")

    # Cross-check: OpenCV-style mutual consistency check
    if use_crosscheck or use_symmetric:
        # Backward matching (image 2 to image 1)
        dist_matrix_T = dist_matrix.T
        matches_21, distances_21 = match_with_ratio_test(dist_matrix_T, ratio_threshold, debug=False)

        if debug:
            print(f"    Backward matches (after ratio test): {len(matches_21)}")

        # Keep only cross-consistent matches (i->j and j->i)
        matches, distances = cross_check_matching(matches_12, distances_12,
                                                  matches_21, distances_21)

        if debug:
            print(f"    Cross-checked matches: {len(matches)}")
    else:
        matches = matches_12
        distances = distances_12

    # Additional distance threshold filtering (OpenCV-style)
    if distance_threshold is not None and len(matches) > 0:
        valid_mask = distances < distance_threshold
        matches = matches[valid_mask]
        distances = distances[valid_mask]

        if debug:
            print(f"    After distance threshold filtering: {len(matches)}")

    return matches, distances


def compute_distance_matrix(desc1: np.ndarray, desc2: np.ndarray,
                           descriptor_type: str = 'float') -> np.ndarray:
    """
    Compute pairwise distance matrix between descriptors

    For float descriptors: Euclidean distance
    For binary descriptors: Hamming distance

    Args:
        desc1: Descriptors from image 1 (N1, D)
        desc2: Descriptors from image 2 (N2, D)
        descriptor_type: Type of descriptors ('float' or 'binary')

    Returns:
        dist_matrix: Distance matrix (N1, N2)
    """
    if descriptor_type == 'binary':
        # Hamming distance for binary descriptors
        # Unpack bits and count differences
        N1, N2 = len(desc1), len(desc2)
        dist_matrix = np.zeros((N1, N2), dtype=np.float32)

        # Efficient Hamming distance computation using XOR and popcount
        for i in range(N1):
            # XOR to find differing bits
            xor_result = np.bitwise_xor(desc1[i:i+1], desc2)
            # Count set bits in each byte
            dist_matrix[i] = np.unpackbits(xor_result, axis=1).sum(axis=1)

        return dist_matrix

    else:
        # Euclidean distance for float descriptors
        # Efficient computation using: ||a - b||² = ||a||² + ||b||² - 2*a·b

        # Compute squared norms
        norm1_sq = np.sum(desc1 ** 2, axis=1, keepdims=True)  # (N1, 1)
        norm2_sq = np.sum(desc2 ** 2, axis=1, keepdims=True)  # (N2, 1)

        # Compute dot products
        dot_products = desc1 @ desc2.T  # (N1, N2)

        # Compute squared distances
        dist_sq = norm1_sq + norm2_sq.T - 2 * dot_products

        # Ensure non-negative (numerical stability)
        dist_sq = np.maximum(dist_sq, 0)

        # Take square root to get Euclidean distance
        dist_matrix = np.sqrt(dist_sq)

        return dist_matrix.astype(np.float32)


def match_with_ratio_test(dist_matrix: np.ndarray,
                          ratio_threshold: float = 0.8,
                          debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform matching with Lowe's ratio test

    For each descriptor in image 1:
    - Find the two nearest neighbors in image 2
    - Accept match only if distance to 1st NN / distance to 2nd NN < threshold

    Args:
        dist_matrix: Distance matrix (N1, N2)
        ratio_threshold: Ratio test threshold (typically 0.75-0.8)
        debug: Print debug information (default: False)

    Returns:
        matches: Array of matches (M, 2) where each row is [idx1, idx2]
        distances: Array of match distances (M,)
    """
    N1, N2 = dist_matrix.shape

    if N2 < 2:
        # Not enough descriptors for ratio test
        return np.array([]), np.array([])

    # Find two nearest neighbors for each descriptor in image 1
    # argsort returns indices that would sort the array
    sorted_indices = np.argsort(dist_matrix, axis=1)  # (N1, N2)

    # Get indices and distances of 1st and 2nd nearest neighbors
    nn1_indices = sorted_indices[:, 0]  # (N1,)
    nn2_indices = sorted_indices[:, 1]  # (N1,)

    nn1_distances = dist_matrix[np.arange(N1), nn1_indices]  # (N1,)
    nn2_distances = dist_matrix[np.arange(N1), nn2_indices]  # (N1,)

    # Ratio test
    ratios = nn1_distances / (nn2_distances + 1e-10)  # Add small epsilon to avoid division by zero
    valid_mask = ratios < ratio_threshold

    if debug:
        print(f"    Ratio test stats - min: {ratios.min():.4f}, max: {ratios.max():.4f}, mean: {ratios.mean():.4f}")
        print(f"    Passed ratio test: {valid_mask.sum()}/{len(valid_mask)} ({100*valid_mask.sum()/len(valid_mask):.1f}%)")

    # Extract valid matches
    idx1 = np.where(valid_mask)[0]
    idx2 = nn1_indices[valid_mask]

    matches = np.column_stack([idx1, idx2])
    distances = nn1_distances[valid_mask]

    return matches, distances


def symmetric_matching(matches_12: np.ndarray, distances_12: np.ndarray,
                      matches_21: np.ndarray, distances_21: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only symmetric matches (mutual nearest neighbors)

    A match (i, j) is symmetric if:
    - i is matched to j in forward matching (image 1 -> image 2)
    - j is matched to i in backward matching (image 2 -> image 1)

    Args:
        matches_12: Forward matches (M1, 2)
        distances_12: Forward match distances (M1,)
        matches_21: Backward matches (M2, 2)
        distances_21: Backward match distances (M2,)

    Returns:
        symmetric_matches: Symmetric matches (M, 2)
        symmetric_distances: Symmetric match distances (M,)
    """
    # Create dictionaries for fast lookup
    forward_dict = {(m[0], m[1]): d for m, d in zip(matches_12, distances_12)}
    backward_dict = {(m[1], m[0]): d for m, d in zip(matches_21, distances_21)}

    # Find symmetric matches
    symmetric_matches = []
    symmetric_distances = []

    for (i, j), dist in forward_dict.items():
        if (i, j) in backward_dict:
            symmetric_matches.append([i, j])
            symmetric_distances.append(dist)

    if len(symmetric_matches) == 0:
        return np.array([]), np.array([])

    symmetric_matches = np.array(symmetric_matches)
    symmetric_distances = np.array(symmetric_distances)

    return symmetric_matches, symmetric_distances


def cross_check_matching(matches_12: np.ndarray, distances_12: np.ndarray,
                         matches_21: np.ndarray, distances_21: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV-style cross-check: Keep only matches where (i,j) in forward matches AND (j,i) in backward matches
    This implements the crossCheck parameter from OpenCV's BFMatcher

    The difference from symmetric_matching is that cross-check ensures mutual best matches:
    - i's best match is j, AND j's best match is i

    Args:
        matches_12: Forward matches (M1, 2) where each row is [idx1, idx2]
        distances_12: Forward match distances (M1,)
        matches_21: Backward matches (M2, 2) where each row is [idx2, idx1]
        distances_21: Backward match distances (M2,)

    Returns:
        cross_checked_matches: Cross-checked matches (M, 2)
        cross_checked_distances: Cross-checked match distances (M,)
    """
    if len(matches_12) == 0 or len(matches_21) == 0:
        return np.array([]), np.array([])

    # Build lookup set for backward matches (store as (j, i) pairs)
    backward_set = set()
    for match in matches_21:
        # matches_21 has format [idx2, idx1], so we swap to get (idx1, idx2) format
        backward_set.add((match[1], match[0]))

    # Filter forward matches
    cross_checked_matches = []
    cross_checked_distances = []

    for idx, match in enumerate(matches_12):
        i, j = match[0], match[1]
        # Check if (i, j) exists in backward matches
        if (i, j) in backward_set:
            cross_checked_matches.append([i, j])
            cross_checked_distances.append(distances_12[idx])

    if len(cross_checked_matches) == 0:
        return np.array([]), np.array([])

    return np.array(cross_checked_matches), np.array(cross_checked_distances)


def filter_matches_by_distance(matches: np.ndarray, distances: np.ndarray,
                               max_distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter matches by maximum distance threshold

    Args:
        matches: Matches (M, 2)
        distances: Match distances (M,)
        max_distance: Maximum allowed distance

    Returns:
        filtered_matches: Filtered matches
        filtered_distances: Filtered distances
    """
    valid_mask = distances < max_distance
    return matches[valid_mask], distances[valid_mask]


def get_matched_points(keypoints1: np.ndarray, keypoints2: np.ndarray,
                      matches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract matched point coordinates from keypoints and matches

    Args:
        keypoints1: Keypoints from image 1 (N1, 2)
        keypoints2: Keypoints from image 2 (N2, 2)
        matches: Matches (M, 2) where each row is [idx1, idx2]

    Returns:
        points1: Matched points from image 1 (M, 2)
        points2: Matched points from image 2 (M, 2)
    """
    idx1 = matches[:, 0]
    idx2 = matches[:, 1]

    points1 = keypoints1[idx1]
    points2 = keypoints2[idx2]

    return points1, points2


if __name__ == "__main__":
    # Test descriptor matching
    import sys
    sys.path.insert(0, '.')

    from src.preprocessing.image_loader import load_images
    from src.features.harris_detector import detect_harris_corners
    from src.features.descriptor import compute_descriptors

    print("Testing Descriptor Matching")
    print("=" * 50)

    # Load images
    print("\nLoading images...")
    images, metadata = load_images("data/scene1")

    # Process first two images
    img1, img2 = images[0], images[1]
    print(f"\nImage 1: {metadata[0]['filename']}")
    print(f"Image 2: {metadata[1]['filename']}")

    # Detect features
    print("\nDetecting features...")
    corners1 = detect_harris_corners(img1)
    corners2 = detect_harris_corners(img2)
    print(f"Corners in image 1: {len(corners1)}")
    print(f"Corners in image 2: {len(corners2)}")

    # Compute descriptors
    print("\nComputing descriptors...")
    desc1 = compute_descriptors(img1, corners1)
    desc2 = compute_descriptors(img2, corners2)
    print(f"Descriptors in image 1: {len(desc1)}")
    print(f"Descriptors in image 2: {len(desc2)}")

    # Match descriptors
    print("\nMatching descriptors...")
    matches, distances = match_descriptors(desc1, desc2)

    print(f"\nFound {len(matches)} matches")
    print(f"Match distance statistics:")
    print(f"  Mean: {distances.mean():.4f}")
    print(f"  Std: {distances.std():.4f}")
    print(f"  Min: {distances.min():.4f}")
    print(f"  Max: {distances.max():.4f}")

    # Show first 10 matches
    print(f"\nFirst 10 matches:")
    for i, (idx1, idx2) in enumerate(matches[:10]):
        print(f"  {i+1}. ({idx1:4d}, {idx2:4d}) - distance: {distances[i]:.4f}")

    # Get matched points
    points1, points2 = get_matched_points(corners1, corners2, matches)
    print(f"\nMatched points shape: {points1.shape}")
    print(f"First matched point pair:")
    print(f"  Image 1: ({points1[0, 0]:.1f}, {points1[0, 1]:.1f})")
    print(f"  Image 2: ({points2[0, 0]:.1f}, {points2[0, 1]:.1f})")
