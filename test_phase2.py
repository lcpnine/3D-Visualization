"""
Test script for Phase 2: Feature Detection
"""

import sys
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import image_loader
from src.features import harris_detector, descriptor
from config import cfg

def test_phase2():
    print("=" * 70)
    print("Testing Phase 2: Feature Detection")
    print("=" * 70)

    # Load images
    print("\n[Preparation] Loading images...")
    print("-" * 70)
    images, metadata = image_loader.load_images("data/scene1")
    print(f"Loaded {len(images)} images")

    # Test on first image
    test_image = images[0]
    print(f"\nTesting on: {metadata[0]['filename']}")
    print(f"Image size: {test_image.shape}")

    # Test 1: Harris Corner Detection
    print("\n" + "=" * 70)
    print("[Test 1] Harris Corner Detection")
    print("-" * 70)

    try:
        start_time = time.time()
        corners = harris_detector.detect_harris_corners(test_image)
        elapsed = time.time() - start_time

        print(f"\nDetected {len(corners)} corners in {elapsed:.2f} seconds")
        print(f"First 10 corners (x, y):")
        for i, (x, y) in enumerate(corners[:10]):
            print(f"  {i+1}. ({x:.1f}, {y:.1f})")

        # Statistics
        print(f"\nCorner statistics:")
        print(f"  X range: [{corners[:, 0].min():.1f}, {corners[:, 0].max():.1f}]")
        print(f"  Y range: [{corners[:, 1].min():.1f}, {corners[:, 1].max():.1f}]")

        # Visualize intermediate results
        print(f"\nComputing intermediate results for analysis...")
        Ix, Iy = harris_detector.compute_gradients(test_image)
        print(f"  Gradient Ix range: [{Ix.min():.3f}, {Ix.max():.3f}]")
        print(f"  Gradient Iy range: [{Iy.min():.3f}, {Iy.max():.3f}]")

        M = harris_detector.compute_structure_tensor(Ix, Iy)
        print(f"  Structure tensor shape: {M.shape}")

        R = harris_detector.corner_response(M)
        print(f"  Corner response range: [{R.min():.6f}, {R.max():.6f}]")
        print(f"  Number of positive responses: {np.sum(R > 0)}")

    except Exception as e:
        print(f"Error in Harris corner detection: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Feature Descriptor Generation
    print("\n" + "=" * 70)
    print("[Test 2] Feature Descriptor Generation")
    print("-" * 70)

    try:
        start_time = time.time()
        descriptors = descriptor.compute_descriptors(test_image, corners)
        elapsed = time.time() - start_time

        print(f"\nGenerated {len(descriptors)} descriptors in {elapsed:.2f} seconds")
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"Descriptor dimension: {descriptors.shape[1]}")

        # Check if some keypoints were filtered
        if len(descriptors) < len(corners):
            print(f"Note: {len(corners) - len(descriptors)} keypoints were filtered (too close to borders)")

        # Descriptor statistics
        print(f"\nDescriptor statistics:")
        print(f"  Mean: {descriptors.mean():.6f}")
        print(f"  Std: {descriptors.std():.6f}")
        print(f"  Min: {descriptors.min():.6f}")
        print(f"  Max: {descriptors.max():.6f}")

        # Check L2 normalization
        norms = np.linalg.norm(descriptors, axis=1)
        print(f"\nL2 norms (should be close to 1.0):")
        print(f"  Mean: {norms.mean():.6f}")
        print(f"  Std: {norms.std():.6f}")
        print(f"  Min: {norms.min():.6f}")
        print(f"  Max: {norms.max():.6f}")

        # Test descriptor distances
        print(f"\nDescriptor distance tests:")
        d1 = descriptors[0]
        d2 = descriptors[1]
        d_same = descriptors[0]

        dist_different = descriptor.match_descriptor_distance(d1, d2)
        dist_same = descriptor.match_descriptor_distance(d1, d_same)

        print(f"  Distance between different descriptors: {dist_different:.6f}")
        print(f"  Distance between same descriptor: {dist_same:.10f}")
        print(f"  Self-distance is near zero: {dist_same < 1e-8}")

    except Exception as e:
        print(f"Error in descriptor generation: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Process Multiple Images
    print("\n" + "=" * 70)
    print("[Test 3] Processing Multiple Images")
    print("-" * 70)

    try:
        # Process first 3 images
        num_test_images = min(3, len(images))
        print(f"\nProcessing first {num_test_images} images...")

        all_corners = []
        all_descriptors = []

        for i in range(num_test_images):
            print(f"\nImage {i+1}: {metadata[i]['filename']}")
            corners_i = harris_detector.detect_harris_corners(images[i])
            desc_i = descriptor.compute_descriptors(images[i], corners_i)

            all_corners.append(corners_i)
            all_descriptors.append(desc_i)

            print(f"  Corners: {len(corners_i)}, Descriptors: {len(desc_i)}")

        print(f"\nSummary:")
        print(f"  Total corners detected: {sum(len(c) for c in all_corners)}")
        print(f"  Total descriptors: {sum(len(d) for d in all_descriptors)}")
        print(f"  Average corners per image: {np.mean([len(c) for c in all_corners]):.1f}")

    except Exception as e:
        print(f"Error processing multiple images: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("Phase 2 Test Complete - All tests passed!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_phase2()
    sys.exit(0 if success else 1)
