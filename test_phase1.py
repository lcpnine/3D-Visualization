"""
Test script for Phase 1: Image preprocessing and camera calibration
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import image_loader, camera_calibration
from config import cfg

def test_phase1():
    print("=" * 70)
    print("Testing Phase 1: Image Preprocessing and Camera Calibration")
    print("=" * 70)

    # Test 1: Image Loading
    print("\n[Test 1] Loading HEIC images from data/scene1/")
    print("-" * 70)

    try:
        images, metadata = image_loader.load_images("data/scene1")

        print(f"\nSuccessfully loaded {len(images)} images")
        print(f"\nFirst image details:")
        print(f"  Filename: {metadata[0]['filename']}")
        print(f"  Shape: {images[0].shape}")
        print(f"  Dtype: {images[0].dtype}")
        print(f"  Value range: [{images[0].min():.3f}, {images[0].max():.3f}]")
        print(f"  Original shape: {metadata[0]['original_shape']}")

        print(f"\nAll images:")
        for i, meta in enumerate(metadata):
            print(f"  {i+1}. {meta['filename']:15s} - {meta['width']}x{meta['height']}")

    except Exception as e:
        print(f"Error loading images: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Camera Calibration
    print("\n" + "=" * 70)
    print("[Test 2] Camera Intrinsic Matrix Estimation")
    print("-" * 70)

    try:
        # Get image dimensions from first image
        width = metadata[0]['width']
        height = metadata[0]['height']

        # Estimate intrinsic matrix
        K = camera_calibration.estimate_intrinsic_matrix(width, height)

        print(f"\nImage size: {width} x {height}")
        print(f"Field of View: {cfg.DEFAULT_FOV} degrees")
        print(f"\nIntrinsic Matrix K:")
        print(K)

        # Extract parameters
        params = camera_calibration.camera_params_from_intrinsic(K)
        print(f"\nCamera Parameters:")
        print(f"  fx = {params['fx']:.2f} pixels")
        print(f"  fy = {params['fy']:.2f} pixels")
        print(f"  cx = {params['cx']:.2f} pixels")
        print(f"  cy = {params['cy']:.2f} pixels")

        # Validate
        is_valid = camera_calibration.validate_intrinsic_matrix(K)
        print(f"\nIntrinsic matrix is valid: {is_valid}")

        # Verify FOV
        fov_computed = camera_calibration.get_fov_from_focal_length(params['fx'], width)
        print(f"\nVerification:")
        print(f"  Computed FOV from fx: {fov_computed:.2f} degrees")
        print(f"  Original FOV: {cfg.DEFAULT_FOV:.2f} degrees")
        print(f"  Match: {np.isclose(fov_computed, cfg.DEFAULT_FOV)}")

    except Exception as e:
        print(f"Error in camera calibration: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Sample Statistics
    print("\n" + "=" * 70)
    print("[Test 3] Image Statistics")
    print("-" * 70)

    try:
        # Compute statistics for first image
        img = images[0]
        print(f"\nFirst image statistics:")
        print(f"  Mean: {img.mean():.3f}")
        print(f"  Std: {img.std():.3f}")
        print(f"  Min: {img.min():.3f}")
        print(f"  Max: {img.max():.3f}")

        # Histogram
        hist, bins = np.histogram(img.flatten(), bins=10)
        print(f"\nValue distribution (10 bins):")
        for i in range(len(hist)):
            bar = '█' * int(hist[i] / hist.max() * 50)
            print(f"  [{bins[i]:.2f} - {bins[i+1]:.2f}]: {bar} ({hist[i]})")

    except Exception as e:
        print(f"Error computing statistics: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("Phase 1 Test Complete - All tests passed!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_phase1()
    sys.exit(0 if success else 1)
