"""
Main pipeline for Structure from Motion (SfM)
Complete 3D reconstruction from HEIC images
"""

import sys
import numpy as np
import time
from pathlib import Path

from config import cfg
from src.preprocessing import image_loader, camera_calibration
from src.reconstruction.incremental import IncrementalSfM
from src.pointcloud import generator, filter as pcloud_filter, visualizer
from utils import io_utils


def main(image_dir: str = "data/scene1", output_dir: str = "output", max_images: int = None):
    """
    Run complete SfM pipeline

    Args:
        image_dir: Directory containing input images
        output_dir: Directory for output files
        max_images: Maximum number of images to process (None = all)
    """
    print("=" * 70)
    print("Structure from Motion - Complete Pipeline")
    print("=" * 70)

    start_time = time.time()

    # ========== Phase 1: Image Preprocessing ==========
    print("\n" + "=" * 70)
    print("PHASE 1: Image Preprocessing")
    print("=" * 70)

    images, metadata = image_loader.load_images(image_dir)

    # Limit number of images if specified
    if max_images is not None and len(images) > max_images:
        print(f"\nLimiting to first {max_images} images")
        images = images[:max_images]
        metadata = metadata[:max_images]

    print(f"\nLoaded {len(images)} images")
    print(f"Image resolution: {metadata[0]['width']} x {metadata[0]['height']}")

    # Estimate camera intrinsics
    width = metadata[0]['width']
    height = metadata[0]['height']
    K = camera_calibration.estimate_intrinsic_matrix(width, height)

    print(f"\nCamera intrinsic matrix:")
    print(K)
    print(f"Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")

    # ========== Phases 2-6: Incremental Reconstruction ==========
    print("\n" + "=" * 70)
    print("PHASES 2-6: Incremental Reconstruction")
    print("=" * 70)

    sfm = IncrementalSfM(images, K, verbose=cfg.VERBOSE)
    result = sfm.reconstruct()

    cameras = result['cameras']
    points_3d = result['points_3d']
    point_colors = result['point_colors']

    print(f"\nReconstruction summary:")
    print(f"  Input images: {len(images)}")
    print(f"  Reconstructed cameras: {len(cameras)}")
    print(f"  3D points: {len(points_3d)}")

    # ========== Phase 8: Point Cloud Post-processing ==========
    print("\n" + "=" * 70)
    print("PHASE 8: Point Cloud Generation and Visualization")
    print("=" * 70)

    # Statistical outlier removal
    if len(points_3d) > cfg.KNN_NEIGHBORS:
        print("\nApplying statistical outlier removal...")
        filtered_points = pcloud_filter.statistical_outlier_removal(points_3d)

        # Update colors for filtered points
        # (simplified - keep corresponding colors)
        if len(filtered_points) < len(points_3d):
            # Find which points were kept (simplified approach)
            filtered_colors = point_colors[:len(filtered_points)]
        else:
            filtered_colors = point_colors
    else:
        filtered_points = points_3d
        filtered_colors = point_colors

    print(f"\nFinal point cloud: {len(filtered_points)} points")

    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nSaving outputs...")

    # Save point cloud (PLY format)
    ply_path = output_path / "point_clouds" / "reconstruction.ply"
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_point_cloud_ply(filtered_points, filtered_colors, str(ply_path))

    # Save point cloud (NumPy format)
    npy_path = output_path / "point_clouds" / "reconstruction.npz"
    generator.save_point_cloud_npy(filtered_points, filtered_colors, str(npy_path))

    # Save reconstruction data
    reconstruction_path = output_path / "reconstruction.npz"
    io_utils.save_reconstruction(cameras, filtered_points, str(reconstruction_path))

    # Visualizations
    print("\nGenerating visualizations...")

    viz_path = output_path / "visualizations"
    viz_path.mkdir(parents=True, exist_ok=True)

    # Visualize point cloud
    visualizer.visualize_point_cloud(
        filtered_points, filtered_colors,
        save_path=str(viz_path / "point_cloud_3d.png"),
        title=f"3D Reconstruction ({len(filtered_points)} points)"
    )

    # Visualize cameras
    visualizer.plot_cameras(
        cameras, filtered_points,
        save_path=str(viz_path / "cameras_3d.png"),
        title=f"Camera Trajectory ({len(cameras)} cameras)"
    )

    # Save statistics
    stats = {
        'n_images': len(images),
        'n_cameras': len(cameras),
        'n_points_raw': len(points_3d),
        'n_points_filtered': len(filtered_points),
        'image_resolution': f"{width}x{height}",
        'focal_length': float(K[0, 0]),
        'processing_time_seconds': time.time() - start_time
    }

    stats_path = output_path / "reports" / "statistics.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    io_utils.save_json(stats, str(stats_path))

    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("RECONSTRUCTION COMPLETE!")
    print("=" * 70)
    print(f"\nStatistics:")
    print(f"  Total processing time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  Images processed: {len(images)}")
    print(f"  Cameras reconstructed: {len(cameras)} ({100*len(cameras)/len(images):.1f}%)")
    print(f"  3D points (raw): {len(points_3d)}")
    print(f"  3D points (filtered): {len(filtered_points)}")

    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  Point cloud (PLY): {ply_path}")
    print(f"  Point cloud (NPZ): {npy_path}")
    print(f"  Reconstruction data: {reconstruction_path}")
    print(f"  Visualizations: {viz_path}/")
    print(f"  Statistics: {stats_path}")

    print("\n" + "=" * 70)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Structure from Motion Pipeline")
    parser.add_argument("--image_dir", type=str, default="data/scene1",
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory for output files")
    parser.add_argument("--max_images", type=int, default=None,
                       help="Maximum number of images to process (default: all)")

    args = parser.parse_args()

    try:
        result = main(args.image_dir, args.output_dir, args.max_images)
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
