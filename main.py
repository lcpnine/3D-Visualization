"""
Main pipeline for Structure from Motion (SfM)
Complete 3D reconstruction from HEIC images
"""

import sys
import time
from pathlib import Path

import numpy as np

from config import cfg
from src.pointcloud import filter as pcloud_filter
from src.pointcloud import generator, visualizer
from src.preprocessing import camera_calibration, image_loader
from src.reconstruction.incremental import IncrementalSfM
from utils import io_utils


def main(image_dir: str = "data/scene1", output_dir: str = "output", max_images: int = None,
         enable_debug: bool = False):
    """
    Run complete SfM pipeline

    Args:
        image_dir: Directory containing input images
        output_dir: Directory for output files
        max_images: Maximum number of images to process (None = all)
        enable_debug: Enable visual debugging (generates debug visualizations)
    """
    print("=" * 70)
    print(f"Structure from Motion - Processing: {image_dir}")
    if enable_debug:
        print("DEBUG MODE ENABLED")
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

    debug_dir = str(Path(output_dir) / "debug")
    sfm = IncrementalSfM(images, K, verbose=cfg.VERBOSE,
                         enable_debug=enable_debug, debug_dir=debug_dir)
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
    print(f"RECONSTRUCTION COMPLETE for {image_dir}")
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
    # import argparse # 원본 argparse 코드 제거

    # 처리할 씬 목록
    scenes_to_process = ["data/scene5", "data/scene8"]
    
    # 공통 설정
    base_output_dir = "output"
    max_images_all = None  # 모든 이미지 처리
    debug_enabled = True   # 디버그 모드 활성화

    all_scenes_successful = True

    for scene_dir in scenes_to_process:
        print("*" * 80)
        print(f"STARTING PIPELINE FOR SCENE: {scene_dir}")
        print("*" * 80)

        # 각 씬에 대한 고유한 출력 디렉터리 생성 (예: output/scene1)
        scene_name = Path(scene_dir).name
        current_output_dir = str(Path(base_output_dir) / scene_name)

        try:
            result = main(
                image_dir=scene_dir,
                output_dir=current_output_dir,
                max_images=max_images_all,
                enable_debug=debug_enabled
            )
            print(f"[SUCCESS] Finished processing {scene_dir}")
        
        except Exception as e:
            print(f"\n[ERROR] FAILED processing {scene_dir}: {e}")
            import traceback
            traceback.print_exc()
            all_scenes_successful = False
            print(f"--- Skipping to next scene ---")

    print("*" * 80)
    print("All processing finished.")
    if not all_scenes_successful:
        print("One or more scenes failed to process.")
        sys.exit(1)
    else:
        print("All scenes processed successfully.")
        sys.exit(0)
