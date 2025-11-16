"""
File I/O utility functions for Structure from Motion
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any
import PIL.Image
from PIL import Image


def save_ply(points: np.ndarray, colors: np.ndarray = None,
             filepath: str = "output.ply", format: str = "ascii") -> None:
    """
    Save point cloud to PLY file

    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3) with values 0-255, optional
        filepath: Output file path
        format: 'ascii' or 'binary'
    """
    N = points.shape[0]

    # Ensure colors are in correct format
    if colors is not None:
        if colors.shape[0] != N:
            raise ValueError(f"Points and colors must have same length: {N} vs {colors.shape[0]}")
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        # Default to white
        colors = np.full((N, 3), 255, dtype=np.uint8)

    # Create output directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Write PLY file
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write(f"format {format} 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i in range(N):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"Saved {N} points to {filepath}")


def load_ply(filepath: str) -> tuple:
    """
    Load point cloud from PLY file

    Args:
        filepath: PLY file path

    Returns:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3)
    """
    with open(filepath, 'r') as f:
        # Read header
        line = f.readline()
        if line.strip() != "ply":
            raise ValueError("Not a valid PLY file")

        # Skip header lines
        vertex_count = 0
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line == "end_header":
                break

        # Read data
        points = []
        colors = []
        for _ in range(vertex_count):
            parts = f.readline().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            points.append([x, y, z])
            colors.append([r, g, b])

    return np.array(points), np.array(colors)


def save_reconstruction(cameras: List[Dict], points_3d: np.ndarray,
                       filepath: str = "reconstruction.npz") -> None:
    """
    Save reconstruction results to compressed NumPy file

    Args:
        cameras: List of camera dictionaries with 'R', 't', 'K'
        points_3d: 3D points (N, 3)
        filepath: Output file path
    """
    # Create output directory
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Prepare camera data
    camera_Rs = np.array([cam['R'] for cam in cameras])
    camera_ts = np.array([cam['t'] for cam in cameras])
    camera_Ks = np.array([cam['K'] for cam in cameras])

    # Save
    np.savez_compressed(
        filepath,
        points_3d=points_3d,
        camera_Rs=camera_Rs,
        camera_ts=camera_ts,
        camera_Ks=camera_Ks
    )

    print(f"Saved reconstruction to {filepath}")
    print(f"  Cameras: {len(cameras)}")
    print(f"  3D Points: {len(points_3d)}")


def load_reconstruction(filepath: str) -> tuple:
    """
    Load reconstruction from compressed NumPy file

    Args:
        filepath: Input file path

    Returns:
        cameras: List of camera dictionaries
        points_3d: 3D points (N, 3)
    """
    data = np.load(filepath)

    points_3d = data['points_3d']
    camera_Rs = data['camera_Rs']
    camera_ts = data['camera_ts']
    camera_Ks = data['camera_Ks']

    cameras = []
    for i in range(len(camera_Rs)):
        cameras.append({
            'R': camera_Rs[i],
            't': camera_ts[i],
            'K': camera_Ks[i]
        })

    return cameras, points_3d


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    data_converted = convert_numpy(data)

    with open(filepath, 'w') as f:
        json.dump(data_converted, f, indent=2)

    print(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file

    Args:
        filepath: Input file path

    Returns:
        Dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_image(image: np.ndarray, filepath: str) -> None:
    """
    Save image to file

    Args:
        image: Image array (grayscale or RGB)
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Save using PIL
    img = Image.fromarray(image)
    img.save(filepath)


def save_statistics(stats: Dict[str, Any], filepath: str = "output/reports/statistics.json") -> None:
    """
    Save reconstruction statistics to JSON file

    Args:
        stats: Dictionary containing statistics
        filepath: Output file path
    """
    save_json(stats, filepath)


def write_log(message: str, filepath: str = "output/reports/reconstruction_log.txt",
              mode: str = 'a') -> None:
    """
    Write message to log file

    Args:
        message: Message to write
        filepath: Log file path
        mode: File mode ('a' for append, 'w' for write)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, mode) as f:
        f.write(message + "\n")


if __name__ == "__main__":
    # Test I/O utilities
    print("Testing I/O Utilities")
    print("=" * 50)

    # Test PLY save
    print("\n1. Testing PLY save/load:")
    points = np.random.rand(100, 3) * 10
    colors = np.random.randint(0, 256, (100, 3))

    save_ply(points, colors, "output/test.ply")
    loaded_points, loaded_colors = load_ply("output/test.ply")

    print(f"Original points shape: {points.shape}")
    print(f"Loaded points shape: {loaded_points.shape}")
    print(f"Points match: {np.allclose(points, loaded_points, atol=1e-5)}")

    # Test JSON save
    print("\n2. Testing JSON save/load:")
    data = {
        'n_cameras': 5,
        'n_points': 1000,
        'reprojection_error': 1.23,
        'intrinsic': np.eye(3)
    }
    save_json(data, "output/test.json")
    loaded_data = load_json("output/test.json")
    print(f"Original keys: {data.keys()}")
    print(f"Loaded keys: {loaded_data.keys()}")

    print("\nI/O utilities test complete!")
