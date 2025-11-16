"""
Phase 1 - Step 1.1: Image Loading and Normalization
Loads HEIC images and converts them to normalized grayscale format.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import PIL.Image
from PIL import Image
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
    print("Warning: pillow_heif not available. HEIC support will be limited.")

from config import cfg


def load_images(image_dir: str, pattern: str = "*.HEIC") -> Tuple[List[np.ndarray], List[dict]]:
    """
    Load all images from a directory

    Args:
        image_dir: Path to directory containing images
        pattern: File pattern to match (default: "*.HEIC")

    Returns:
        images: List of normalized grayscale images (0-1 range)
        metadata: List of dictionaries containing image metadata
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    # Get all image files matching pattern
    image_files = sorted(list(image_path.glob(pattern)))
    if not image_files:
        raise ValueError(f"No images found in {image_dir} matching pattern {pattern}")

    print(f"Found {len(image_files)} images")

    images = []
    metadata = []

    for i, img_file in enumerate(image_files):
        print(f"Loading image {i+1}/{len(image_files)}: {img_file.name}")

        # Load image
        img_rgb = load_image(str(img_file))

        # Convert to grayscale
        img_gray = rgb_to_grayscale(img_rgb)

        # Normalize to 0-1 range
        img_norm = normalize_image(img_gray)

        images.append(img_norm)
        metadata.append({
            'filename': img_file.name,
            'width': img_norm.shape[1],
            'height': img_norm.shape[0],
            'index': i,
            'original_shape': img_rgb.shape
        })

    # Optionally resize all images to same resolution
    if cfg.TARGET_IMAGE_SIZE is not None:
        print(f"Resizing all images to {cfg.TARGET_IMAGE_SIZE}")
        images = resize_images(images, cfg.TARGET_IMAGE_SIZE)
        for meta in metadata:
            meta['width'] = cfg.TARGET_IMAGE_SIZE[0]
            meta['height'] = cfg.TARGET_IMAGE_SIZE[1]

    return images, metadata


def load_image(filepath: str) -> np.ndarray:
    """
    Load a single image file (supports HEIC, JPEG, PNG, etc.)

    Args:
        filepath: Path to image file

    Returns:
        RGB image as numpy array (height, width, 3) with values 0-255
    """
    img = Image.open(filepath)

    # Convert to RGB if needed (handles RGBA, L, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    return img_array


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using weighted channel averaging

    Mathematical formula: I_gray = 0.299*R + 0.587*G + 0.114*B
    These weights reflect human perception (more sensitive to green)

    Args:
        image: RGB image (H, W, 3) with values 0-255

    Returns:
        Grayscale image (H, W) with values 0-255
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

    # Manual weighted sum (no OpenCV)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.dot(image, weights)

    return gray.astype(np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-1 range

    Args:
        image: Grayscale image with values 0-255

    Returns:
        Normalized image with values 0-1
    """
    return image / 255.0


def resize_images(images: List[np.ndarray], target_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Resize all images to the same resolution using bilinear interpolation

    Args:
        images: List of grayscale images
        target_size: Target size as (width, height)

    Returns:
        List of resized images
    """
    resized_images = []
    target_width, target_height = target_size

    for img in images:
        resized = bilinear_resize(img, target_width, target_height)
        resized_images.append(resized)

    return resized_images


def bilinear_resize(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """
    Resize image using bilinear interpolation (manual implementation)

    Args:
        image: Input grayscale image (H, W)
        new_width: Target width
        new_height: Target height

    Returns:
        Resized image (new_height, new_width)
    """
    old_height, old_width = image.shape

    # Calculate scale factors
    scale_x = old_width / new_width
    scale_y = old_height / new_height

    # Create coordinate grids for new image
    new_x = np.arange(new_width)
    new_y = np.arange(new_height)

    # Map to old image coordinates
    old_x = (new_x + 0.5) * scale_x - 0.5
    old_y = (new_y + 0.5) * scale_y - 0.5

    # Clip to valid range
    old_x = np.clip(old_x, 0, old_width - 1)
    old_y = np.clip(old_y, 0, old_height - 1)

    # Get integer and fractional parts
    x0 = np.floor(old_x).astype(np.int32)
    y0 = np.floor(old_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, old_width - 1)
    y1 = np.minimum(y0 + 1, old_height - 1)

    # Fractional parts
    fx = old_x - x0
    fy = old_y - y0

    # Create output image
    resized = np.zeros((new_height, new_width), dtype=image.dtype)

    # Bilinear interpolation
    for j in range(new_height):
        for i in range(new_width):
            # Get four neighboring pixels
            I00 = image[y0[j], x0[i]]
            I01 = image[y0[j], x1[i]]
            I10 = image[y1[j], x0[i]]
            I11 = image[y1[j], x1[i]]

            # Interpolate
            wx = fx[i]
            wy = fy[j]

            resized[j, i] = (
                I00 * (1 - wx) * (1 - wy) +
                I01 * wx * (1 - wy) +
                I10 * (1 - wx) * wy +
                I11 * wx * wy
            )

    return resized


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Convert normalized image (0-1) back to 0-255 range

    Args:
        image: Normalized image (0-1)

    Returns:
        Image with values 0-255
    """
    return (image * 255.0).astype(np.uint8)


if __name__ == "__main__":
    # Test image loading
    import sys

    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = "data/scene1"

    print(f"Testing image loading from {image_dir}")
    print("=" * 50)

    images, metadata = load_images(image_dir)

    print("\n" + "=" * 50)
    print(f"Successfully loaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    print(f"Image dtype: {images[0].dtype}")
    print(f"Value range: [{images[0].min():.3f}, {images[0].max():.3f}]")

    print("\nMetadata:")
    for i, meta in enumerate(metadata):
        print(f"  Image {i}: {meta['filename']} - {meta['width']}x{meta['height']}")
