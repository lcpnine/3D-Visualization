#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all required dependencies are installed and working correctly.
"""

import sys

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("="*70)
    print("Verifying 3D Visualization Setup")
    print("="*70)

    all_ok = True

    # Check Python version
    print("\n1. Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro}")
        print("     WARNING: Python 3.8 or higher is recommended")
        all_ok = False

    # Check NumPy
    print("\n2. Checking NumPy...")
    try:
        import numpy as np
        print(f"   ✓ NumPy {np.__version__}")
    except ImportError:
        print("   ✗ NumPy not found")
        print("     Install with: pip install numpy")
        all_ok = False

    # Check SciPy
    print("\n3. Checking SciPy...")
    try:
        import scipy
        print(f"   ✓ SciPy {scipy.__version__}")
    except ImportError:
        print("   ✗ SciPy not found")
        print("     Install with: pip install scipy")
        all_ok = False

    # Check Pillow
    print("\n4. Checking Pillow...")
    try:
        import PIL
        from PIL import Image
        print(f"   ✓ Pillow {PIL.__version__}")
    except ImportError:
        print("   ✗ Pillow not found")
        print("     Install with: pip install pillow")
        all_ok = False

    # Check pillow-heif (critical for HEIC support)
    print("\n5. Checking pillow-heif (HEIC support)...")
    try:
        import pillow_heif
        from pillow_heif import register_heif_opener
        register_heif_opener()
        print(f"   ✓ pillow-heif {pillow_heif.__version__}")
        print("     HEIC/HEIF images are supported!")
    except ImportError:
        print("   ✗ pillow-heif not found")
        print("     CRITICAL: This is required for loading HEIC images!")
        print("     Install with: pip install pillow-heif")
        all_ok = False

    # Check Matplotlib
    print("\n6. Checking Matplotlib...")
    try:
        import matplotlib
        print(f"   ✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("   ✗ Matplotlib not found")
        print("     Install with: pip install matplotlib")
        all_ok = False

    # Test HEIC loading capability
    print("\n7. Testing HEIC loading capability...")
    try:
        from PIL import Image
        # Try to check if HEIC is registered
        if 'HEIF' in Image.OPEN or hasattr(Image, 'register_heif_opener'):
            print("   ✓ HEIC format is registered with PIL")
        else:
            print("   ℹ HEIC registration status unclear, but pillow-heif is installed")
    except Exception as e:
        print(f"   ⚠ Could not verify HEIC support: {e}")

    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("✓ All dependencies are installed correctly!")
        print("  You can now run: python main.py --image_dir data/scene1")
    else:
        print("✗ Some dependencies are missing!")
        print("\nQuick fix - Install all dependencies:")
        print("  pip install -r requirements.txt")
    print("="*70)

    return all_ok

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
