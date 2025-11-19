"""
Global configuration parameters for Structure from Motion (SfM) pipeline
OPTIMIZED WITH OPENCV-STYLE ALGORITHMS
Version 2.0 - Improved matching, RANSAC, and reconstruction quality
"""

class Config:
    """Global configuration class with all parameters for SfM pipeline"""

    # ==================== Phase 1: Image Preprocessing ====================
    # Image normalization
    TARGET_IMAGE_SIZE = None  # None = keep original size, or tuple (width, height)

    # Camera parameters
    DEFAULT_FOV = 60.0  # Field of View in degrees (typical for smartphones)

    # ==================== Phase 2: Feature Detection ====================
    # Feature Detector Selection
    FEATURE_DETECTOR = 'sift'  # Options: 'harris', 'sift', 'orb'
    TARGET_FEATURES = 8000  # ⭐ INCREASED from 5000 - 더 많은 feature로 매칭 확률 높이기

    # Harris Corner Detection (used when FEATURE_DETECTOR='harris')
    HARRIS_K = 0.04
    HARRIS_THRESHOLD = 0.005
    HARRIS_WINDOW_SIZE = 5
    HARRIS_SIGMA = 1.5
    NMS_WINDOW_SIZE = 5
    TARGET_CORNERS_MIN = 500
    TARGET_CORNERS_MAX = 3000

    # Enhanced Detection Methods
    USE_MULTISCALE_DETECTION = True
    USE_ADAPTIVE_THRESHOLD = True

    # SIFT Parameters (used when FEATURE_DETECTOR='sift')
    SIFT_N_OCTAVES = 4
    SIFT_CONTRAST_THRESHOLD = 0.03  # ⭐ DECREASED from 0.04 - 더 많은 feature
    SIFT_EDGE_THRESHOLD = 10
    SIFT_SIGMA = 1.6

    # ORB Parameters (used when FEATURE_DETECTOR='orb')
    ORB_N_KEYPOINTS = 8000  # INCREASED
    ORB_SCALE_FACTOR = 1.2
    ORB_N_LEVELS = 8
    ORB_EDGE_THRESHOLD = 31
    ORB_FAST_THRESHOLD = 20

    # Canny Edge Detection
    CANNY_SIGMA = 1.4
    CANNY_LOW_THRESHOLD = 0.05
    CANNY_HIGH_THRESHOLD = 0.15
    EDGE_CORNER_THRESHOLD = 4
    EDGE_CORNERS_MAX = 2000

    # Feature Descriptor
    PATCH_SIZE = 11
    DESCRIPTOR_BORDER = 10

    # ==================== Phase 3: Feature Matching ====================
    # Descriptor Matching (OpenCV-style improvements)
    RATIO_TEST_THRESHOLD = 0.75   # OpenCV default for SIFT (0.7-0.8 range)
    USE_SYMMETRIC_MATCHING = True # Cross-check matching (OpenCV crossCheck=True)
    MIN_MATCHES = 20              # Minimum matches to attempt geometry estimation

    # RANSAC for Fundamental Matrix (Improved with adaptive + 7-point)
    RANSAC_ITERATIONS = 5000      # Maximum iterations (will adapt based on inlier ratio)
    RANSAC_THRESHOLD = 1.0        # OpenCV default is 1.0 pixels (was 2.0)
    MIN_INLIERS = 15              # Lowered from 30 for better success rate

    # ==================== Phase 4: Camera Pose Estimation ====================
    # Essential Matrix
    ESSENTIAL_RANK_TOLERANCE = 1e-6

    # ==================== Phase 5: Triangulation ====================
    # Triangulation Quality (Balanced for real-world data)
    REPROJ_ERROR_THRESHOLD = 10.0  # Pixels (was 3.0 - slightly relaxed)
    MIN_PARALLAX_ANGLE = 0.5
    MIN_DEPTH = 0.1
    MAX_DEPTH = 1000.0
    TRIANGULATION_METHOD = 'dlt'  # 'dlt' or 'midpoint'

    # ==================== Phase 6: Incremental Reconstruction ====================
    # PnP RANSAC (Using P3P + EPnP)
    PNP_RANSAC_ITERATIONS = 3000  # Reduced (P3P is faster with minimal set of 3)
    PNP_RANSAC_THRESHOLD = 4.0    # Pixels (relaxed from 3.0)
    MIN_PNP_POINTS = 3            # Changed from 6 (now using P3P)
    MIN_PNP_INLIERS = 4           # Lowered from 6 (P3P needs fewer points)

    # Next Image Selection
    MIN_2D3D_MATCHES = 10         # Further decreased for better coverage

    # ==================== Phase 7: Bundle Adjustment ====================
    # Optimization
    BA_MAX_ITERATIONS = 100
    BA_LOSS = 'huber'
    BA_F_TOL = 1e-4
    BA_X_TOL = 1e-4
    FIX_FIRST_CAMERA = True

    # Outlier Rejection
    OUTLIER_THRESHOLD_FACTOR = 3.0
    BA_OUTLIER_ITERATIONS = 2
    MIN_TRACK_LENGTH = 2

    # ==================== Phase 8: Point Cloud Post-processing ====================
    # Statistical Outlier Removal
    KNN_NEIGHBORS = 20
    OUTLIER_STD_RATIO = 2.5  # ⭐ INCREASED - 덜 aggressive

    # Visualization
    POINT_CLOUD_SUBSAMPLE = None
    CAMERA_SCALE = 0.1

    # ==================== General Settings ====================
    # Logging
    VERBOSE = True
    SAVE_INTERMEDIATE_RESULTS = True

    # Output
    OUTPUT_DIR = "output"
    PLY_FORMAT = "ascii"

    # Random seed for reproducibility
    RANDOM_SEED = 42


# Create a global instance
cfg = Config()
