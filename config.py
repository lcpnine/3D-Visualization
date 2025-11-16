"""
Global configuration parameters for Structure from Motion (SfM) pipeline
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
    TARGET_FEATURES = 2000  # Target number of features to extract per image

    # Harris Corner Detection (used when FEATURE_DETECTOR='harris')
    HARRIS_K = 0.04  # Harris corner response parameter (lower = more corners)
    HARRIS_THRESHOLD = 0.005  # Corner response threshold (lowered from 0.01 for better detection)
    HARRIS_WINDOW_SIZE = 5  # Window size for structure tensor
    HARRIS_SIGMA = 1.5  # Gaussian sigma for structure tensor
    NMS_WINDOW_SIZE = 5  # Non-maximum suppression window size
    TARGET_CORNERS_MIN = 500  # Minimum number of corners per image
    TARGET_CORNERS_MAX = 1000  # Maximum number of corners per image (optimized for speed)

    # Enhanced Detection Methods
    USE_MULTISCALE_DETECTION = False  # Enable multi-scale detection (disabled for speed)
    USE_ADAPTIVE_THRESHOLD = True  # Enable adaptive thresholding (more robust than max-based threshold)

    # SIFT Parameters (used when FEATURE_DETECTOR='sift')
    SIFT_N_OCTAVES = 4  # Number of octaves in scale space
    SIFT_CONTRAST_THRESHOLD = 0.04  # Contrast threshold for filtering weak features
    SIFT_EDGE_THRESHOLD = 10  # Edge threshold for filtering edge responses
    SIFT_SIGMA = 1.6  # Gaussian sigma for scale space

    # ORB Parameters (used when FEATURE_DETECTOR='orb')
    ORB_N_KEYPOINTS = 2000  # Maximum number of keypoints to retain
    ORB_SCALE_FACTOR = 1.2  # Pyramid decimation ratio
    ORB_N_LEVELS = 8  # Number of pyramid levels
    ORB_EDGE_THRESHOLD = 31  # Size of border where features are not detected
    ORB_FAST_THRESHOLD = 20  # FAST detector threshold

    # Canny Edge Detection (for detecting object outlines like bottle bodies)
    CANNY_SIGMA = 1.4  # Gaussian sigma for Canny edge detection
    CANNY_LOW_THRESHOLD = 0.05  # Low threshold for hysteresis (relative to max gradient)
    CANNY_HIGH_THRESHOLD = 0.15  # High threshold for hysteresis (relative to max gradient)
    EDGE_CORNER_THRESHOLD = 4  # Minimum edge neighbors for corner detection
    EDGE_CORNERS_MAX = 1000  # Maximum edge-based corners to extract

    # Feature Descriptor
    PATCH_SIZE = 11  # Patch size for descriptor (11x11 = 121 dimensions, balanced)
    DESCRIPTOR_BORDER = 10  # Border pixels to exclude from descriptor extraction

    # ==================== Phase 3: Feature Matching ====================
    # Descriptor Matching
    RATIO_TEST_THRESHOLD = 0.95  # Lowe's ratio test threshold (relaxed but not too much)
    USE_SYMMETRIC_MATCHING = False  # Disabled for simple descriptors to get more matches
    MIN_MATCHES = 20  # Minimum number of matches required (lowered for simple descriptors)

    # RANSAC for Fundamental Matrix
    RANSAC_ITERATIONS = 5000  # Number of RANSAC iterations (increased for better convergence)
    RANSAC_THRESHOLD = 5.0  # Inlier threshold in pixels (relaxed for simple descriptors)
    MIN_INLIERS = 10  # Minimum number of inliers required (relaxed)

    # ==================== Phase 4: Camera Pose Estimation ====================
    # Essential Matrix
    ESSENTIAL_RANK_TOLERANCE = 1e-6  # Tolerance for rank constraint

    # ==================== Phase 5: Triangulation ====================
    # Triangulation Quality
    REPROJ_ERROR_THRESHOLD = 100.0  # Reprojection error threshold in pixels (relaxed for robustness)
    MIN_PARALLAX_ANGLE = 0.01  # Minimum parallax angle in degrees (relaxed)
    MIN_DEPTH = 0.1  # Minimum depth for valid points (very permissive)
    MAX_DEPTH = 10000.0  # Maximum depth for valid points (very permissive)

    # ==================== Phase 6: Incremental Reconstruction ====================
    # PnP RANSAC
    PNP_RANSAC_ITERATIONS = 2000  # Number of RANSAC iterations for PnP (increased)
    PNP_RANSAC_THRESHOLD = 50.0  # Inlier threshold in pixels for PnP (relaxed)
    MIN_PNP_POINTS = 6  # Minimum number of 2D-3D correspondences for PnP
    MIN_PNP_INLIERS = 5  # Minimum number of inliers for successful PnP (relaxed)

    # Next Image Selection
    MIN_2D3D_MATCHES = 15  # Minimum 2D-3D matches to add new image (relaxed)

    # ==================== Phase 7: Bundle Adjustment ====================
    # Optimization
    BA_MAX_ITERATIONS = 100  # Maximum iterations for bundle adjustment
    BA_LOSS = 'huber'  # Loss function: 'linear', 'huber', 'soft_l1', 'cauchy'
    BA_F_TOL = 1e-4  # Function tolerance for convergence
    BA_X_TOL = 1e-4  # Parameter tolerance for convergence
    FIX_FIRST_CAMERA = True  # Fix first camera to remove gauge freedom

    # Outlier Rejection
    OUTLIER_THRESHOLD_FACTOR = 3.0  # Factor for adaptive threshold (median + k*MAD)
    BA_OUTLIER_ITERATIONS = 2  # Number of BA + outlier rejection iterations
    MIN_TRACK_LENGTH = 2  # Minimum number of views for each 3D point

    # ==================== Phase 8: Point Cloud Post-processing ====================
    # Statistical Outlier Removal
    KNN_NEIGHBORS = 20  # Number of neighbors for k-NN
    OUTLIER_STD_RATIO = 2.0  # Standard deviation ratio for outlier removal

    # Visualization
    POINT_CLOUD_SUBSAMPLE = None  # Subsample point cloud for visualization (None = no subsampling)
    CAMERA_SCALE = 0.1  # Camera frustum scale for visualization

    # ==================== General Settings ====================
    # Logging
    VERBOSE = True  # Print progress information
    SAVE_INTERMEDIATE_RESULTS = True  # Save intermediate results for debugging

    # Output
    OUTPUT_DIR = "output"  # Output directory
    PLY_FORMAT = "ascii"  # PLY file format: 'ascii' or 'binary'

    # Random seed for reproducibility
    RANDOM_SEED = 42


# Create a global instance
cfg = Config()
