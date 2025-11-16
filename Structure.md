# SfM Project - File Structure & Module Documentation

## 📁 Project Directory Structure

```
sfm_project/
├── README.md                          # Main project guide
├── STRUCTURE.md                       # File structure and module documentation
├── requirements.txt                   # Python package dependencies
├── config.py                          # Global configuration and parameters
├── main.py                            # Execute entire pipeline
│
├── data/                              # Input data
│   ├── scene1/                        # Scene 1 images
│   │   ├── IMG_2040.HEIC
│   │   ├── IMG_2041.HEIC
│   │   └── ...
│   ├── scene2/                        # Scene 2 images (for generalization testing)
│   └── camera_params.json             # Camera parameters (optional)
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── preprocessing/                 # Phase 1: Image Preprocessing
│   │   ├── __init__.py
│   │   ├── image_loader.py           # Step 1.1: Image loading and normalization
│   │   └── camera_calibration.py     # Step 1.2: Camera intrinsic parameter estimation
│   │
│   ├── features/                      # Phase 2: Feature Detection
│   │   ├── __init__.py
│   │   ├── harris_detector.py        # Step 2.1: Harris Corner Detection
│   │   └── descriptor.py             # Step 2.2: Feature descriptor generation
│   │
│   ├── matching/                      # Phase 3: Feature Matching
│   │   ├── __init__.py
│   │   ├── matcher.py                # Step 3.1: Brute-force Matching
│   │   └── ransac.py                 # Step 3.2: RANSAC & F-matrix estimation
│   │
│   ├── geometry/                      # Phase 4: Camera Pose Estimation
│   │   ├── __init__.py
│   │   ├── essential_matrix.py       # Step 4.1: Essential Matrix recovery
│   │   └── pose_recovery.py          # Step 4.2: R, t extraction
│   │
│   ├── triangulation/                 # Phase 5: 3D Point Triangulation
│   │   ├── __init__.py
│   │   ├── triangulate.py            # Step 5.1: Two-view Triangulation
│   │   └── validation.py             # Step 5.2: Triangulation quality validation
│   │
│   ├── reconstruction/                # Phase 6: Incremental Reconstruction
│   │   ├── __init__.py
│   │   ├── incremental.py            # Step 6.1-6.4: Incremental SfM
│   │   └── pnp.py                    # Step 6.2: PnP Solver
│   │
│   ├── optimization/                  # Phase 7: Bundle Adjustment
│   │   ├── __init__.py
│   │   ├── bundle_adjustment.py      # Step 7.1-7.2: BA optimization
│   │   └── outlier_filter.py         # Step 7.3: Outlier Rejection
│   │
│   └── pointcloud/                    # Phase 8: Point Cloud Generation
│       ├── __init__.py
│       ├── generator.py              # Step 8.1: Point Cloud extraction
│       ├── filter.py                 # Step 8.2: Statistical Outlier Removal
│       └── visualizer.py             # Step 8.3: Visualization
│
├── utils/                             # Common utility functions
│   ├── __init__.py
│   ├── math_utils.py                 # Mathematical functions (SVD, normalize, etc.)
│   ├── io_utils.py                   # File I/O utilities
│   └── visualization.py              # Visualization helper functions
│
├── output/                            # Result files
│   ├── point_clouds/                 # Generated point clouds
│   │   ├── scene1.ply
│   │   └── scene1.npy
│   ├── visualizations/               # Visualization images
│   │   ├── features_scene1.png
│   │   ├── matches_scene1.png
│   │   └── cameras_scene1.png
│   └── reports/                      # Intermediate results and statistics
│       └── reconstruction_log.txt
│
└── tests/                             # Unit tests (optional)
    ├── __init__.py
    ├── test_features.py
    ├── test_matching.py
    └── test_triangulation.py
```

---

## 📋 Module Responsibilities and README Mapping

### **Phase 1: preprocessing/** - Image Preprocessing and Preparation

#### `image_loader.py`
- **README Section**: Phase 1 > Step 1.1
- **Key Functions**:
  ```python
  load_images(image_dir: str) -> List[np.ndarray]
  rgb_to_grayscale(image: np.ndarray) -> np.ndarray
  normalize_image(image: np.ndarray) -> np.ndarray
  resize_images(images: List[np.ndarray], target_size: tuple) -> List[np.ndarray]
  ```
- **Input**: Image directory path
- **Output**: List of normalized grayscale images
- **Implementation Requirements**:
  - Read images using PIL/imageio
  - Manual RGB → Grayscale conversion (0.299R + 0.587G + 0.114B)
  - Implement bilinear interpolation manually

#### `camera_calibration.py`
- **README Section**: Phase 1 > Step 1.2
- **Key Functions**:
  ```python
  estimate_intrinsic_matrix(image_width: int, image_height: int, fov_degrees: float) -> np.ndarray
  compute_focal_length(width: int, fov: float) -> float
  get_principal_point(width: int, height: int) -> tuple
  ```
- **Input**: Image resolution, FOV
- **Output**: 3×3 Intrinsic matrix K
- **Implementation Requirements**:
  - Calculate focal length from FOV
  - Assume principal point at image center

---

### **Phase 2: features/** - Feature Detection

#### `harris_detector.py`
- **README Section**: Phase 2 > Step 2.1
- **Key Functions**:
  ```python
  detect_harris_corners(image: np.ndarray, k: float = 0.05, threshold: float = 0.01) -> np.ndarray
  compute_gradients(image: np.ndarray) -> tuple  # Ix, Iy
  compute_structure_tensor(Ix: np.ndarray, Iy: np.ndarray, window_size: int) -> np.ndarray
  corner_response(M: np.ndarray, k: float) -> np.ndarray
  non_maximum_suppression(response: np.ndarray, window_size: int) -> np.ndarray
  ```
- **Input**: Grayscale image
- **Output**: Corner coordinates (N × 2 array)
- **Implementation Requirements**:
  - Implement Sobel filter manually
  - Gaussian weighting
  - NMS (Non-maximum Suppression)

#### `descriptor.py`
- **README Section**: Phase 2 > Step 2.2
- **Key Functions**:
  ```python
  compute_descriptors(image: np.ndarray, keypoints: np.ndarray, patch_size: int = 8) -> np.ndarray
  extract_patch(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray
  normalize_descriptor(descriptor: np.ndarray) -> np.ndarray
  ```
- **Input**: Image, keypoint coordinates
- **Output**: N × D descriptor matrix
- **Implementation Requirements**:
  - Patch extraction (bilinear interpolation)
  - Patch normalization (mean=0, std=1)
  - L2 normalization

---

### **Phase 3: matching/** - Feature Matching

#### `matcher.py`
- **README Section**: Phase 3 > Step 3.1
- **Key Functions**:
  ```python
  match_descriptors(desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.8) -> List[tuple]
  compute_distance_matrix(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray
  apply_ratio_test(distances: np.ndarray, ratio: float) -> List[tuple]
  symmetric_matching(matches_12: List[tuple], matches_21: List[tuple]) -> List[tuple]
  ```
- **Input**: Descriptors from two images
- **Output**: Match list [(idx1, idx2), ...]
- **Implementation Requirements**:
  - Euclidean distance matrix (vectorized)
  - Lowe's ratio test
  - Symmetric matching (optional)

#### `ransac.py`
- **README Section**: Phase 3 > Step 3.2
- **Key Functions**:
  ```python
  estimate_fundamental_matrix_ransac(points1: np.ndarray, points2: np.ndarray, 
                                      iterations: int = 2000, threshold: float = 3.0) -> tuple
  eight_point_algorithm(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray
  normalize_points(points: np.ndarray) -> tuple  # normalized_pts, T_matrix
  compute_sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray
  enforce_rank2_constraint(F: np.ndarray) -> np.ndarray
  ```
- **Input**: 2D point matches
- **Output**: Fundamental matrix F, inlier mask
- **Implementation Requirements**:
  - Implement 8-point algorithm
  - Coordinate normalization (essential!)
  - RANSAC loop
  - Rank-2 constraint enforcement

---

### **Phase 4: geometry/** - Camera Pose Estimation

#### `essential_matrix.py`
- **README Section**: Phase 4 > Step 4.1
- **Key Functions**:
  ```python
  fundamental_to_essential(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray
  enforce_essential_constraints(E: np.ndarray) -> np.ndarray
  ```
- **Input**: Fundamental matrix F, Intrinsic matrices
- **Output**: Essential matrix E
- **Implementation Requirements**:
  - E = K2^T * F * K1
  - Force singular values to (1,1,0) via SVD

#### `pose_recovery.py`
- **README Section**: Phase 4 > Step 4.2
- **Key Functions**:
  ```python
  decompose_essential_matrix(E: np.ndarray) -> List[tuple]  # 4 possible (R, t)
  select_valid_pose(poses: List[tuple], points1: np.ndarray, points2: np.ndarray, 
                    K: np.ndarray) -> tuple  # best (R, t)
  check_chirality(R: np.ndarray, t: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, 
                  K: np.ndarray) -> int  # count of valid points
  ```
- **Input**: Essential matrix E, point correspondences
- **Output**: Rotation R, Translation t
- **Implementation Requirements**:
  - SVD decomposition
  - Generate 4 (R,t) combinations
  - Select best combination via chirality check

---

### **Phase 5: triangulation/** - 3D Point Triangulation

#### `triangulate.py`
- **README Section**: Phase 5 > Step 5.1
- **Key Functions**:
  ```python
  triangulate_points(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray
  dlt_triangulation(P1: np.ndarray, P2: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray
  construct_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray
  ```
- **Input**: Camera matrices P1, P2, 2D point correspondences
- **Output**: 3D points (N × 3)
- **Implementation Requirements**:
  - DLT (Direct Linear Transform)
  - Handle homogeneous coordinates
  - SVD minimization

#### `validation.py`
- **README Section**: Phase 5 > Step 5.2
- **Key Functions**:
  ```python
  compute_reprojection_error(X: np.ndarray, P: np.ndarray, observed_p: np.ndarray) -> float
  check_parallax_angle(X: np.ndarray, C1: np.ndarray, C2: np.ndarray) -> float
  filter_triangulated_points(points_3d: np.ndarray, P1: np.ndarray, P2: np.ndarray, 
                              pts1: np.ndarray, pts2: np.ndarray, 
                              reproj_threshold: float = 5.0) -> np.ndarray
  ```
- **Input**: 3D points, camera matrices, observations
- **Output**: Filtered 3D points
- **Implementation Requirements**:
  - Compute reprojection error
  - Check parallax angle
  - Verify depth positivity

---

### **Phase 6: reconstruction/** - Incremental Reconstruction

#### `incremental.py`
- **README Section**: Phase 6 > Step 6.1, 6.3, 6.4
- **Key Functions**:
  ```python
  select_next_image(reconstructed_points: np.ndarray, image_features: dict, 
                    reconstructed_images: set) -> int
  add_new_points(new_camera_pose: tuple, existing_cameras: List[tuple], 
                 matches: dict, reconstructed_point_ids: set) -> np.ndarray
  incremental_sfm(images: List[np.ndarray], features: List, matches: dict, 
                  K: np.ndarray) -> dict  # main reconstruction loop
  ```
- **Input**: Images, features, matches
- **Output**: Reconstructed cameras, 3D points, track information
- **Implementation Requirements**:
  - Image selection strategy
  - Track management
  - Iterative camera/point addition

#### `pnp.py`
- **README Section**: Phase 6 > Step 6.2
- **Key Functions**:
  ```python
  solve_pnp_ransac(points_3d: np.ndarray, points_2d: np.ndarray, K: np.ndarray,
                   iterations: int = 1000, threshold: float = 5.0) -> tuple
  dlt_pnp(X: np.ndarray, p: np.ndarray, K: np.ndarray) -> tuple  # R, t
  project_to_rotation(R_approx: np.ndarray) -> np.ndarray
  ```
- **Input**: 3D-2D correspondences, intrinsic K
- **Output**: Camera pose (R, t)
- **Implementation Requirements**:
  - DLT-based PnP
  - RANSAC framework
  - Rotation matrix projection (SVD)

---

### **Phase 7: optimization/** - Bundle Adjustment

#### `bundle_adjustment.py`
- **README Section**: Phase 7 > Step 7.1, 7.2
- **Key Functions**:
  ```python
  bundle_adjustment(cameras: List[dict], points_3d: np.ndarray, observations: List[dict],
                    K: np.ndarray, fix_first_camera: bool = True) -> tuple
  parametrize_cameras(cameras: List[dict]) -> np.ndarray  # to axis-angle + translation
  unparametrize_cameras(params: np.ndarray) -> List[dict]  # back to R, t
  compute_residuals(params: np.ndarray, observations: List[dict], K: np.ndarray) -> np.ndarray
  rodrigues_rotation(axis_angle: np.ndarray) -> np.ndarray  # axis-angle to R
  ```
- **Input**: Initial cameras, 3D points, 2D observations
- **Output**: Optimized cameras and points
- **Implementation Requirements**:
  - Construct parameter vector (6*N_cam + 3*N_pts)
  - Residual function
  - Use scipy.optimize.least_squares
  - Implement Rodrigues formula

#### `outlier_filter.py`
- **README Section**: Phase 7 > Step 7.3
- **Key Functions**:
  ```python
  reject_outliers(cameras: List[dict], points_3d: np.ndarray, observations: List[dict],
                  K: np.ndarray, threshold_factor: float = 3.0) -> dict
  compute_mad(errors: np.ndarray) -> float  # Median Absolute Deviation
  adaptive_threshold(errors: np.ndarray, k: float = 3.0) -> float
  ```
- **Input**: Optimized parameters, observations
- **Output**: Cleaned observations
- **Implementation Requirements**:
  - Compute reprojection error
  - Adaptive thresholding (MAD)
  - Iterative refinement

---

### **Phase 8: pointcloud/** - Point Cloud Generation and Post-processing

#### `generator.py`
- **README Section**: Phase 8 > Step 8.1
- **Key Functions**:
  ```python
  extract_point_cloud(points_3d: np.ndarray, cameras: List[dict], 
                      images: List[np.ndarray], observations: List[dict]) -> dict
  compute_point_colors(point_idx: int, observations: List[dict], 
                       images: List[np.ndarray]) -> np.ndarray  # RGB
  save_point_cloud(points: np.ndarray, colors: np.ndarray, filename: str, format: str = 'ply')
  ```
- **Input**: 3D points, camera info, original images
- **Output**: Point cloud with colors
- **Implementation Requirements**:
  - Extract colors from observations
  - Export PLY file format
  - Export NumPy array

#### `filter.py`
- **README Section**: Phase 8 > Step 8.2
- **Key Functions**:
  ```python
  statistical_outlier_removal(points: np.ndarray, k_neighbors: int = 20, 
                               std_ratio: float = 2.0) -> np.ndarray
  compute_knn_distances(points: np.ndarray, k: int) -> np.ndarray
  ```
- **Input**: Raw point cloud
- **Output**: Cleaned point cloud
- **Implementation Requirements**:
  - k-NN distance computation
  - Statistical filtering (mean ± k*std)

#### `visualizer.py`
- **README Section**: Phase 8 > Step 8.3
- **Key Functions**:
  ```python
  visualize_point_cloud(points: np.ndarray, colors: np.ndarray = None, save_path: str = None)
  plot_cameras(cameras: List[dict], points: np.ndarray = None, save_path: str = None)
  show_reprojection_errors(cameras: List[dict], points_3d: np.ndarray, 
                            observations: List[dict], K: np.ndarray)
  draw_camera_frustum(ax, R: np.ndarray, t: np.ndarray, K: np.ndarray, scale: float = 1.0)
  ```
- **Input**: Point cloud, cameras
- **Output**: Matplotlib plots, saved images
- **Implementation Requirements**:
  - 3D scatter plot
  - Draw camera frustums
  - Reprojection visualization

---

## 🔧 Common Utilities: utils/

### `math_utils.py`
- **General Mathematical Functions**:
  ```python
  # Linear Algebra
  svd_solve(A: np.ndarray) -> np.ndarray  # Solve Ax=0 via SVD
  normalize_vector(v: np.ndarray) -> np.ndarray
  skew_symmetric(v: np.ndarray) -> np.ndarray  # [v]_×
  
  # Coordinate transformations
  homogeneous(points: np.ndarray) -> np.ndarray  # 2D/3D -> homogeneous
  euclidean(points: np.ndarray) -> np.ndarray    # homogeneous -> euclidean
  
  # Rotation utilities
  rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray
  axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray
  
  # Matrix operations
  enforce_orthogonal(R: np.ndarray) -> np.ndarray  # Project to SO(3)
  ```

### `io_utils.py`
- **File I/O**:
  ```python
  load_image(filepath: str) -> np.ndarray
  save_image(image: np.ndarray, filepath: str)
  save_ply(points: np.ndarray, colors: np.ndarray, filepath: str)
  load_camera_params(filepath: str) -> dict
  save_reconstruction(cameras: List, points: np.ndarray, filepath: str)
  ```

### `visualization.py`
- **Visualization Helpers**:
  ```python
  plot_features(image: np.ndarray, keypoints: np.ndarray, title: str = None)
  plot_matches(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
               matches: List[tuple], inliers: np.ndarray = None)
  create_figure_grid(n_rows: int, n_cols: int) -> tuple
  ```

---

## 🚀 Main Execution Files

### `config.py`
```python
# Global configuration parameters
class Config:
    # Feature Detection
    HARRIS_K = 0.05
    HARRIS_THRESHOLD = 0.01
    PATCH_SIZE = 8
    
    # Matching
    RATIO_TEST_THRESHOLD = 0.8
    
    # RANSAC
    RANSAC_ITERATIONS = 2000
    RANSAC_THRESHOLD = 3.0
    MIN_INLIERS = 50
    
    # Triangulation
    REPROJ_ERROR_THRESHOLD = 5.0
    MIN_PARALLAX_ANGLE = 1.0  # degrees
    
    # Bundle Adjustment
    BA_MAX_ITERATIONS = 100
    BA_LOSS = 'huber'
    
    # Point Cloud
    KNN_NEIGHBORS = 20
    OUTLIER_STD_RATIO = 2.0
    
    # Camera
    DEFAULT_FOV = 60.0  # degrees
```

### `main.py`
```python
"""
Execute entire SfM pipeline
Run each Phase sequentially and save results
"""
from src.preprocessing import image_loader, camera_calibration
from src.features import harris_detector, descriptor
from src.matching import matcher, ransac
# ... (import all modules)

def main(image_dir: str, output_dir: str):
    # Phase 1: Preprocessing
    images = image_loader.load_images(image_dir)
    K = camera_calibration.estimate_intrinsic_matrix(...)
    
    # Phase 2: Feature Detection
    keypoints_list = []
    descriptors_list = []
    for img in images:
        kpts = harris_detector.detect_harris_corners(img)
        desc = descriptor.compute_descriptors(img, kpts)
        keypoints_list.append(kpts)
        descriptors_list.append(desc)
    
    # Phase 3-8: ... (remaining phases)
    
    # Save results
    save_results(output_dir, point_cloud, cameras)

if __name__ == "__main__":
    main("data/scene1", "output/scene1")
```

---

## 📝 Recommended Development Order

1. **Week 1**: Implement and test Phase 1-2
   - `image_loader.py`, `camera_calibration.py`
   - `harris_detector.py`, `descriptor.py`
   - Verify feature detection on single image

2. **Week 2**: Implement Phase 3-4
   - `matcher.py`, `ransac.py`
   - `essential_matrix.py`, `pose_recovery.py`
   - Test camera pose recovery with two images

3. **Week 3**: Implement Phase 5-6
   - `triangulate.py`, `validation.py`
   - `incremental.py`, `pnp.py`
   - Test incremental reconstruction with 3-5 images

4. **Week 4**: Implement Phase 7-8 and optimization
   - `bundle_adjustment.py`, `outlier_filter.py`
   - `generator.py`, `filter.py`, `visualizer.py`
   - Integrate entire pipeline and debug

5. **Week 5**: Test multiple scenes and write report
   - Generalization testing on Scene 2, 3
   - Result analysis and documentation

---

## 🎯 Testing Methods for Each File

Write each module to be independently testable:

```python
# Example: test_harris_detector.py
if __name__ == "__main__":
    from src.preprocessing.image_loader import load_images
    from src.features.harris_detector import detect_harris_corners
    from utils.visualization import plot_features
    
    images = load_images("data/scene1")
    keypoints = detect_harris_corners(images[0])
    plot_features(images[0], keypoints, "Harris Corners")
```

---

## 📊 Output File Descriptions

### `output/point_clouds/`
- `scene1.ply`: PLY format point cloud (viewable in MeshLab, CloudCompare)
- `scene1.npy`: NumPy array (reusable in Python)

### `output/visualizations/`
- `features_img001.png`: Detected corners visualization
- `matches_001_002.png`: Feature matches visualization
- `cameras_3d.png`: Camera trajectory 3D plot
- `point_cloud_3d.png`: Final point cloud plot

### `output/reports/`
- `reconstruction_log.txt`: Statistics for each step (processing time, point count, errors, etc.)
- `statistics.json`: Statistics in JSON format

---

## ⚠️ Important Notes

1. **All functions use NumPy array input/output**
2. **Write each module to be independently testable**
3. **Docstrings required** (function description, parameters, return values)
4. **Check numerical stability** (especially normalization, SVD)
5. **Save intermediate results** (for easier debugging)

Following this structure enables systematic and modular SfM implementation!
