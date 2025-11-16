# Structure from Motion (SfM) - Vanilla Implementation

## Project Overview
A pure Python/NumPy implementation of Structure from Motion system based on mathematical principles.
Reconstructs 3D point clouds from multi-view images without external computer vision libraries (e.g., OpenCV).

---

## Phase 1: Image Preprocessing and Preparation

### Step 1.1: Image Loading and Normalization
**Objective**: Load input images into memory and convert to processable format

**Input**: 
- Raw image files (HEIC)

**Output**:
- Normalized grayscale image arrays (0-1 range)
- Image metadata (width, height, capture order)

**Mathematical Background**:
- RGB to Grayscale: `I_gray = 0.299*R + 0.587*G + 0.114*B`
- Normalization: `I_norm = I / 255.0`

**Implementation Details**:
- Read images using PIL/imageio (I/O only)
- Convert to grayscale via weighted channel averaging
- Resample all images to same resolution (implement bilinear interpolation manually)

---

### Step 1.2: Camera Intrinsic Parameter Estimation
**Objective**: Compute camera intrinsic matrix K

**Input**:
- Image resolution (width, height)
- Estimated FOV (Field of View) or focal length

**Output**:
- 3x3 Intrinsic matrix K
```
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
```

**Mathematical Background**:
- Focal length estimation: `fx = fy = width / (2 * tan(FOV/2))`
- Principal point: `cx = width/2, cy = height/2`

**Implementation Details**:
- Default smartphone FOV: 60-70 degrees
- Square pixel assumption (fx = fy)
- Center alignment assumption (principal point at image center)

---

## Phase 2: Feature Detection

### Step 2.1: Harris Corner Detection
**Objective**: Detect corner points in each image

**Input**:
- Grayscale image I

**Output**:
- List of corner coordinates [(x1, y1), (x2, y2), ...]
- Corner response values

**Mathematical Background**:
1. Image gradients:
   - `Ix = ∂I/∂x` (Sobel filter: [-1, 0, 1])
   - `Iy = ∂I/∂y` (Sobel filter: [-1, 0, 1]^T)

2. Structure tensor (per pixel):
   ```
   M = [Σ(Ix²)    Σ(IxIy)]
       [Σ(IxIy)   Σ(Iy²) ]
   ```
   (Σ is Gaussian weighted sum over local window)

3. Corner response:
   - `R = det(M) - k*trace(M)²`  (k = 0.04-0.06)
   - `det(M) = λ1*λ2`, `trace(M) = λ1 + λ2`

4. Non-maximum suppression: Select only local maxima in 3x3 windows

**Implementation Details**:
- Sobel kernel: `[[-1,0,1],[-2,0,2],[-1,0,1]]` implemented manually
- Gaussian window: σ=1.5, 5x5 kernel
- Threshold: R > 0.01 * max(R)
- NMS window size: 3x3 or 5x5
- Target per image: 500-2000 corners

---

### Step 2.2: Feature Descriptor Generation
**Objective**: Encode neighborhood characteristics of each keypoint as a vector

**Input**:
- Image I
- Keypoint coordinates
- Keypoint scale/orientation (optional)

**Output**:
- N × D descriptor matrix (N: number of keypoints, D: descriptor dimension)

**Mathematical Background**:
- Simple Patch Descriptor:
  - Extract w×w patch centered at each keypoint (e.g., 8x8 = 64 dimensions)
  - Normalize patch: `(patch - mean) / std`
  - Flatten to 1D vector

- Rotation invariance (optional):
  - Compute dominant angle using gradient orientation histogram
  - Rotate patch by dominant angle

**Implementation Details**:
- Patch size: 8x8 or 16x16
- Extract patches using bilinear interpolation
- L2 normalization: `descriptor = descriptor / ||descriptor||`
- Border handling: Exclude keypoints near image boundaries

---

## Phase 3: Feature Matching

### Step 3.1: Brute-force Descriptor Matching
**Objective**: Find feature correspondences between two images

**Input**:
- Descriptors from Image 1: D1 (N1 × d)
- Descriptors from Image 2: D2 (N2 × d)

**Output**:
- List of matches: [(idx1_a, idx2_a), (idx1_b, idx2_b), ...]
- Match distances

**Mathematical Background**:
1. Distance matrix computation:
   - Euclidean distance: `dist[i,j] = ||D1[i] - D2[j]||²`
   - Efficient computation: `||a-b||² = ||a||² + ||b||² - 2*a·b`

2. Nearest neighbor matching:
   - For each D1[i], find closest D2[j]
   - `j* = argmin_j dist[i,j]`

3. Ratio test (Lowe's ratio):
   - 1st nearest: d1, 2nd nearest: d2
   - Accept if `d1 / d2 < 0.8`
   - Removes false matches

**Implementation Details**:
- Vectorized distance computation (NumPy broadcasting)
- Ratio test threshold: 0.75-0.8
- Symmetric matching (optional): Accept only if i↔j is best match in both directions
- Minimum match count: 50+

---

### Step 3.2: Outlier Removal via RANSAC
**Objective**: Remove geometric outliers while estimating fundamental matrix

**Input**:
- Initial matches: [(p1, p2), ...] (pixel coordinates)
- Intrinsic matrix K

**Output**:
- Inlier matches
- Fundamental matrix F or Essential matrix E

**Mathematical Background**:
1. Epipolar constraint:
   - `p2^T * F * p1 = 0`
   - F: 3×3 fundamental matrix (rank 2)

2. 8-point algorithm:
   - Estimate F from 8 point pairs
   - Normalize coordinates (critical!): 
     - Shift mean to origin, average distance = √2
     - `T * p → p'`, compute after normalization, inverse transform
   - Linear system: `Af = 0` (A is 8×9 matrix)
   - Solution via SVD: `f = last column of V`
   - Enforce rank-2 constraint: SVD(F), set smallest singular value to 0

3. RANSAC:
   ```
   best_inliers = []
   for iteration in 1..N:
       sample 8 random matches
       compute F using 8-point
       count inliers: |p2^T*F*p1| < threshold
       if inliers > best_inliers:
           update best_F, best_inliers
   recompute F using all inliers
   ```

**Implementation Details**:
- RANSAC iterations: 1000-2000
- Inlier threshold: 1-3 pixels (Sampson distance)
- Minimum inliers: 30-50
- Coordinate normalization is essential!

---

## Phase 4: Camera Pose Estimation

### Step 4.1: Essential Matrix Recovery
**Objective**: Compute Essential matrix from Fundamental matrix

**Input**:
- Fundamental matrix F
- Intrinsic matrices K1, K2

**Output**:
- Essential matrix E

**Mathematical Background**:
```
E = K2^T * F * K1
```

Essential matrix properties:
- E = [t]_× * R (rotation R, translation t)
- Rank 2, two equal singular values
- Normalize via SVD: `E = U*diag(1,1,0)*V^T`

**Implementation Details**:
- If K1, K2 are same (same camera): E = K^T * F * K
- Force singular values to (1, 1, 0) after SVD
- Enforce essential matrix constraints

---

### Step 4.2: Camera Pose Recovery (R, t)
**Objective**: Extract relative camera pose from Essential matrix

**Input**:
- Essential matrix E
- Inlier point correspondences

**Output**:
- Rotation matrix R (3×3)
- Translation vector t (3×1, unit vector)

**Mathematical Background**:
1. SVD decomposition: `E = U*Σ*V^T`

2. Four possible solutions:
   ```
   W = [0  -1   0]
       [1   0   0]
       [0   0   1]
   
   R1 = U*W*V^T
   R2 = U*W^T*V^T
   t1 = U[:, 2]  (3rd column of U)
   t2 = -U[:, 2]
   
   4 combinations: (R1,t1), (R1,t2), (R2,t1), (R2,t2)
   ```

3. Chirality check:
   - Triangulate points for each combination
   - Count points with positive depth in both cameras
   - Select combination with most points in front

**Implementation Details**:
- Verify det(R) = 1 (rotation matrix)
- If det(R) = -1, set R = -R (remove reflection)
- At least 75% of points must have positive depth

---

## Phase 5: 3D Point Triangulation

### Step 5.1: Two-view Triangulation
**Objective**: Recover 3D points from corresponding 2D points in two views

**Input**:
- Camera matrices: P1 = K[I|0], P2 = K[R|t]
- Matched 2D points: p1, p2

**Output**:
- 3D point X (world coordinates)

**Mathematical Background**:
1. Projection equation:
   - `λ1*p1 = P1*X`
   - `λ2*p2 = P2*X`

2. DLT (Direct Linear Transform):
   - Cross product: `p × (PX) = 0`
   - Skew-symmetric matrix:
     ```
     [p]_× = [ 0   -1   py]
             [ 1    0  -px]
             [-py  px   0 ]
     ```
   - Linear system: `AX = 0` (4×4 matrix, 2 equations per view)
   - Solution via SVD: X = last column of V, normalize by X[3]

3. Alternative - Midpoint method:
   - Compute ray from each view
   - Find closest point between two rays (least squares)

**Implementation Details**:
- Convert points to homogeneous coordinates: `[x, y, 1]^T`
- Construct 4 equations (2 from each view)
- SVD minimization
- Validate using reprojection error
- Remove infinite points (X[3] ≈ 0)

---

### Step 5.2: Triangulation Quality Validation
**Objective**: Filter poorly triangulated points

**Input**:
- 3D points
- Original 2D observations
- Camera matrices

**Output**:
- Filtered 3D points (good quality only)

**Mathematical Background**:
1. Reprojection error:
   - `p'1 = P1*X / (P1*X)[2]`  (project back)
   - `error1 = ||p1 - p'1||`
   - Total error = `error1 + error2`

2. Angle check:
   - Compute viewing angle from both cameras
   - Unreliable if parallax angle too small (< 2 degrees)

**Implementation Details**:
- Reprojection error threshold: 2-5 pixels
- Minimum parallax angle: 1-2 degrees
- Depth positivity: depth > 0 in both cameras
- Distance bounds: Remove extremely far or close points

---

## Phase 6: Incremental Reconstruction

### Step 6.1: Next Image Selection
**Objective**: Select optimal next image to add to reconstruction

**Input**:
- Existing reconstructed 3D points
- 2D features from each image
- Matched features between images

**Output**:
- Index of next image to add

**Selection Criteria**:
1. Maximum 2D-3D correspondences with existing 3D points
2. Sufficient baseline (avoid too close views)
3. Match quality (inlier ratio)

**Implementation Details**:
- Compute visibility score for each remaining image
- Score = (number of observed 3D points) × (match quality)
- Prioritize next/previous frames for sequential captures

---

### Step 6.2: New Camera Estimation via PnP
**Objective**: Estimate camera pose from 2D-3D correspondences

**Input**:
- 3D points: X_i (world coordinates)
- 2D observations: p_i (image coordinates)
- Intrinsic matrix K

**Output**:
- Camera pose: R, t

**Mathematical Background**:
1. DLT-based PnP:
   - Projection: `λ*p = K*[R|t]*X`
   - Rearrange: `[R|t] * X_homogeneous = K^(-1) * λ*p`
   - Construct linear system with 6+ points
   - Estimate [R|t] via SVD
   - Project R to valid rotation (SVD, enforce orthonormal)

2. RANSAC-PnP:
   - Minimal sample: 6 points
   - Inlier counting
   - Refine with all inliers

**Implementation Details**:
- Minimum points: 6 (DLT)
- RANSAC iterations: 500-1000
- Inlier threshold: 3-5 pixels
- R normalization: `U*V^T` from SVD(R)

---

### Step 6.3: Add New 3D Points
**Objective**: Triangulate additional 3D points from new camera

**Input**:
- Newly added camera pose
- Existing cameras
- Untriangulated 2D matches

**Output**:
- New 3D points

**Process**:
1. Check matches between new image and existing images
2. Select matches not yet reconstructed in 3D
3. Perform two-view triangulation
4. Add only points passing quality checks

**Implementation Details**:
- Each 2D point triangulated at most once
- Consider multi-view triangulation (if observed in multiple views)
- Incremental addition with track management

---

### Step 6.4: Iteration
**Objective**: Repeat until all images are added

**Iteration Process**:
```
while images remain to be added:
    next_image = select_next_image()
    if insufficient 2D-3D matches:
        break
    
    R, t = estimate_pose_PnP(next_image)
    add_camera(R, t)
    
    new_3D_points = triangulate_new_points(next_image)
    add_points(new_3D_points)
    
    # Optional: local bundle adjustment
```

**Implementation Details**:
- Failure handling: Skip if match count insufficient
- Track management: Record which images observe each 3D point
- Progressive validation: Prevent gradual quality degradation

---

## Phase 7: Bundle Adjustment

### Step 7.1: Reprojection Error Definition
**Objective**: Define objective function to optimize

**Mathematical Background**:
- Projection of 3D point i in camera j:
  ```
  p_ij = π(K * [R_j|t_j] * X_i)
  where π(x,y,z) = (x/z, y/z)
  ```

- Reprojection error:
  ```
  e_ij = ||observed_p_ij - projected_p_ij||²
  ```

- Total cost:
  ```
  C = Σ_i Σ_j e_ij²
  ```
  (over all observed point-camera pairs)

**Implementation Details**:
- Visibility matrix: Which points are seen by which cameras
- Residual vector: Concatenate all errors
- Jacobian required (for optimization)

---

### Step 7.2: Optimization
**Objective**: Jointly optimize camera poses and 3D points

**Input**:
- Initial camera poses: {R_j, t_j}
- Initial 3D points: {X_i}
- 2D observations

**Output**:
- Optimized camera poses
- Optimized 3D points

**Mathematical Background**:
1. Parameter vector:
   - Camera: Parametrize each camera as 6D (axis-angle 3D + translation 3D)
   - Points: Each point is 3D (x,y,z)
   - Total: `6*N_cameras + 3*N_points` parameters

2. Levenberg-Marquardt:
   - Iterative: `θ_{k+1} = θ_k - (J^T*J + λ*I)^(-1) * J^T * r`
   - J: Jacobian (residuals w.r.t. parameters)
   - r: residual vector
   - λ: damping factor

3. Sparse structure:
   - J is very sparse (each residual depends on only 1 camera + 1 point)
   - Efficient solution via Schur complement

**Implementation Details**:
- Use scipy.optimize.least_squares
- Method: 'trf' or 'lm'
- Loss: 'huber' (outlier robustness)
- Max iterations: 50-100
- Fix first camera (remove gauge freedom)
- Convert between axis-angle ↔ rotation matrix using Rodrigues formula

---

### Step 7.3: Outlier Rejection
**Objective**: Remove observations with large errors after optimization

**Input**:
- Optimized parameters
- Reprojection errors

**Output**:
- Cleaned observation set

**Process**:
1. Compute all reprojection errors
2. Set threshold (e.g., median + 3*MAD)
3. Remove observations exceeding threshold
4. Re-run BA (optional)

**Implementation Details**:
- Adaptive threshold: `median(errors) + k*MAD(errors)`
- k = 2-4
- Iterative refinement: Repeat multiple times
- Minimum track length: Each 3D point must be observed in at least 2 views

---

## Phase 8: Point Cloud Generation and Post-processing

### Step 8.1: Point Cloud Extraction
**Objective**: Convert optimized 3D points to point cloud format

**Input**:
- Optimized 3D points
- (Optional) Color information

**Output**:
- Point cloud: N × 3 array (x, y, z)
- (Optional) Colors: N × 3 array (r, g, b)

**Color Extraction (Optional)**:
- Find images observing each 3D point
- Extract pixel color from each image
- Compute average/median color

**Implementation Details**:
- Verify coordinate system (first camera at origin)
- Handle scale ambiguity (no absolute scale)
- Point cloud file format: PLY, XYZ, or NumPy array

---

### Step 8.2: Statistical Outlier Removal
**Objective**: Remove noise points

**Input**:
- Raw point cloud

**Output**:
- Cleaned point cloud

**Mathematical Background**:
- For each point:
  - Find k-nearest neighbors (k=10-50)
  - Compute mean distance `d_mean`
  - Global statistics: `μ = mean(d_mean)`, `σ = std(d_mean)`
  - Outlier if `d_mean > μ + 3*σ`

**Implementation Details**:
- k-NN: Brute-force distance computation or KD-tree (optional)
- Standard deviations: 2-3 sigma
- Preserve structure: Don't remove too aggressively

---

### Step 8.3: Visualization
**Objective**: Visualize results

**Output Forms**:
1. 3D point cloud plot
2. Camera positions and orientations
3. Reprojection examples

**Implementation Tools**:
- Matplotlib 3D scatter plot
- Draw camera frustums
- Save to PLY file (for external viewers)

---

## Complete Pipeline Summary

```
[Phase 1] Image Preprocessing
  └─ Image loading, grayscale conversion, camera calibration

[Phase 2] Feature Detection
  ├─ Harris corner detection
  └─ Patch descriptor generation

[Phase 3] Feature Matching
  ├─ Brute-force matching + ratio test
  └─ RANSAC + Fundamental matrix

[Phase 4] Initial Camera Pose
  ├─ Essential matrix recovery
  └─ R, t extraction (select from 4 solutions)

[Phase 5] Initial Triangulation
  ├─ Two-view triangulation (DLT)
  └─ Quality filtering

[Phase 6] Incremental Reconstruction
  ├─ Next image selection
  ├─ Pose estimation via PnP
  ├─ New points triangulation
  └─ Repeat until all images added

[Phase 7] Bundle Adjustment
  ├─ Reprojection error minimization
  ├─ Levenberg-Marquardt optimization
  └─ Outlier rejection

[Phase 8] Point Cloud Generation
  ├─ 3D points extraction
  ├─ Statistical outlier removal
  └─ Visualization & export
```

---

## Implementation Considerations

### Numerical Stability
- **Coordinate normalization**: Essential for Fundamental matrix estimation
- **Unit vectors**: Normalize translation to unit vector
- **Condition number**: Check if matrices are ill-conditioned
- **Homogeneous coordinates**: Normalize by last element

### Degeneracy Cases
- **Pure rotation**: Triangulation fails if translation ≈ 0
- **Planar scenes**: Fundamental matrix estimation difficult
- **Low texture**: Feature detection/matching fails

### Performance
- **Vectorization**: Maximize NumPy broadcasting usage
- **Early termination**: Terminate RANSAC early when sufficient inliers found
- **Sparse matrices**: Use scipy.sparse for bundle adjustment

### Debugging
- **Step-by-step visualization**: Monitor progress
- **Sanity checks**: Matrix determinant, orthogonality, etc.
- **Save intermediate results**: Enable tracking when issues arise

---

## Expected Results

### Minimum Requirements
- Sparse point cloud (hundreds to thousands of points) from 10-30 images
- Camera trajectory visualization
- Reprojection error < 5 pixels

### Good Results
- Thousands to tens of thousands of points
- Multiple scene testing
- Reprojection error < 2 pixels
- 90%+ inlier ratio after BA

---

## References

### Mathematical Background
- Multiple View Geometry (Hartley & Zisserman)
- Computer Vision: Algorithms and Applications (Szeliski)

### Algorithms
- Fundamental matrix: 8-point algorithm (Hartley 1997)
- Essential matrix decomposition (Hartley & Zisserman Ch. 9)
- Bundle adjustment: Sparse BA (Triggs et al. 1999)

### Implementation Tips
- Validate output of each step
- Always check numerical stability
- Incremental debugging (step-by-step verification)
