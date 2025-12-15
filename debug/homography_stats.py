# %%
import os, sys
if __name__ == "__main__":
    __name__ = 'score_learn.evaluate_H.py'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False

from ..load_data import *
from ..model_H import SampsonBM, unnormalize_models
from ..model import unnormalize_points
from ..score_weights import sufficient_statistic_GT
from ..local_optimization import compose_essential_matrix
from ..model_E import SampsonBM as SampsonBM_E
import cv2

if __run__:
    __name__ = "__main__"


def estimate_plane_algebraic(inlier_pts1, inlier_pts2, gt_R, gt_t):
    """
    Estimate plane parameters using algebraic method.
    Solves: λ*pts2 = R*pts1 + t*(m^T*pts1) for m, using λ = [(R*pts1)_z + t_z*(m^T*pts1)]
    
    Returns:
        m: plane parameter vector (n/d) [3]
        lambdas: scale factors [N]
    """
    N = inlier_pts1.size(0)
    R_pts1 = (gt_R @ inlier_pts1.T).T  # [N, 3]
    
    # Build system: each point gives 2 equations (j=0,1)
    A = torch.zeros(2*N, 3, device=inlier_pts1.device, dtype=gt_R.dtype)
    b = torch.zeros(2*N, device=inlier_pts1.device, dtype=gt_R.dtype)
    
    for i in range(N):
        for j in range(2):  # Only j=0,1
            row_idx = 2*i + j
            # Coefficient for m: (t[2]*pts2[i,j] - t[j]) * pts1[i]
            coeff = gt_t[2]*inlier_pts2[i, j] - gt_t[j]
            A[row_idx, :] = coeff * inlier_pts1[i, :]
            # Right hand side: (R*pts1[i])[j] - (R*pts1[i])[2]*pts2[i,j]
            b[row_idx] = R_pts1[i, j] - R_pts1[i, 2]*inlier_pts2[i, j]
    
    # Solve the overdetermined system
    m, _, _, _ = torch.linalg.lstsq(A, b.unsqueeze(1))
    m = m.squeeze()
    
    # Compute λ values
    lambdas = R_pts1[:, 2] + gt_t[2] * (inlier_pts1 @ m.unsqueeze(1)).squeeze()
    
    return m, lambdas


def estimate_plane_triangulation(inlier_pts1, inlier_pts2, gt_R, gt_t):
    """
    Estimate plane parameters by triangulating 3D points and fitting a plane.
    
    Returns:
        m: plane parameter vector (n/d) [3]
        lambdas: scale factors [N] (computed from the fitted plane)
    """
    # Convert to numpy for OpenCV
    pts1_np = inlier_pts1[:, :2].cpu().numpy()  # [N, 2] normalized coords
    pts2_np = inlier_pts2[:, :2].cpu().numpy()
    
    # Construct projection matrices
    # P1 = [I | 0] for first camera (identity)
    P1 = np.eye(3, 4, dtype=np.float64)
    
    # P2 = [R | t] for second camera
    P2 = np.hstack([gt_R.cpu().numpy(), gt_t.cpu().numpy().reshape(3, 1)])
    
    # Triangulate points
    # cv2.triangulatePoints expects 2xN arrays
    points_4d = cv2.triangulatePoints(P1, P2, pts1_np.T, pts2_np.T)  # [4, N]
    
    # Convert to 3D by dividing by homogeneous coordinate
    points_3d = points_4d[:3, :] / points_4d[3:4, :]  # [3, N]
    points_3d = points_3d.T  # [N, 3]
    
    # Fit plane to 3D points using SVD
    # Plane equation: n^T * (X - X_mean) = 0
    X_mean = np.mean(points_3d, axis=0)  # [3]
    X_centered = points_3d - X_mean  # [N, 3]
    
    # SVD: the plane normal is the last right singular vector
    U, S, Vt = np.linalg.svd(X_centered.T @ X_centered)
    n = Vt[-1, :]  # [3] - plane normal (smallest singular value)
    
    # Ensure normal points towards camera (positive Z in first camera frame)
    if n[2] < 0:
        n = -n
    
    # Compute d: distance from origin to plane
    # n^T * X_mean = d
    d = np.dot(n, X_mean)
    
    # Ensure d > 0 (plane in front of camera)
    if d < 0:
        n = -n
        d = -d
    
    # Compute m = n/d
    m = torch.from_numpy(n / d).to(inlier_pts1.device).to(inlier_pts1.dtype)
    
    # Compute λ values using the estimated m
    R_pts1 = (gt_R @ inlier_pts1.T).T
    lambdas = R_pts1[:, 2] + gt_t[2] * (inlier_pts1 @ m.unsqueeze(1)).squeeze()
    
    return m, lambdas

# %%
#_______________________________________________________________________________
#_______________________________________________________________________________

dataset_info = HEB
# scene = 'Alamo'
# scene = 'Piazza_del_Popolo'
scene = 'Ellis_Island'
dataset = H_dataset(dataset_info, scene, padding=True, snn_threshold=None)  # Use all correspondences, not filtered by SNN
loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)

# Task: go over the dataset, filter inlier correspondences only, compute the historgram of Sampson error of inlier correspondences to the GT model, reconstructed from the reference [R,t]
# Plot the histograh
# Collect Sampson errors for inlier correspondences
sampson_errors_H = []  # Homography errors
sampson_errors_E = []  # Essential matrix errors
sampson_errors_H_cv_norm = []  # OpenCV-estimated homography errors (normalized→pixels)
rot_errs_cv = []  # Rotation angular error (deg) between decomposed H and GT R
trans_dir_errs_cv = []  # Translation direction angular error (deg) between decomposed H and GT t
valid_pairs_processed = 0
max_valid_pairs = 100

for idx, data in enumerate(loader):
    if valid_pairs_processed >= max_valid_pairs:
        break
    C = data['correspondences']
    for b in range(C.size(0)):  # Process all images in batch
        pts1 = C[b, :, :3]  # Correspondences from first image, in normalized coordinates
        pts2 = C[b, :, 3:6]  # Correspondences from second image, in normalized coordinates
        inliers = data['inliers'][b]  # Inlier mask
        gt_R = data['gt_R'][b]
        gt_t = data['gt_t'][b]
        K1 = data['K1'][b].float()  # Camera intrinsics for first image
        K2 = data['K2'][b].float()  # Camera intrinsics for second image
        
        # There is no ground truth H in the data, we reconstruct it from GT [R,t] and estimated plane parameters
        # Report statistics for this image
        n_total = len(inliers)
        n_inliers = inliers.sum().item()
        
        # Skip images with too few inliers
        if n_inliers < 50:
            continue
        
        valid_pairs_processed += 1
        if valid_pairs_processed % 10 == 0:
            print(f"Processed {valid_pairs_processed}/{max_valid_pairs} valid pairs...")
        
        # Debug: print details for first few pairs
        if valid_pairs_processed <= 3:
            print(f"\n=== Pair {valid_pairs_processed}: {data['files'][b]} ===")
            print(f"Inliers: {n_inliers}, ||t||: {torch.norm(gt_t).item():.4f}")
            # K sanity: focal lengths and principal points
            fx1, fy1 = K1[0,0].item(), K1[1,1].item()
            cx1, cy1 = K1[0,2].item(), K1[1,2].item()
            fx2, fy2 = K2[0,0].item(), K2[1,1].item()
            cx2, cy2 = K2[0,2].item(), K2[1,2].item()
            # print(f"K1: fx={fx1:.1f}, fy={fy1:.1f}, cx={cx1:.1f}, cy={cy1:.1f}")
            # print(f"K2: fx={fx2:.1f}, fy={fy2:.1f}, cx={cx2:.1f}, cy={cy2:.1f}")
            
        # print(f"\n=== Image {idx}-{b}: {data['files'][b]} ===")
        # print(f"Total correspondences: {n_total}")
        # print(f"Inliers: {n_inliers}")
        # print(f"Inlier fraction: {n_inliers/n_total:.2%}")
        # print(f"||t||: {torch.norm(gt_t).item():.4f}")
        #
        # Estimate surface normal from inlier correspondences
        inlier_pts1 = pts1[inliers].to(gt_R)
        inlier_pts2 = pts2[inliers].to(gt_R)

        # Try both methods and compare
        m_tri, lambdas_tri = estimate_plane_triangulation(inlier_pts1, inlier_pts2, gt_R, gt_t)
        m_alg, lambdas_alg = estimate_plane_algebraic(inlier_pts1, inlier_pts2, gt_R, gt_t)
        
        # Use algebraic method (seems to work better)
        m, lambdas = m_alg, lambdas_alg
        
        # Debug: print plane estimate for first few pairs
        if valid_pairs_processed <= 3:
            print(f"  Estimated m: {m.cpu().numpy()}, ||m||: {torch.norm(m).item():.4f}")
            print(f"  Triangulation m: {m_tri.cpu().numpy()}, ||m_tri||: {torch.norm(m_tri).item():.4f}")
            print(f"  Difference: {torch.norm(m - m_tri).item():.4f}")
        
        # Normalize to get unit normal: n = m / ||m||, d = 1/||m||
        n = m / torch.norm(m)
        d = 1.0 / torch.norm(m)
        
        # print(f"Estimated plane: n={n.cpu().numpy()}, d={d.item():.4f}")
        
        # Verify the estimation: check if inliers satisfy H*pts1 ≈ pts2
        # H = R + t*n^T/d = R + t*m^T
        H_reconstructed = gt_R + torch.outer(gt_t, m)
        
        # Check reconstruction error in normalized coordinates
        pts1_homo = inlier_pts1  # Already homogeneous [n, 3]
        pts2_homo = inlier_pts2  # Already homogeneous [n, 3]
        
        # Apply H to pts1
        pts2_reconstructed = (H_reconstructed @ pts1_homo.T).T  # [n, 3]
        
        # Normalize to compare
        pts2_reconstructed_norm = pts2_reconstructed / (pts2_reconstructed[:, 2:3] + 1e-12)
        pts2_homo_norm = pts2_homo / (pts2_homo[:, 2:3] + 1e-12)
        
        # Compute error
        recon_error = torch.norm(pts2_reconstructed_norm[:, :2] - pts2_homo_norm[:, :2], dim=1)
        
        # print(f"Reconstruction error in normalized coords:")
        # print(f"  Mean: {recon_error.mean().item():.6f}, Median: {recon_error.median().item():.6f}, Max: {recon_error.max().item():.6f}")
        
        # Also check if the linear system was solved correctly
        # Verify: λ[i]*pts2[i] = R*pts1[i] + t*(m^T*pts1[i]) for each inlier
        R_pts1 = (gt_R @ inlier_pts1.T).T  # [N, 3]
        predicted_pts2_homog = R_pts1 + gt_t.unsqueeze(0) * (inlier_pts1 @ m.unsqueeze(1))  # [n, 3]
        actual_pts2_homog = lambdas.unsqueeze(1) * inlier_pts2  # [n, 3]
        
        system_error = torch.norm(predicted_pts2_homog - actual_pts2_homog, dim=1)
        # print(f"Linear system residual:")
        # print(f"  Mean: {system_error.mean().item():.6f}, Max: {system_error.max().item():.6f}")
        
        # Check scale factors λ
        # print(f"Scale factors λ: Mean: {lambdas.mean().item():.6f}, Std: {lambdas.std().item():.6f}, Range: [{lambdas.min().item():.6f}, {lambdas.max().item():.6f}]")

        # Reconstruct GT homography: H = R + t*n^T/d = R + t*m^T
        H_gt = gt_R + torch.outer(gt_t, m) # , reconstructed H_gt in normalized coordinates
        
        # If we have GT H from data, use it instead and compare
        
        # Unnormalize points and homography to pixel coordinates
        # Convert to float32 for consistency
        inlier_pts1_px = unnormalize_points(inlier_pts1.float().unsqueeze(0), K1.unsqueeze(0))[0]  # [n, 3]
        inlier_pts2_px = unnormalize_points(inlier_pts2.float().unsqueeze(0), K2.unsqueeze(0))[0]  # [n, 3]
        # Round-trip K consistency check for early pairs
        if valid_pairs_processed <= 3 and False: # Consistency check has passed
            # Normalize back
            K1_inv = torch.inverse(K1)
            K2_inv = torch.inverse(K2)
            pts1_back = (K1_inv @ inlier_pts1_px.T).T
            pts2_back = (K2_inv @ inlier_pts2_px.T).T
            # Compare homogeneous normalized coords (scale-invariant: use xy/z)
            pts1_back_xy = pts1_back[:, :2] / (pts1_back[:, 2:3] + 1e-12)
            pts2_back_xy = pts2_back[:, :2] / (pts2_back[:, 2:3] + 1e-12)
            inlier_pts1_xy = inlier_pts1[:, :2] / (inlier_pts1[:, 2:3] + 1e-12)
            inlier_pts2_xy = inlier_pts2[:, :2] / (inlier_pts2[:, 2:3] + 1e-12)
            diff1 = torch.norm(pts1_back_xy - inlier_pts1_xy, dim=1)
            diff2 = torch.norm(pts2_back_xy - inlier_pts2_xy, dim=1)
            print(f"  K round-trip diff: mean1={diff1.mean().item():.3e}, mean2={diff2.mean().item():.3e}, max1={diff1.max().item():.3e}, max2={diff2.max().item():.3e}")
        H_gt_px = unnormalize_models(H_gt.float().unsqueeze(0).unsqueeze(0), K1.unsqueeze(0), K2.unsqueeze(0))[0, 0]  # [3, 3]
        
        # Compute residuals (Sampson errors) of inliers in pixel coordinates
        # SampsonBM expects: H [B, M, 3, 3], x [B, n, 3], y [B, n, 3]
        H_batch = H_gt_px.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        x_batch = inlier_pts1_px.unsqueeze(0)  # [1, n, 3]
        y_batch = inlier_pts2_px.unsqueeze(0)  # [1, n, 3]
        residuals_H = SampsonBM(x_batch, y_batch, H_batch)[0, 0]  # Extract [n] from [1, 1, n]
        
        # Also compute Essential matrix Sampson errors
        # E = [t]_x * R (cross product matrix of t times R)
        E_gt = compose_essential_matrix(gt_R, gt_t)  # In normalized coordinates
        
        # For Essential matrix, we need to convert to Fundamental matrix in pixel coords
        # F = K2^(-T) * E * K1^(-1)
        K1_inv = torch.inverse(K1)
        K2_inv = torch.inverse(K2)
        F_gt = K2_inv.T @ E_gt.to(K2.dtype) @ K1_inv  # Fundamental matrix in pixel coords
        
        # Compute Sampson errors w.r.t. Essential/Fundamental matrix
        # Note: SampsonBM_E expects (y, x, F) order, and points should be in pixel coordinates
        F_batch = F_gt.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        residuals_E = SampsonBM_E(y_batch, x_batch, F_batch)[0, 0].abs()  # Note: (y, x, F) order!

        
        # Debug: print residuals for first few pairs
        if valid_pairs_processed <= 3:
            print(f"  Sampson errors H (px): mean={residuals_H.mean().item():.2f}, median={residuals_H.median().item():.2f}, std={residuals_H.std().item():.2f}")
            print(f"  Sampson errors E12 (px): mean={residuals_E.mean().item():.2f}, median={residuals_E.median().item():.2f}, std={residuals_E.std().item():.2f}")
        
        # Filter only inlier correspondences
        sampson_errors_H.extend(residuals_H.cpu().numpy())
        sampson_errors_E.extend(residuals_E.cpu().numpy())

        # Removed: OpenCV homography estimation in pixel domain and GT-aligned H for cleanup

        # Estimate homography directly in normalized coordinates (to validate conversions), then evaluate in pixels
        pts1_norm_np = (inlier_pts1[:, :2] / (inlier_pts1[:, 2:3] + 1e-12)).cpu().numpy()
        pts2_norm_np = (inlier_pts2[:, :2] / (inlier_pts2[:, 2:3] + 1e-12)).cpu().numpy()
        Hn_cv, mask_n = cv2.findHomography(pts1_norm_np, pts2_norm_np, method=cv2.RANSAC, ransacReprojThreshold=1e-3, maxIters=2000, confidence=0.995)
        # Convert normalized homography to pixel domain: H_px = K2 * Hn * K1^{-1}
        K1_np = K1.cpu().numpy(); K2_np = K2.cpu().numpy()
        Hn_cv_px = (K2_np @ Hn_cv @ np.linalg.inv(K1_np)).astype(np.float64)
        Hn_cv_px_t = torch.from_numpy(Hn_cv_px).to(inlier_pts1_px)
        Hn_cv_px_batch = Hn_cv_px_t.unsqueeze(0).unsqueeze(0)
        residuals_Hn_cv_px = SampsonBM(x_batch, y_batch, Hn_cv_px_batch)[0, 0]
        sampson_errors_H_cv_norm.extend(residuals_Hn_cv_px.detach().cpu().numpy())
        if valid_pairs_processed <= 3:
            print(f"  OpenCV H (normalized→pixels) Sampson (px): mean={residuals_Hn_cv_px.mean().item():.4f}, median={residuals_Hn_cv_px.median().item():.4f}")
        # Removed recomposition via decomposition for cleanup
        
        # Decompose OpenCV H to get R,t,n and recompose E, then compute Sampson errors
        # Decompose the OpenCV homography in normalized coordinates
        # Decompose homography: H = R + t*n^T
        # Use OpenCV's decomposeHomographyMat
        retval, rotations, translations, normals = cv2.decomposeHomographyMat(Hn_cv, np.eye(3))
        assert(retval > 0)
        # Choose the solution closest to ground truth
        best_idx = 0
        best_dist = float('inf')
        for i in range(retval):
            R_cv = torch.from_numpy(rotations[i]).to(gt_R)
            t_cv = torch.from_numpy(translations[i].flatten()).to(gt_t)
            dist = torch.norm(R_cv - gt_R).item() + torch.norm(t_cv - gt_t).item()
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        R_cv = torch.from_numpy(rotations[best_idx]).to(gt_R)
        t_cv = torch.from_numpy(translations[best_idx].flatten()).to(gt_t)
        n_cv = torch.from_numpy(normals[best_idx].flatten()).to(gt_R)
        
        # Compose Essential matrix from decomposed R,t
        E_cv = compose_essential_matrix(R_cv, t_cv)
        
        # Convert to Fundamental matrix: F = K2^(-T) * E * K1^(-1)
        F_cv = K2_inv.T @ E_cv.to(K2.dtype) @ K1_inv
        
        # Compute Sampson errors
        F_cv_batch = F_cv.unsqueeze(0).unsqueeze(0)
        residuals_E_cv = SampsonBM_E(y_batch, x_batch, F_cv_batch)[0, 0].abs()
        
        # Compute angular errors vs GT pose
        def rotation_error_deg(R_gt_t, R_est_t):
            R2R1 = R_gt_t @ R_est_t.T
            cos_angle = np.clip(0.5 * (np.trace(R2R1.cpu().numpy()) - 1.0), -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_angle)))
        def translation_dir_error_deg(t_gt_t, t_est_t):
            eps = 1e-15
            u1 = (t_gt_t.flatten() / (t_gt_t.norm() + eps)).cpu().numpy()
            u2 = (t_est_t.flatten() / (t_est_t.norm() + eps)).cpu().numpy()
            cos_angle = np.clip(np.dot(u1, u2), -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_angle)))
        rot_err = rotation_error_deg(gt_R, R_cv)
        trans_err = translation_dir_error_deg(gt_t, t_cv)
        rot_errs_cv.append(rot_err)
        trans_dir_errs_cv.append(trans_err)
        if valid_pairs_processed <= 3:
            print(f"  OpenCV H decomposed E Sampson (px): mean={residuals_E_cv.mean().item():.4f}, median={residuals_E_cv.median().item():.4f}")
            print(f"  Pose errors: rot_err={rot_err:.2f} deg, trans_dir_err={trans_err:.2f} deg")


        # Print statistics for this image
        # print(f"Sampson errors for {n_inliers} inliers (pixels):")
        # print(f"  Mean: {residuals.mean().item():.4f}, Median: {residuals.median().item():.4f}, Std: {residuals.std().item():.4f}")
        
        # Only process one good image
        # break

# Convert to numpy array
sampson_errors_H = np.array(sampson_errors_H)
sampson_errors_E = np.array(sampson_errors_E)
sampson_errors_H_cv_norm = np.array(sampson_errors_H_cv_norm)

print(f"\n{'='*60}")
print(f"FINAL STATISTICS - {scene}")
print(f"{'='*60}")
print(f"Valid pairs processed: {valid_pairs_processed}")
print(f"Total inlier correspondences: {len(sampson_errors_H)}")
print(f"\nHomography Sampson Error Statistics (pixels):")
print(f"  Mean:   {np.mean(sampson_errors_H):.4f}")
print(f"  Median: {np.median(sampson_errors_H):.4f}")
print(f"  Std:    {np.std(sampson_errors_H):.4f}")
print(f"  Min:    {np.min(sampson_errors_H):.4f}")
print(f"  Max:    {np.max(sampson_errors_H):.4f}")
print(f"  90th percentile: {np.percentile(sampson_errors_H, 90):.4f}")
print(f"  95th percentile: {np.percentile(sampson_errors_H, 95):.4f}")
print(f"  99th percentile: {np.percentile(sampson_errors_H, 99):.4f}")
print(f"\nEssential Matrix Sampson Error Statistics (pixels):")
print(f"  Mean:   {np.mean(sampson_errors_E):.4f}")
print(f"  Median: {np.median(sampson_errors_E):.4f}")
print(f"  Std:    {np.std(sampson_errors_E):.4f}")
print(f"  Min:    {np.min(sampson_errors_E):.4f}")
print(f"  Max:    {np.max(sampson_errors_E):.4f}")
print(f"  90th percentile: {np.percentile(sampson_errors_E, 90):.4f}")
print(f"  95th percentile: {np.percentile(sampson_errors_E, 95):.4f}")
print(f"  99th percentile: {np.percentile(sampson_errors_E, 99):.4f}")
if sampson_errors_H_cv_norm.size > 0:
    print(f"\nOpenCV Homography (normalized→pixels) Sampson Error Statistics (pixels):")
    print(f"  Mean:   {np.mean(sampson_errors_H_cv_norm):.4f}")
    print(f"  Median: {np.median(sampson_errors_H_cv_norm):.4f}")
    print(f"  Std:    {np.std(sampson_errors_H_cv_norm):.4f}")
    print(f"  Min:    {np.min(sampson_errors_H_cv_norm):.4f}")
    print(f"  Max:    {np.max(sampson_errors_H_cv_norm):.4f}")
if len(rot_errs_cv) > 0:
    print(f"\nOpenCV H Decomposition Pose Error (deg):")
    print(f"  Rotation: mean={np.mean(rot_errs_cv):.2f}, median={np.median(rot_errs_cv):.2f}, std={np.std(rot_errs_cv):.2f}")
    print(f"  Translation direction: mean={np.mean(trans_dir_errs_cv):.2f}, median={np.median(trans_dir_errs_cv):.2f}, std={np.std(trans_dir_errs_cv):.2f}")
print(f"{'='*60}\n")

# Plot histogram
import matplotlib.pyplot as plt
num_plots = 2
if sampson_errors_H_cv_norm.size > 0:
    num_plots += 1
fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots + 2, 6))

# Homography errors
axes[0].hist(sampson_errors_H, bins=100, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Sampson Error (pixels)')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Homography Sampson Errors - {scene}')
axes[0].grid(True, alpha=0.3)

# Essential matrix errors
axes[1].hist(sampson_errors_E, bins=100, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Sampson Error (pixels)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Essential Matrix Sampson Errors - {scene}')
axes[1].grid(True, alpha=0.3)

plot_idx = 2
if sampson_errors_H_cv_norm.size > 0:
    axes[plot_idx].hist(sampson_errors_H_cv_norm, bins=100, edgecolor='black', alpha=0.7, color='teal')
    axes[plot_idx].set_xlabel('Sampson Error (pixels)')
    axes[plot_idx].set_ylabel('Frequency')
    axes[plot_idx].set_title(f'OpenCV H (normalized→pixels) Sampson - {scene}')
    axes[plot_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'sampson_histogram_{scene}.png', dpi=150)
print(f"Histogram saved to: sampson_histogram_{scene}.png")
plt.show()

# Print statistics
# print(f"Total inlier correspondences: {len(sampson_errors)}")
# print(f"Mean Sampson error: {np.mean(sampson_errors):.4f} pixels")
# print(f"Median Sampson error: {np.median(sampson_errors):.4f} pixels")
# print(f"Std Sampson error: {np.std(sampson_errors):.4f} pixels")

# %%
