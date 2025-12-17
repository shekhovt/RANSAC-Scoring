# %%
# HPatches loader + SIFT detection + GT-based matching + Sampson error histogram
# Paste into a notebook or run as a script.
# Requirements: opencv-python, numpy, matplotlib, pillow. Optional: scipy (for KDTree).

import os, sys
if __name__ == "__main__":
    __name__ = 'score_learn.evaluate_H.py'
    __package__ = 'score_learn'
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2

use_kdtree = False
try:
    from scipy.spatial import cKDTree as KDTree
    use_kdtree = True
except Exception:
    use_kdtree = False

try:
    from scipy.stats import chi
    has_scipy_stats = True
except Exception:
    has_scipy_stats = False

from .model_H import SampsonBM
from .model_E import SampsonBM as SampsonBM_E
from .drawing import savefig

# ========== USER PARAMETERS ==========
ROOT = "data/hpatches-sequences-release"                 # set path here (string), or leave None to be prompted
MAX_KEYPOINTS = 2000        # max keypoints per image
NN_DIST_THRESHOLD = 5     # pixels: max NN distance to accept match
UNIQ_DIST_THRESHOLD = 5     # pixels: max NN distance to accept match
# DESCRIPTOR_RATIO_THRESHOLD = 0.9  # Lowe's ratio test threshold for descriptor matching
# REPROJECTION_BARRIER_THRESHOLD = 10.0  # pixels: max reprojection error barrier for geometric filtering
DESCRIPTOR_DISTANCE_THRESHOLD = 400.0  # L2 distance: max descriptor distance to keep a match
USE_DESCRIPTORS = True      # whether to compute SIFT descriptors (needed for descriptor-based matching)
VERBOSE = True
HIST_RANGE = (0.0, 3.0)    # histogram x-range in px
HIST_BINS = 100
NUM_MIXTURE_COMPONENTS = 3  # number of chi(2) components in mixture model
RECOMPUTE = False           # ignore cache and recompute residuals
RECOMPUTE_RES = False         # ignore cache and recompute residuals
SUMMARY = True             # print per-sequence image size and avg keypoints
MAX_IMAGE_PAIRS = None       # max image pairs per sequence (None = all)
INCLUDE_I_SEQUENCES = False   # include illumination sequences (i_*)
INCLUDE_V_SEQUENCES = True   # include viewpoint sequences (v_*)
# =====================================
import random
from tqdm import tqdm
random.seed(42)  # For reproducibility

def input_root():
    if ROOT:
        return ROOT
    else:
        p = input("Enter path to HPatches root folder (contains sequence subfolders): ").strip()
        return p

def get_image_path(folder, base_name):
    """Return existing image path trying multiple extensions.
    base_name should be without extension, e.g., '1' or '2'.
    Supported: png, ppm, jpg, jpeg.
    """
    exts = ("png", "ppm", "jpg", "jpeg")
    for ext in exts:
        p = os.path.join(folder, f"{base_name}.{ext}")
        if os.path.isfile(p):
            return p
    return None

def load_homography(path):
    return np.loadtxt(path).astype(np.float64)

def warp_point(H, p):
    v = np.array([p[0], p[1], 1.0], dtype=np.float64)
    w = H @ v
    return (w[0]/w[2], w[1]/w[2])

def invert_homography(H):
    return np.linalg.inv(H)

def detect_sift_keypoints(image_np, max_kp=MAX_KEYPOINTS, compute_descriptors=True):
    gray = image_np if image_np.ndim == 2 else cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    try:
        sift = cv2.SIFT_create()
    except Exception:
        sift = cv2.xfeatures2d.SIFT_create()
    
    if compute_descriptors:
        kps, descs = sift.detectAndCompute(gray, None)
        if descs is None or len(kps) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128)
        # Sort by response and limit
        indices = np.argsort([-kp.response for kp in kps])[:max_kp]
        kps = [kps[i] for i in indices]
        descs = descs[indices]
        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float64)
        return pts, descs
    else:
        # Only detect keypoints, no descriptors
        kps = sift.detect(gray, None)
        if len(kps) == 0:
            return np.array([]).reshape(0, 2), None
        # Sort by response and limit
        indices = np.argsort([-kp.response for kp in kps])[:max_kp]
        kps = [kps[i] for i in indices]
        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float64)
        return pts, None

def nearest_neighbors(pts_query, pts_db):
    if pts_db.shape[0] == 0 or pts_query.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if use_kdtree:
        tree = KDTree(pts_db)
        dists, idxs = tree.query(pts_query, k=1)
        return idxs, dists
    else:
        dif = pts_db[None,:,:] - pts_query[:,None,:]
        d2 = np.sum(dif*dif, axis=2)
        idxs = np.argmin(d2, axis=1)
        dists = np.sqrt(d2[np.arange(d2.shape[0]), idxs])
        return idxs, dists

def match_descriptors_ratio_test(desc1, desc2, pts1, pts2, H):
    """Match descriptors using Lowe's ratio test with geometric barrier (GPU-accelerated).
    Returns indices of matches in desc1 and desc2.
    
    Args:
        desc1: Descriptors from image 1 (N1 x 128)
        desc2: Descriptors from image 2 (N2 x 128)
        pts1: Keypoint coordinates from image 1 (N1 x 2)
        pts2: Keypoint coordinates from image 2 (N2 x 2)
        H: Homography matrix from image 1 to image 2
        ratio_threshold: Lowe's ratio test threshold
        reproj_barrier_threshold: Reprojection error threshold for barrier (pixels)
    """
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    desc1_t = torch.from_numpy(desc1).float().to(device)  # N1 x 128
    desc2_t = torch.from_numpy(desc2).float().to(device)  # N2 x 128
    pts1_t = torch.from_numpy(pts1).float().to(device)    # N1 x 2
    pts2_t = torch.from_numpy(pts2).float().to(device)    # N2 x 2
    H_t = torch.from_numpy(H).float().to(device)          # 3 x 3
    
    # Compute pairwise descriptor distances: (N1 x 128) - (N2 x 128) -> N1 x N2
    desc_dists = torch.cdist(desc1_t, desc2_t, p=2)  # N1 x N2
    
    # Compute reprojection errors for all pairs
    # Warp all pts1 using H
    pts1_h = torch.cat([pts1_t, torch.ones(pts1_t.shape[0], 1, device=device)], dim=1)  # N1 x 3
    warped_h = (H_t @ pts1_h.T).T  # N1 x 3
    warped = warped_h[:, :2] / warped_h[:, 2:3]  # N1 x 2
    
    # Compute pairwise reprojection errors: N1 x N2
    reproj_dists = torch.cdist(warped.unsqueeze(0), pts2_t.unsqueeze(0), p=2).squeeze(0)  # N1 x N2
    
    # Add barrier to descriptor distances where reprojection error is too high
    barrier_value = 1e6
    barrier_mask = reproj_dists > reproj_barrier_threshold
    combined_dists = desc_dists + barrier_value * barrier_mask.float()
    
    # For each descriptor in desc1, find 2 smallest distances
    # topk returns (values, indices) for k smallest elements
    if combined_dists.shape[1] >= 2:
        top2_dists, top2_indices = torch.topk(combined_dists, k=2, dim=1, largest=False)  # N1 x 2
        
        # Lowe's ratio test: dist_nearest < ratio * dist_second_nearest
        nearest_dists = top2_dists[:, 0]
        second_dists = top2_dists[:, 1]
        nearest_indices = top2_indices[:, 0]
        
        # Check if nearest match passes barrier and ratio test
        nearest_has_barrier = barrier_mask[torch.arange(barrier_mask.shape[0], device=device), nearest_indices]
        ratio_test = nearest_dists < ratio_threshold * second_dists
        valid_mask = (~nearest_has_barrier) & ratio_test
        
        matches_idx1 = torch.where(valid_mask)[0].cpu().numpy()
        matches_idx2 = nearest_indices[valid_mask].cpu().numpy()
    elif combined_dists.shape[1] == 1:
        # Only one descriptor in desc2
        nearest_indices = torch.zeros(combined_dists.shape[0], dtype=torch.long, device=device)
        nearest_has_barrier = barrier_mask[:, 0]
        valid_mask = ~nearest_has_barrier
        
        matches_idx1 = torch.where(valid_mask)[0].cpu().numpy()
        matches_idx2 = nearest_indices[valid_mask].cpu().numpy()
    else:
        matches_idx1 = np.array([], dtype=int)
        matches_idx2 = np.array([], dtype=int)
    
    return matches_idx1, matches_idx2

def match_geometric_unique(desc1, desc2, pts1, pts2, H):
    """Match keypoints based purely on geometric uniqueness (no descriptor matching).
    
    Matching process:
    1. For each point in pts1, find all pts2 within reproj_threshold of H*pts1
    2. Accept match only if exactly one point in pts2 is within threshold (uniqueness forward)
    3. For each point in pts2, find all pts1 within reproj_threshold of H^-1*pts2
    4. Accept match only if exactly one point in pts1 is within threshold (uniqueness backward)
    5. Keep only bidirectionally unique matches
    
    Args:
        desc1: Descriptors from image 1 (N1 x 128) [unused, kept for API compatibility]
        desc2: Descriptors from image 2 (N2 x 128) [unused, kept for API compatibility]
        pts1: Keypoint coordinates from image 1 (N1 x 2)
        pts2: Keypoint coordinates from image 2 (N2 x 2)
        H: Homography matrix from image 1 to image 2
    
    Returns:
        matches_idx1: Indices of matched points in pts1
        matches_idx2: Indices of matched points in pts2
    """
    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pts1_t = torch.from_numpy(pts1).float().to(device)
    pts2_t = torch.from_numpy(pts2).float().to(device)
    H_t = torch.from_numpy(H).float().to(device)
    
    # Compute H inverse
    try:
        H_inv_t = torch.inverse(H_t)
    except:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # Forward: H * pts1 -> pts2_space
    pts1_h = torch.cat([pts1_t, torch.ones(pts1_t.shape[0], 1, device=device)], dim=1)
    warped_fwd_h = (H_t @ pts1_h.T).T
    warped_fwd = warped_fwd_h[:, :2] / warped_fwd_h[:, 2:3]  # N1 x 2
    
    # Backward: H^-1 * pts2 -> pts1_space
    pts2_h = torch.cat([pts2_t, torch.ones(pts2_t.shape[0], 1, device=device)], dim=1)
    warped_bwd_h = (H_inv_t @ pts2_h.T).T
    warped_bwd = warped_bwd_h[:, :2] / warped_bwd_h[:, 2:3]  # N2 x 2
    
    # Compute pairwise distances: warped_fwd (N1x2) to pts2 (N2x2)
    dists_fwd = torch.cdist(warped_fwd, pts2_t, p=2)  # N1 x N2
    
    # Compute pairwise distances: warped_bwd (N2x2) to pts1 (N1x2)
    dists_bwd = torch.cdist(warped_bwd, pts1_t, p=2)  # N2 x N1
    
    # For each pts1, find candidate pts2 within threshold
    # Count how many pts2 are within threshold for each pts1
    within_threshold_fwd = dists_fwd < UNIQ_DIST_THRESHOLD  # N1 x N2 (boolean)
    num_candidates_fwd = torch.sum(within_threshold_fwd, dim=1)  # N1
    
    # For each pts2, find candidate pts1 within threshold
    within_threshold_bwd = dists_bwd < UNIQ_DIST_THRESHOLD  # N2 x N1 (boolean)
    num_candidates_bwd = torch.sum(within_threshold_bwd, dim=1)  # N2
    
    # Find points with exactly one candidate (unique matches)
    unique_fwd_mask = num_candidates_fwd == 1  # N1 (boolean)
    unique_bwd_mask = num_candidates_bwd == 1  # N2 (boolean)
    
    # Get indices of unique forward matches
    unique_fwd_idx1 = torch.where(unique_fwd_mask)[0]  # indices in pts1
    
    valid_matches_idx1 = []
    valid_matches_idx2 = []
    
    for i1 in unique_fwd_idx1:
        i1_item = i1.item()
        # Find the unique match in pts2
        i2 = torch.where(within_threshold_fwd[i1])[0][0].item()

        if (dists_fwd[i1, i2] + dists_bwd[i2, i1]) /2 >= NN_DIST_THRESHOLD:
            continue
        
        # Verify backward uniqueness: pts2[i2] should also have unique match to pts1[i1]
        if not unique_bwd_mask[i2]:
            continue
        
        # Check that backward match points back to i1
        i1_back = torch.where(within_threshold_bwd[i2])[0][0].item()
        if i1_back != i1_item:
            continue
        
        valid_matches_idx1.append(i1_item)
        valid_matches_idx2.append(i2)
    
    return np.array(valid_matches_idx1, dtype=int), np.array(valid_matches_idx2, dtype=int)



def compute_residuals(root, recompute=False, summary=False):
    """Compute or load Sampson residuals from HPatches dataset.
    
    Returns:
        tuple: (all_sampson_sq, seq_summaries)
    """
    seq_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if len(seq_dirs) == 0:
        print("No sequences found under root:", root)
        return None, None

    out_npz = os.path.join(root, "hpatches_symtransfer_stats.npz")
    all_sampson_sq = None


    # Try to load cached results
    if os.path.isfile(out_npz) and not RECOMPUTE_RES:
        try:
            cache = np.load(out_npz)
            all_sampson_sq = np.array(cache["sampson_sq"]).copy()
            all_sampson_E_sq = np.array(cache.get("sampson_E_sq", [])).copy()
            all_descriptor_distances = np.array(cache.get("descriptor_distances", [])).copy()
            if VERBOSE:
                print(f"Loaded cached results from {out_npz}")
            return all_sampson_sq, all_sampson_E_sq, all_descriptor_distances, None
        except Exception as e:
            print("Failed to load cached results, recomputing:", e)
            all_sampson_sq = None

    # Compute if no cache
    # Filter sequences by type
    if not INCLUDE_I_SEQUENCES:
        seq_dirs = [s for s in seq_dirs if not s.startswith('i_')]
    if not INCLUDE_V_SEQUENCES:
        seq_dirs = [s for s in seq_dirs if not s.startswith('v_')]
    
    sampson_sq_list = []
    sampson_E_sq_list = []
    descriptor_distances_list = []
    processed_pairs = 0
    seq_summaries = [] if summary else None

    for seq in tqdm(seq_dirs, desc="Processing sequences"):
        seqp = os.path.join(root, seq)
        
        # Check for cached keypoints/descriptors
        cache_file = os.path.join(seqp, f"keypoints_cache_maxkp{MAX_KEYPOINTS}_desc{USE_DESCRIPTORS}.npz")
        
        if os.path.isfile(cache_file) and not RECOMPUTE:
            # Load cached keypoints and descriptors
            try:
                cache_data = np.load(cache_file, allow_pickle=True)
                keypoints = cache_data['keypoints'].item()
                descriptors = cache_data['descriptors'].item() if USE_DESCRIPTORS else None
                images = {}
                for idx in keypoints.keys():
                    img_path = get_image_path(seqp, f"{idx}")
                    if img_path is not None:
                        images[idx] = np.array(Image.open(img_path).convert("RGB"))
                # if VERBOSE and len(keypoints) > 0:
                    # print(f"Loaded cached keypoints for {seq}")
            except Exception as e:
                if VERBOSE:
                    print(f"Failed to load keypoint cache for {seq}: {e}, recomputing")
                cache_data = None
        else:
            cache_data = None
        
        if cache_data is None:
            # Load all available images (1 through 6) and compute keypoints
            images = {}
            keypoints = {}
            descriptors = {} if USE_DESCRIPTORS else None
            for idx in range(1, 7):
                img_path = get_image_path(seqp, f"{idx}")
                if img_path is None:
                    continue
                img = np.array(Image.open(img_path).convert("RGB"))
                pts, descs = detect_sift_keypoints(img, compute_descriptors=USE_DESCRIPTORS)
                if pts.shape[0] > 0:
                    images[idx] = img
                    keypoints[idx] = pts
                    if USE_DESCRIPTORS:
                        descriptors[idx] = descs
            
            # Save keypoints and descriptors to cache
            if len(keypoints) > 0:
                save_dict = {
                    'keypoints': keypoints,
                    'descriptors': descriptors if USE_DESCRIPTORS else {}
                }
                np.savez_compressed(cache_file, **save_dict)
                if VERBOSE:
                    print(f"Saved keypoint cache for {seq}")
        
        if len(images) < 2:
            continue
        
        # Generate all possible pairs
        available_indices = list(images.keys())
        all_possible_pairs = [(a, b) for i, a in enumerate(available_indices) 
                             for b in available_indices[i+1:]]
        
        # Create permutation and select first MAX_IMAGE_PAIRS (or all if fewer)
        perm = np.random.permutation(len(all_possible_pairs))
        if MAX_IMAGE_PAIRS is not None:
            num_pairs = min(MAX_IMAGE_PAIRS, len(all_possible_pairs))
            selected_indices = perm[:num_pairs]
        else:
            selected_indices = perm
        
        pairs = [all_possible_pairs[i] for i in selected_indices]
        
        if summary:
            total_matches = 0

        for idx_a, idx_b in pairs:
            Hpath = os.path.join(seqp, f"H_{idx_a}_{idx_b}")
            if not os.path.isfile(Hpath):
                continue

            pts1 = keypoints[idx_a]
            pts2 = keypoints[idx_b]
            desc1 = descriptors[idx_a] if USE_DESCRIPTORS else None
            desc2 = descriptors[idx_b] if USE_DESCRIPTORS else None
            
            if pts2.shape[0] == 0:
                continue

            H = load_homography(Hpath)
            try:
                Hinv = invert_homography(H)
            except np.linalg.LinAlgError:
                continue

            # Match using geometric uniqueness (no descriptors)
            matches_idx1, matches_idx2 = match_geometric_unique(desc1, desc2, pts1, pts2, H)
            
            if len(matches_idx1) == 0:
                continue
            
            # Filter matches by descriptor distance threshold
            if USE_DESCRIPTORS and desc1 is not None and desc2 is not None:
                matched_desc1 = desc1[matches_idx1]
                matched_desc2 = desc2[matches_idx2]
                # Compute L2 distances between matched descriptors
                desc_dists = np.linalg.norm(matched_desc1 - matched_desc2, axis=1)
                # Keep only matches with descriptor distance below threshold
                valid_desc_mask = desc_dists <= DESCRIPTOR_DISTANCE_THRESHOLD
                matches_idx1 = matches_idx1[valid_desc_mask]
                matches_idx2 = matches_idx2[valid_desc_mask]
                
                if len(matches_idx1) == 0:
                    continue
            
            # Get matched keypoints
            matched_pts1 = pts1[matches_idx1]
            matched_pts2 = pts2[matches_idx2]
            
            # Collect descriptor distances for matched pairs
            if USE_DESCRIPTORS and desc1 is not None and desc2 is not None:
                matched_desc1 = desc1[matches_idx1]
                matched_desc2 = desc2[matches_idx2]
                # Compute L2 distances between matched descriptors
                desc_dists = np.linalg.norm(matched_desc1 - matched_desc2, axis=1)
                descriptor_distances_list.extend(desc_dists.tolist())
            
            # Vectorized Sampson residuals using SampsonBM
            H_t = torch.from_numpy(H).double().unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
            x_h = np.hstack([matched_pts1, np.ones((matched_pts1.shape[0], 1))])
            y_h = np.hstack([matched_pts2, np.ones((matched_pts2.shape[0], 1))])
            x_t = torch.from_numpy(x_h).double().unsqueeze(0)  # [1,n,3]
            y_t = torch.from_numpy(y_h).double().unsqueeze(0)  # [1,n,3]
            sampson_res = SampsonBM(x_t, y_t, H_t)  # [1,1,n]
            sampson_res_np = sampson_res.squeeze().cpu().numpy()  # [n]
            sampson_sq = sampson_res_np ** 2
            # Ensure we have an iterable even for single values
            sampson_sq_list.extend(np.atleast_1d(sampson_sq).tolist())
            
            # Compute fundamental matrix from homography using synthetic points
            try:
                # Generate synthetic point correspondences using the homography
                # Use 8 points in a grid pattern to avoid degeneracies
                img_h, img_w = images[idx_a].shape[:2]
                
                # Create grid of points covering the image
                grid_x = np.linspace(img_w * 0.1, img_w * 0.9, 3)
                grid_y = np.linspace(img_h * 0.1, img_h * 0.9, 3)
                synth_pts1 = np.array([[x, y] for y in grid_y for x in grid_x], dtype=np.float64)
                
                # Warp through homography to get corresponding points
                synth_pts1_h = np.hstack([synth_pts1, np.ones((synth_pts1.shape[0], 1))])
                synth_pts2_h = (H @ synth_pts1_h.T).T
                synth_pts2 = synth_pts2_h[:, :2] / synth_pts2_h[:, 2:3]
                
                # Compute fundamental matrix using 8-point algorithm
                # Normalize coordinates for numerical stability
                mean1 = np.mean(synth_pts1, axis=0)
                mean2 = np.mean(synth_pts2, axis=0)
                scale1 = np.sqrt(2) / np.mean(np.linalg.norm(synth_pts1 - mean1, axis=1))
                scale2 = np.sqrt(2) / np.mean(np.linalg.norm(synth_pts2 - mean2, axis=1))
                
                T1 = np.array([[scale1, 0, -scale1*mean1[0]],
                              [0, scale1, -scale1*mean1[1]],
                              [0, 0, 1]], dtype=np.float64)
                T2 = np.array([[scale2, 0, -scale2*mean2[0]],
                              [0, scale2, -scale2*mean2[1]],
                              [0, 0, 1]], dtype=np.float64)
                
                pts1_norm = (T1 @ synth_pts1_h.T).T
                pts2_norm = (T2 @ np.hstack([synth_pts2, np.ones((synth_pts2.shape[0], 1))]).T).T
                
                # Build constraint matrix for F
                A = np.zeros((synth_pts1.shape[0], 9), dtype=np.float64)
                for i in range(synth_pts1.shape[0]):
                    x1, y1, _ = pts1_norm[i]
                    x2, y2, _ = pts2_norm[i]
                    A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
                
                # Solve for F using SVD
                _, _, Vt = np.linalg.svd(A)
                F_norm = Vt[-1].reshape(3, 3)
                
                # Enforce rank-2 constraint
                U, S, Vt = np.linalg.svd(F_norm)
                S[2] = 0
                F_norm = U @ np.diag(S) @ Vt
                
                # Denormalize
                F = T2.T @ F_norm @ T1
                
                # Compute Sampson_E residuals using F (same as essential matrix residuals)
                F_t = torch.from_numpy(F).double().unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
                
                sampson_E_res = SampsonBM_E(y_t, x_t, F_t)  # [1,1,n]
                sampson_E_res_np = sampson_E_res.squeeze().cpu().numpy()  # [n]
                sampson_E_sq = sampson_E_res_np ** 2
                sampson_E_sq_list.extend(np.atleast_1d(sampson_E_sq).tolist())
            except Exception as e:
                # If F computation fails, skip E residuals for this pair
                pass
            
            processed_pairs += 1
            
            if summary:
                total_matches += len(matches_idx1)

        if summary:
            # Use first image from the sequence for size info
            first_img_idx = min(images.keys())
            h, w = images[first_img_idx].shape[:2]
            avg_kpts = float(np.mean([kp.shape[0] for kp in keypoints.values()]))
            seq_summaries.append({
                "seq": seq,
                "size": (int(h), int(w)),
                "kpts_avg": avg_kpts,
                "matches": total_matches,
            })

    if len(sampson_sq_list) == 0:
        print("No matches found. Try increasing MAX_KEYPOINTS or NN_DIST_THRESHOLD.")
        return None, None, None, None

    all_sampson_sq = np.array(sampson_sq_list)
    all_sampson_E_sq = np.array(sampson_E_sq_list) if len(sampson_E_sq_list) > 0 else np.array([])
    all_descriptor_distances = np.array(descriptor_distances_list) if len(descriptor_distances_list) > 0 else np.array([])

    # Save cache
    np.savez_compressed(out_npz, sampson_sq=all_sampson_sq, sampson_E_sq=all_sampson_E_sq, descriptor_distances=all_descriptor_distances)
    if VERBOSE:
        print("Saved results to", out_npz)
    
    return all_sampson_sq, all_sampson_E_sq, all_descriptor_distances, seq_summaries

def plot_descriptor_distances(descriptor_distances, dataset_name=''):
    """Plot histogram of SIFT descriptor distances for matched inliers."""
    if descriptor_distances.size == 0:
        print("WARNING: No descriptor distances to plot.")
        return
    
    plt.figure(figsize=(8, 4))
    plt.hist(descriptor_distances, bins=100, range=(0, 1000), density=True, alpha=0.7, label='Descriptor L2 distances')
    
    title_prefix = f"{dataset_name} - " if dataset_name else ""
    plt.title(f"{title_prefix}Histogram of SIFT Descriptor Distances (Matched Inliers)")
    plt.xlabel("L2 Distance")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("Descriptor distance stats:", {
        "count": int(descriptor_distances.size),
        "min": float(np.min(descriptor_distances)),
        "mean": float(np.mean(descriptor_distances)),
        "median": float(np.median(descriptor_distances)),
        "max": float(np.max(descriptor_distances)),
        "std": float(np.std(descriptor_distances)),
        "p90": float(np.percentile(descriptor_distances, 90)),
        "p99": float(np.percentile(descriptor_distances, 99)),
    })

def plot_sampson_histogram(sampson_residuals, kind='H', hist_range=HIST_RANGE, hist_bins=HIST_BINS, n_components=NUM_MIXTURE_COMPONENTS, dataset_name=''):
    """Plot histogram of Sampson residuals with mixture of chi distributions.
    
    Args:
        sampson_residuals: Array of Sampson residual values
        kind: 'H' for homography (chi(2)) or 'F' for fundamental/essential matrix (chi(1))
        hist_range: Range for histogram plot
        hist_bins: Number of histogram bins
        n_components: Number of chi components in mixture
        dataset_name: Name of dataset for filename prefix
    """
    if sampson_residuals.size == 0:
        print("WARNING: No residuals to plot.")
        return
    
    # Set parameters based on kind
    if kind == 'H':
        df = 2  # degrees of freedom for homography
        initial_scale_min = 0.25
        initial_scale_max = 1.0
        scale_bound_min = 0.01
        scale_bound_max = 10.0
        title = "Homography Residuals"
    elif kind == 'F':
        df = 1  # degrees of freedom for fundamental/essential matrix
        initial_scale_min = 0.15
        initial_scale_max = 0.8
        scale_bound_min = 0.01
        scale_bound_max = 5.0
        title = "Epipolar Residuals"
    else:
        raise ValueError(f"Unknown kind: {kind}. Must be 'H' or 'F'.")

    label = 'Histogram of inlier residuals'

    fig = plt.figure(figsize=(6, 3.5))
    counts, bins, _ = plt.hist(sampson_residuals, bins=hist_bins, range=hist_range, 
                                density=True, alpha=0.7, label=label, color='gray')
    
    # Fit mixture of n_components chi distributions with different scales
    if has_scipy_stats and sampson_residuals.size > 0:
        from scipy.optimize import minimize
        
        def softmax(logits):
            """Compute softmax to get positive weights that sum to 1."""
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            return exp_logits / np.sum(exp_logits)
        
        def negative_log_likelihood(params):
            """Negative log-likelihood for mixture model."""
            # First n_components params are logits, next n_components are scales
            logits = params[:n_components]
            scales = params[n_components:]
            
            # Map unconstrained logits to weights via softmax
            weights = softmax(logits)
            
            # Keep scales positive and reasonable
            scales = np.clip(scales, scale_bound_min, scale_bound_max)
            
            # Compute mixture PDF
            mixture_pdf = np.zeros_like(sampson_residuals)
            for i in range(n_components):
                chi_pdf = chi.pdf(sampson_residuals, df, loc=0, scale=scales[i])
                mixture_pdf += weights[i] * chi_pdf
            
            mixture_pdf = np.clip(mixture_pdf, 1e-10, None)  # Avoid log(0)
            
            return -np.sum(np.log(mixture_pdf))
        
        # Initial guess: uniform logits -> equal weights, exponentially spaced scales
        initial_logits = np.zeros(n_components)
        initial_scales = np.logspace(np.log10(initial_scale_min), np.log10(initial_scale_max), n_components)
        initial_params = np.concatenate([initial_logits, initial_scales])
        
        # Bounds: logits unconstrained, scales positive
        if kind == 'H':
            # For homography, fix scales near initial for stability
            bounds = [(-10, 10)] * n_components + [(s*0.9, s*1.5) for s in initial_scales]
        else:
            # For fundamental matrix, allow wider scale range
            bounds = [(-10, 10)] * n_components + [(scale_bound_min, scale_bound_max)] * n_components
        
        result = minimize(negative_log_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            logits_fit = result.x[:n_components]
            scales_fit = result.x[n_components:]
            
            # Convert logits to weights via softmax
            weights_fit = softmax(logits_fit)
            
            print(f"Fitted {n_components}-component Chi({df}) mixture ({kind}-residuals):")
            for i in range(n_components):
                print(f"  Component {i+1}: weight={weights_fit[i]:.4f}, Chi({df}, σ={scales_fit[i]:.4f})")
            
            # Generate fitted curves
            x = np.linspace(hist_range[0], hist_range[1], 1000)
            mixture_pdf = np.zeros_like(x)
            
            colors = ['g', 'b', 'm', 'c', 'y', 'orange', 'purple', 'brown']
            
            # Plot individual components
            for i in range(n_components):
                chi_pdf = chi.pdf(x, df, loc=0, scale=scales_fit[i])
                mixture_pdf += weights_fit[i] * chi_pdf
                color = colors[i % len(colors)]
                # plt.plot(x, weights_fit[i] * chi_pdf, '--', linewidth=1, alpha=0.5, color=color,
                #         label=f'C{i+1}: {weights_fit[i]:.2f}×Chi({df}, σ={scales_fit[i]:.2f})')
                plt.plot(x, weights_fit[i] * chi_pdf, '--', linewidth=1, alpha=0.5, color=color,
                        label=None)
            
            # Plot mixture
            plt.plot(x, mixture_pdf, 'r-', linewidth=2, label=f'Mixture of $\chi_{{{df}}}$ distributions')
        else:
            print("Mixture fitting failed:", result.message)
    
    plt.xlim(left=0, right = 3.0)
    title_prefix = f"{dataset_name} - " if dataset_name else ""
    plt.title(f"{title_prefix}{title}")
    plt.xlabel("Sampson Error [px]")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    filename_prefix = f"{dataset_name}_" if dataset_name else ""
    outf = f'fig/{filename_prefix}dist_{kind}.pdf'
    savefig(outf)
    plt.show()
    plt.close(fig)
    plt.show()

def process_phototourism_data(descriptor_hist_file):
    """Load and process PhotoTourism descriptor distance histogram."""
    import pickle
    
    if not os.path.isfile(descriptor_hist_file):
        print(f"File not found: {descriptor_hist_file}")
        return False
    
    print(f"Loading descriptor distances from {descriptor_hist_file}")
    with open(descriptor_hist_file, 'rb') as f:
        data = pickle.load(f)
    
    # Check if data is histogram format (hist, bin_edges)
    if isinstance(data, dict) and 'hist' in data and 'bin_edges' in data:
        hist = np.array(data['hist'])
        bin_edges = np.array(data['bin_edges'])
        
        # Reconstruct samples deterministically from normalized histogram
        # Create int(hist[i] * 10000) copies of each bin center
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Determine number of samples per bin proportional to density
        target_total = 10000
        samples_per_bin = (hist * target_total).astype(int)
        
        # Create samples by repeating each bin center
        descriptor_distances = np.repeat(bin_centers, samples_per_bin)
        
        print(f"Reconstructed {len(descriptor_distances)} samples from normalized histogram")
        print(f"Histogram has {len(hist)} bins, range [{bin_edges[0]:.3f}, {bin_edges[-1]:.3f}]")
        print("Descriptor distance stats:", {
            "min": float(np.min(descriptor_distances)),
            "mean": float(np.mean(descriptor_distances)),
            "median": float(np.median(descriptor_distances)),
            "max": float(np.max(descriptor_distances)),
            "std": float(np.std(descriptor_distances)),
            "p90": float(np.percentile(descriptor_distances, 90)),
            "p99": float(np.percentile(descriptor_distances, 99)),
        })
        
        # Use exact same range and number of bins as the original histogram
        hist_range = (float(bin_edges[0]), float(bin_edges[-1]))
        hist_bins = len(hist)
        
        # Plot as epipolar residuals using our plotting function
        plot_sampson_histogram(descriptor_distances, kind='F', hist_range=hist_range, hist_bins=hist_bins, n_components=3, dataset_name='PhotoTourism')
    else:
        # Try to extract raw data
        if isinstance(data, dict):
            descriptor_distances = data.get('descriptor_distances') or data.get('residuals') or data.get('sampson_residuals')
        else:
            descriptor_distances = data
        
        if descriptor_distances is not None and len(descriptor_distances) > 0:
            descriptor_distances = np.array(descriptor_distances)
            print(f"Loaded {len(descriptor_distances)} descriptor distance values")
            print("Descriptor distance stats:", {
                "min": float(np.min(descriptor_distances)),
                "mean": float(np.mean(descriptor_distances)),
                "median": float(np.median(descriptor_distances)),
                "max": float(np.max(descriptor_distances)),
                "std": float(np.std(descriptor_distances)),
                "p90": float(np.percentile(descriptor_distances, 90)),
                "p99": float(np.percentile(descriptor_distances, 99)),
            })
            
            # Plot as epipolar residuals using our plotting function
            plot_sampson_histogram(descriptor_distances, kind='F', hist_range=(0.0, 3.0), hist_bins=100, n_components=3, dataset_name='PhotoTourism')
        else:
            print("ERROR: Could not extract descriptor distances from loaded data")
            print("Data structure:", type(data))
            if isinstance(data, dict):
                print("Available keys:", data.keys())
            return False
    
    return True

def main():
    # Process PhotoTourism data
    descriptor_hist_file = "/mnt/datagrid/personal/shekhovt/datagrid/data/PhotoTourism/sampson_error_histogram.pkl"
    process_phototourism_data(descriptor_hist_file)
    
    # Process HPatches dataset
    print("\n" + "="*80)
    print("Processing HPatches dataset...")
    print("="*80 + "\n")
    
    # Resolve dataset root
    root = input_root()
    if not os.path.isdir(root):
        print("ERROR: Provided root does not exist or is not a directory:", root)
        return

    # Compute or load residuals
    all_sampson_sq, all_sampson_E_sq, all_descriptor_distances, seq_summaries = compute_residuals(
        root, recompute=RECOMPUTE_RES, summary=SUMMARY
    )
    
    if all_sampson_sq is None or all_sampson_sq.size == 0:
        print("No matches found. Exiting.")
        return

    # ---- Analysis from cached or computed arrays ----
    total_count = int(all_sampson_sq.size)
    print("Total matched correspondences:", total_count)

    sampson_residuals = np.sqrt(np.maximum(all_sampson_sq, 0.0))
    print("Sampson residuals summary:", {
        "min": float(np.min(sampson_residuals)) if sampson_residuals.size > 0 else None,
        "median": float(np.median(sampson_residuals)) if sampson_residuals.size > 0 else None,
        "max": float(np.max(sampson_residuals)) if sampson_residuals.size > 0 else None,
    })

    # Plot histogram with chi distribution fit
    plot_sampson_histogram(sampson_residuals, kind='H', dataset_name='HPatches')
    
    # Plot Sampson_E residuals if available
    print(all_sampson_E_sq)
    if all_sampson_E_sq is not None and all_sampson_E_sq.size > 0:
        sampson_E_residuals = np.sqrt(np.maximum(all_sampson_E_sq, 0.0))
        print("\nSampson_E residuals (from fundamental matrix):")
        print("Total correspondences with valid E:", int(all_sampson_E_sq.size))
        print("Sampson_E residuals summary:", {
            "min": float(np.min(sampson_E_residuals)),
            "median": float(np.median(sampson_E_residuals)),
            "max": float(np.max(sampson_E_residuals)),
        })
        plot_sampson_histogram(sampson_E_residuals, kind='F', dataset_name='HPatches')
    else:
        print("\nNo Sampson_E residuals available (fundamental matrix computation may have failed)")
    
    # Plot descriptor distances histogram
    if all_descriptor_distances is not None and all_descriptor_distances.size > 0:
        plot_descriptor_distances(all_descriptor_distances, dataset_name='HPatches')
    else:
        print("No descriptor distances available (USE_DESCRIPTORS may be False)")

    def stats(arr):
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }

    print("Sampson residual stats (px):", stats(sampson_residuals))
    print("Algebraic Sampson squared stats:", stats(all_sampson_sq))

    # Optional per-sequence summary (only available when recomputing)
    if seq_summaries is not None:
        print("Per-sequence summary (image size, avg keypoints, inlier matches):")
        for s in seq_summaries:
            sz = f"{s['size'][0]}x{s['size'][1]}"
            print(f"{s['seq']}: {sz}, kpts~={s['kpts_avg']:.0f}, matches={s['matches']}")

if __name__ == "__main__" or True:
    main()
# %%
