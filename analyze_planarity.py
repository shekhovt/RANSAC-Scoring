"""
Analyze how well the inlier correspondences fit a planar scene assumption.

For a perfect planar scene, all inliers should have very low homography Sampson error.
If we see high errors, it means either:
1. The scene is not planar (many 3D structures at different depths)
2. The inliers are misclassified
3. Our homography reconstruction is wrong

We already verified (2) is not the issue (Essential matrix errors are similar).
We verified (3) is not the issue (algorithm works on synthetic data).
So let's check (1) by analyzing the 3D structure of the inliers.
"""

import torch
import numpy as np
import cv2
from load_data import H_dataset, HEB
import torch.utils.data as data

def analyze_scene_planarity():
    dataset = H_dataset(HEB, 'Piazza_del_Popolo', padding=False)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    pair_count = 0
    max_pairs = 10
    
    for data_batch in loader:
        C = data_batch['correspondences']
        pts1 = C[0, :, 0:3]
        pts2 = C[0, :, 3:6]
        inliers = data_batch['inliers'][0]
        gt_R = data_batch['gt_R'][0]
        gt_t = data_batch['gt_t'][0]
        K1 = data_batch['K1'][0].float()
        K2 = data_batch['K2'][0].float()
        
        n_inliers = inliers.sum().item()
        if n_inliers < 50:
            continue
        
        pair_count += 1
        if pair_count > max_pairs:
            break
        
        # Triangulate to get 3D points
        P1 = np.eye(3, 4)
        P2 = np.hstack([gt_R.cpu().numpy(), gt_t.cpu().numpy().reshape(3, 1)])
        
        pts1_in = pts1[inliers].cpu().numpy()[:, :2]
        pts2_in = pts2[inliers].cpu().numpy()[:, :2]
        
        pts_4d = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)
        pts_3d = (pts_4d[:3, :] / pts_4d[3, :]).T  # [N, 3]
        
        # Fit a plane to the 3D points using SVD
        centroid = pts_3d.mean(axis=0)
        centered = pts_3d - centroid
        
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1, :]  # Last row is the normal (smallest singular value)
        
        # Compute distance of each point from the plane
        distances = np.abs(centered @ normal)
        
        # Compute statistics
        mean_dist = distances.mean()
        median_dist = np.median(distances)
        max_dist = distances.max()
        std_dist = distances.std()
        
        # Compute depth range
        depths = pts_3d[:, 2]  # Z coordinate in camera 1 frame
        depth_range = depths.max() - depths.min()
        depth_mean = depths.mean()
        depth_ratio = depth_range / depth_mean
        
        print(f"\n=== Pair {pair_count}: {data_batch['files'][0]} ===")
        print(f"Inliers: {n_inliers}")
        print(f"\n3D Structure Analysis:")
        print(f"  Depth statistics:")
        print(f"    Mean depth: {depth_mean:.2f}")
        print(f"    Depth range: {depth_range:.2f}")
        print(f"    Relative depth variation: {depth_ratio:.2%}")
        print(f"\n  Distance from best-fit plane:")
        print(f"    Mean: {mean_dist:.4f}")
        print(f"    Median: {median_dist:.4f}")
        print(f"    Std: {std_dist:.4f}")
        print(f"    Max: {max_dist:.4f}")
        print(f"    90th percentile: {np.percentile(distances, 90):.4f}")
        
        # Rule of thumb: if depth variation > 20% or median distance > 0.1, scene is not planar
        if depth_ratio > 0.2:
            print(f"  ⚠️ HIGH depth variation ({depth_ratio:.1%})! Scene likely NOT planar.")
        if median_dist > 0.1:
            print(f"  ⚠️ HIGH distance from plane! Scene likely NOT planar.")
        
        # Estimate expected reprojection error from non-planarity
        # Using simplified formula: error ≈ (distance_from_plane / depth) * focal_length
        fx = K1[0, 0].item()
        expected_error_px = (median_dist / depth_mean) * fx
        print(f"\n  Expected pixel error from non-planarity: {expected_error_px:.2f}px")

if __name__ == '__main__':
    analyze_scene_planarity()
