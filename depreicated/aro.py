import cv2
import numpy as np
import torch
import math
import kornia
from torch import Tensor

from ..functional import *
from ..score_weights import compute_residuals, sufficient_statistic


try:
    from pymagsac import optimizeEssentialMatrix
    pymagsac_available = 1

    def numerical_optimization(matches, K1, K2, inlier_indices, best_model, unnormalzied_threshold=0.5, best_score=0):
        # bundle adjustment
        estimated_models, _ = optimizeEssentialMatrix(
            matches,
            K1,
            K2,
            inlier_indices,
            best_model,
            unnormalzied_threshold,
            float(best_score)
        )
        # normalize the models
        # torch.from_numpy(estimated_models).to(matches.device, matches.dtype).unsqueeze(0)/torch.norm(torch.from_numpy(estimated_models).to(matches.device, matches.dtype).unsqueeze(0), dim=(1,2))
        estimated_models = estimated_models/np.linalg.norm(estimated_models)
        return estimated_models

except ImportError:
    pymagsac_available = 0


def ARO_candidates(all_points, residuals, polish_threshold=10., groups=20, K1=None, K2=None, max_iters=1000):

    """
        all_points [N, 4] -- correspondences
        residuals [N] -- residuals of the initial model, in px
        polish threshold -- largest threshold on residuals to ocnsider, in px
        gropus -- how many nested sets   
        
        return: list of candidate models [[3,3] x groups]
    """
    
    idx = np.argsort(residuals)
    residuals = residuals[idx] # sort points by increasing residuals
    all_points = all_points[idx,:]
    
    # select only residuals within the polish threshold
    mask = residuals < polish_threshold
    all_inliers = all_points[mask,:]
    residuals = residuals[mask]
    N = all_inliers.shape[0]

    # Normalize the threshold
    # polish_threshold = 0.75
    # avgDiagonal = (K1[0][0] + K1[1][1] + K2[0][0] + K2[1][1]) / 4 if ((len(K1)==3) and (len(K2)==3)) else 1
    # normalizedThreshold = polish_threshold / avgDiagonal
            
    # LMEDs
    Es  = []
    maskss = []
    for g in range(groups):
        i = int(math.ceil(N / (groups-1) * g))
        if i < 8:
            continue
        try: 
            I_points = all_inliers[:i]
            # return the masks in the increasing residual order
            E, masks = cv2.findEssentialMat(
                I_points[:, :2], # normalized points
   	            I_points[:, 2:],
	            np.eye(3, 3), #K
	            cv2.LMEDS#RANSAC, # method
                # cv2.RANSAC,  # method
	            # prob = 0.99, # confidence
   	            # threshold = normalizedThreshold, # only relevant for RANSAC (threshold appropriate for normalized coordinates)
                # maxIters = max_iters,  # max iterstaions (no effect on the running time of RANSAC?)
	        )
            assert(E.shape[0] == 3)
            Es += [E]

            maskss += [idx[:I_points.shape[0]][masks[:, 0] ==1]]
        except:
            import pdb; pdb.set_trace()
        
    return Es, maskss

def all_combinations_error(R1, R2, t1, gt_R, gt_t):
    
    combine = [(R1, t1), (R2, t1), (R1, -t1), (R2, -t1)]
    err_Rs = []
    err_Ts = []
    for (R, t) in combine:
        err_R, err_t = pose_error(R, gt_R, t, gt_t)
        err_Rs += [err_R]
        err_Ts += [err_t]
    
    return min(err_Rs), min(err_Ts)

def pose_error(R, gt_R, t, gt_t):
    """Compute the angular error between two rotation matrices and two translation vectors.


	Keyword arguments:
	R -- 2D numpy array containing an estimated rotation
	gt_R -- 2D numpy array containing the corresponding ground truth rotation
	t -- 2D numpy array containing an estimated translation as column
	gt_t -- 2D numpy array containing the corresponding ground truth translation
	"""

    # calculate angle between provided rotations
    dR = np.matmul(R, np.transpose(gt_R))
    dR = cv2.Rodrigues(dR)[0]
    dR = np.linalg.norm(dR) * 180 / math.pi

    # calculate angle between provided translations
    dT = float(np.dot(gt_t.T, t))
    dT /= float(np.linalg.norm(gt_t))

    if dT > 1 or dT < -1:
        print("Domain warning! dT:", dT)
        dT = max(-1, min(1, dT))
    dT = math.acos(dT) * 180 / math.pi

    return dR, dT


def pose_error_EE(EE, gt_E, pts, combine=False):
    pts1 = pts[:, 0:2].astype(np.float64)
    pts2 = pts[:, 2:4].astype(np.float64)
    EE = EE.astype(np.float64)
    gt_E = gt_E.astype(np.float64)
    _, gt_R, gt_t, _ = cv2.recoverPose(np.ascontiguousarray(gt_E), pts1, pts2)
    errs = []
    for E in EE:
        # changed to choose the correct R/t by comparing with GT.
        if combine: 
            R1, R2, t1 = kornia.geometry.epipolar.decompose_essential_matrix(torch.from_numpy(E))
            err_R, err_t = all_combinations_error(R1, R2, t1, gt_R, gt_t) 
        else:
            _, R, t, _ = cv2.recoverPose(np.ascontiguousarray(E), pts1, pts2)
            err_R, err_t = pose_error(R, gt_R, t, gt_t)
        err = max(err_R, err_t)
        errs += [err]
    return np.stack(errs).astype(np.float32)

def pose_error_EERT(EE, gt_R, gt_t, pts, combine=False):
    """
    input:
    EE [M, 3,3] -- models
    output:
    err, err_rs, err_ts [M]
    """
    pts1 = pts[:, 0:2].astype(np.float64)
    pts2 = pts[:, 2:4].astype(np.float64)
    EE = EE.astype(np.float64)
    errs = []
    err_rs = []
    err_ts = []
    for E in EE:
        # changed to choose the correct R/t by comparing with GT.
        if combine: 
            try: 
                R1, R2, t1 = kornia.geometry.epipolar.decompose_essential_matrix(torch.from_numpy(E))
            except:
                import pdb; pdb.set_trace()
            err_R, err_t = all_combinations_error(R1[0].numpy(), R2[0].numpy(), t1.numpy(), gt_R, gt_t)
        else:
            _, R, t, _ = cv2.recoverPose(np.ascontiguousarray(E), pts1, pts2)
            err_R, err_t = pose_error(R, gt_R, t, gt_t)
        err = max(err_R, err_t)
        errs += [err]
        err_rs += [err_R]
        err_ts += [err_t]

    return np.stack(errs).astype(np.float32), np.stack(err_rs).astype(np.float32), np.stack(err_ts).astype(np.float32)
# # all correspondences
# filtered_correspondences = np.random.rand(500, 4)
# all_correspondences = np.random.rand(888, 4)

# # best model from N iterations of RANSAC
# best_minimal_model = np.random.rand(3, 3)

# EE = ARO(filtered_correspondences, all_correspondences, None, best_minimal_model)

def pose_error_batch(models, data):
    best_e = []
    best_r = []
    best_t = []
    gt_R = data['gt_R'].numpy()
    gt_t = data['gt_t'].numpy()
    C = data['correspondences'].numpy()
    for b in range(models.shape[0]):
        # todo simplify
        best_e_b, best_r_b, best_t_b = pose_error_EERT(models[b][None, :], gt_R[b], gt_t[b], C[b], combine=True)  # [M]
        best_e += [best_e_b]
        best_r += [best_r_b]
        best_t += [best_t_b]
    best_e = np.stack(best_e)[:, 0]
    best_r = np.stack(best_r)[:, 0]
    best_t = np.stack(best_t)[:, 0]
    return best_e, best_r, best_t

def best_ARO(data, best_idx, M, W):
    """
    data - dict with batched data:
        correspondances [B, N, 4]
        models [B, M, 3, 3]
        errors [B, M]
        num_pts [B] number of valid (not inf) correspondences
        residuals [B, M, N]
    best_idx [B] -- selected models
    M  [N_bins] -- weight matrix of the scoring function
    W - scoring function class
    
    Return:
    best_ee1 [B] -- errors of selected models
    best_ss1 [B] -- scores of selected models
    models [B] -- selected models
    """
    B = best_idx.shape[0]
    best_ss1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(-float('inf'))
    best_ee1 = best_idx.new_empty([B], dtype=torch.float32).fill_(float('inf'))
    best_eer1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(float('inf'))
    best_eet1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(float('inf'))
    best_models = best_idx.new_empty([B, 3, 3])
    num_pts = data['num_pts']
    """Compute sequentially for one image pair at a time"""
    start = time.time()
    dt1 = 0
    dt2 = 0

    # unnormalized_points(data)
    for b in range(best_idx.shape[0]):
        best = best_idx[b].item()
        N = num_pts[b]
        C = data['correspondences'][b][:N, :].cpu().numpy()
        R0 = data['residuals'][b, best, :N].cpu().numpy()
        t1 = time.time()
        EE, _ = ARO_candidates(C, R0, K1=data['K1'][b].cpu().numpy(), K2=data['K2'][b].cpu(
        ).numpy())  # pass residuals of current selected model for building nested sets
        dt1 += (time.time() - t1)
        # EE += [data['models'][b,best]] # add the current best model to the pool, so that we are guaranteed not to worsen the score TODO: reimplemnt explicitly
        if len(EE) == 0:
            # ARO returned no new candidate mdoels (too few points survived threshold)
            continue
        EE = np.stack(EE).astype(np.float32)  # [m 3 3]
        # form the new data struct for evaluation of ARO models
        data1 = dict()
        for k in set(data.keys()) - set(['residuals', 'errorrs', 'models']):
            data1[k] = data[k][b:b+1]
        data1['correspondences'] = data1['correspondences'][:, :N]
        data1['models'] = torch.tensor(EE).unsqueeze(0).float()  # [1 M 3 3]
        compute_residuals(data1)
        # compute errors to GT R, t
        gt_E = data['models'][b, -1].cpu().numpy()
        gt_R = data['gt_R'][b].cpu().numpy()
        gt_t = data['gt_t'][b].cpu().numpy()
        t2 = time.time()
        # changed to evaluate on the GT R, t
        errs1, err_rs1, err_ts1 = torch.tensor(
            pose_error_EERT(EE, gt_R, gt_t, C, True))  # [M]
        # errs1 = torch.tensor(pose_error_EE(EE, gt_E, C)) # [M]
        dt2 += (time.time() - t2)
        # chose the best of ARO models
        ss = sufficient_statistic(
            data1['residuals'], W.N_bins, max_distance=W.max_distance, pow=W.pow)  # [B M K]
        best_s1, best_idx1 = best_score(ss, M)
        best_ss1[b] = best_s1
        best_ee1[b] = errs1[best_idx1.item()]
        best_eer1[b] = err_rs1[best_idx1.item()]
        best_eet1[b] = err_ts1[best_idx1.item()]
        best_models[b] = data1['models'][:, best_idx1.item()]
    # best_ss1 = torch.hstack(best_ss1)
    # best_ee1 = torch.hstack(best_ee1)
    # best_models = torch.vstack(best_models)
    end = time.time()
    # print(f'ARO total:  {end - start} (ARO_candidates: {dt1}, pose error: {dt2})')
    # [B], [B], [B 3 3]
    return best_ee1, best_eer1, best_eet1, best_ss1, best_models


def best_BA(data, best_idx, M, W, threshold=1.5, masks=None):
    """
    data - dict with batched data:
        correspondances [B, N, 4]
        models [B, M, 3, 3]
        errors [B, M]
        num_pts [B] number of valid (not inf) correspondences
        residuals [B, M, N]
    best_idx [B] -- selected models
    M  [N_bins] -- weight matrix of the scoring function
    W - scoring function class
    
    Return:
    best_ee1 [B] -- errors of selected models
    best_ss1 [B] -- scores of selected models
    models [B] -- selected models
    """
    B = best_idx.shape[0]
    best_ss1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(-float('inf'))
    best_ee1 = best_idx.new_empty([B], dtype=torch.float32).fill_(float('inf'))
    best_eer1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(float('inf'))
    best_eet1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(float('inf'))
    best_models = best_idx.new_empty([B, 3, 3])
    num_pts = data['num_pts']
    """Compute sequentially for one image pair at a time"""
    start = time.time()
    dt1 = 0
    dt2 = 0

    # unnormalized_points(data)

    for b in range(best_idx.shape[0]):
        best = best_idx[b].item()
        N = num_pts[b]
        C = data['correspondences'][b][:N, :].cpu().numpy()
        R0 = data['residuals'][b, best, :N].cpu().numpy()
        t1 = time.time()
        # np.arange(R0.shape[0])[masks]
        inliers = np.arange(R0.shape[0])[
            R0 < threshold] if masks is None else masks
        EE = numerical_optimization(
            C,  # [masks.T][0],
            data['K1'][b].cpu().numpy(),
            data['K2'][b].cpu().numpy(),
            inliers,
            data['models'][b][best].cpu().numpy(),
            threshold,
            0)
        # EE = ARO_candidates(C, R0, K1=data['K1'][b].cpu().numpy(), K2=data['K2'][b].cpu().numpy()) # pass residuals of current selected model for building nested sets
        dt1 += (time.time() - t1)
        # EE += [data['models'][b,best]] # add the current best model to the pool, so that we are guaranteed not to worsen the score TODO: reimplemnt explicitly
        if len(EE) == 0:
            # ARO returned no new candidate mdoels (too few points survived threshold)
            continue
        EE = np.stack(EE).astype(np.float32)[None, :]  # [m 3 3]
        # form the new data struct for evaluation of ARO models
        data1 = dict()
        for k in set(data.keys()) - set(['residuals', 'errorrs', 'models']):
            data1[k] = data[k][b:b+1]
        data1['correspondences'] = data1['correspondences'][:, :N]
        data1['models'] = torch.tensor(EE).unsqueeze(0).float()  # [1 M 3 3]
        # import pdb; pdb.set_trace()
        compute_residuals(data1)
        # compute errors to GT R, t
        gt_E = data['models'][b, -1].cpu().numpy()
        gt_R = data['gt_R'][b].cpu().numpy()
        gt_t = data['gt_t'][b].cpu().numpy()
        t2 = time.time()
        # changed to evaluate on the GT R, t
        errs1, errs_r1, errs_t1 = torch.tensor(pose_error_EERT(EE, gt_R, gt_t, C, True
                                                               ))  # [M]
        # errs1 = torch.tensor(pose_error_EE(EE, gt_E, C)) # [M]
        dt2 += (time.time() - t2)
        # chose the best of ARO models
        ss = sufficient_statistic(
            data1['residuals'], W.N_bins, max_distance=W.max_distance, pow=W.pow)  # [B M K]
        best_s1, best_idx1 = best_score(ss, M)
        best_ss1[b] = best_s1
        best_ee1[b] = errs1[best_idx1.item()]
        best_eer1[b] = errs_r1[best_idx1.item()]
        best_eet1[b] = errs_t1[best_idx1.item()]
        best_models[b] = data1['models'][:, best_idx1.item()]
    # best_ss1 = torch.hstack(best_ss1)
    # best_ee1 = torch.hstack(best_ee1)
    # best_models = torch.vstack(best_models)
    end = time.time()
    # print(f'ARO total:  {end - start} (ARO_candidates: {dt1}, pose error: {dt2})')
    # [B], [B], [B 3 3]
    return best_ee1, best_eer1, best_eet1, best_ss1, best_models


def best_LMEDS(data, best_idx, M, W, threshold=1.5, masks=None):
    """
    data - dict with batched data:
        correspondances [B, N, 4]
        models [B, M, 3, 3]
        errors [B, M]
        num_pts [B] number of valid (not inf) correspondences
        residuals [B, M, N]
    best_idx [B] -- selected models
    M  [N_bins] -- weight matrix of the scoring function
    W - scoring function class
    
    Return:
    best_ee1 [B] -- errors of selected models
    best_ss1 [B] -- scores of selected models
    models [B] -- selected models
    """
    B = best_idx.shape[0]
    best_ss1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(-float('inf'))
    best_ee1 = best_idx.new_empty([B], dtype=torch.float32).fill_(float('inf'))
    best_eer1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(float('inf'))
    best_eet1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(float('inf'))
    best_models = best_idx.new_empty([B, 3, 3])
    num_pts = data['num_pts']
    """Compute sequentially for one image pair at a time"""
    start = time.time()
    dt1 = 0
    dt2 = 0

    # unnormalized_points(data)
    # import pdb; pdb.set_trace()
    import cv2
    for b in range(best_idx.shape[0]):
        best = best_idx[b].item()
        N = num_pts[b]
        C = data['correspondences'][b][:N, :].cpu().numpy()
        R0 = data['residuals'][b, best, :N].cpu().numpy()
        t1 = time.time()
        # np.arange(R0.shape[0])[masks]
        inliers = np.arange(R0.shape[0])[
            R0 < threshold] if masks is None else masks
        if len(inliers) < 8:
           inliers = np.arange(R0.shape[0])[R0 < threshold+1]
        # run lmeds here
        EE, mask = cv2.findEssentialMat(
            C[inliers, :2],  # normalized points
   	        C[inliers, 2:],
   	        np.eye(3, 3),  # K
   	        cv2.LMEDS  # RANSAC, # method
            # cv2.RANSAC,  # method
   	        # prob = 0.99, # confidence
   	        # threshold = normalizedThreshold, # only relevant for RANSAC (threshold appropriate for normalized coordinates)
            # maxIters = max_iters,  # max iterstaions (no effect on the running time of RANSAC?)
        )

        # EE = ARO_candidates(C, R0, K1=data['K1'][b].cpu().numpy(), K2=data['K2'][b].cpu().numpy()) # pass residuals of current selected model for building nested sets
        dt1 += (time.time() - t1)
        # EE += [data['models'][b,best]] # add the current best model to the pool, so that we are guaranteed not to worsen the score TODO: reimplemnt explicitly
        # try:
        #     if len(EE) == 0:
        #         continue # ARO returned no new candidate mdoels (too few points survived threshold)
        # except:
        #     import pdb; pdb.set_trace()

        # form the new data struct for evaluation of ARO models
        data1 = dict()
        for k in set(data.keys()) - set(['residuals', 'errorrs', 'models']):
            data1[k] = data[k][b:b+1]
        data1['correspondences'] = data1['correspondences'][:, :N]

        if EE is None:
            EE = data['models'][b, best].numpy()[None, :]
            data1['models'] = torch.from_numpy(EE[None, :])
        elif EE.shape[-2] != 3:
            EE = data['models'][b, best].numpy()[None, :]
            data1['models'] = torch.from_numpy(EE[None, :])
        else:
            EE = np.stack(EE).astype(np.float32)[None, :]  # [m 3 3]
            data1['models'] = torch.tensor(
                EE).unsqueeze(0).float()  # [1 M 3 3]
        compute_residuals(data1)
        # compute errors to GT R, t
        gt_E = data['models'][b, -1].cpu().numpy()
        gt_R = data['gt_R'][b].cpu().numpy()
        gt_t = data['gt_t'][b].cpu().numpy()
        t2 = time.time()
        # changed to evaluate on the GT R, t
        errs1, errs_r1, errs_t1 = torch.tensor(
            pose_error_EERT(EE, gt_R, gt_t, C, True))  # [M]
        # errs1 = torch.tensor(pose_error_EE(EE, gt_E, C)) # [M]
        dt2 += (time.time() - t2)
        # chose the best of ARO models
        ss = sufficient_statistic(
            data1['residuals'], W.N_bins, max_distance=W.max_distance, pow=W.pow)  # [B M K]
        best_s1, best_idx1 = best_score(ss, M)
        best_ss1[b] = best_s1
        best_ee1[b] = errs1[best_idx1.item()]
        best_eer1[b] = errs_r1[best_idx1.item()]
        best_eet1[b] = errs_t1[best_idx1.item()]
        best_models[b] = data1['models'][:, best_idx1.item()]
    # best_ss1 = torch.hstack(best_ss1)
    # best_ee1 = torch.hstack(best_ee1)
    # best_models = torch.vstack(best_models)
    end = time.time()
    # print(f'ARO total:  {end - start} (ARO_candidates: {dt1}, pose error: {dt2})')
    # [B], [B], [B 3 3]
    return best_ee1, best_eer1, best_eet1, best_ss1, best_models


def best_ARO_BA(data, best_idx, M, W, threshold=1.5):
    """
    data - dict with batched data:
        correspondances [B, N, 4]
        models [B, M, 3, 3]
        errors [B, M]
        num_pts [B] number of valid (not inf) correspondences
        residuals [B, M, N]
    best_idx [B] -- selected models
    M  [N_bins] -- weight matrix of the scoring function
    W - scoring function class
    
    Return:
    best_ee1 [B] -- errors of selected models
    best_ss1 [B] -- scores of selected models
    models [B] -- selected models
    """
    B = best_idx.shape[0]
    best_ss1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(-float('inf'))
    best_ee1 = best_idx.new_empty([B], dtype=torch.float32).fill_(float('inf'))
    best_eer1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(float('inf'))
    best_eet1 = best_idx.new_empty(
        [B], dtype=torch.float32).fill_(float('inf'))

    best_models = best_idx.new_empty([B, 3, 3])
    num_pts = data['num_pts']
    """Compute sequentially for one image pair at a time"""
    start = time.time()
    dt1 = 0
    dt2 = 0

    # unnormalized_points(data)
    for b in range(best_idx.shape[0]):
        best = best_idx[b].item()
        N = num_pts[b]
        C = data['correspondences'][b][:N, :].cpu().numpy()
        R0 = data['residuals'][b, best, :N].cpu().numpy()
        t1 = time.time()
        EE, masks = ARO_candidates(C, R0, K1=data['K1'][b].cpu().numpy(), K2=data['K2'][b].cpu(
        ).numpy())  # pass residuals of current selected model for building nested sets
        dt1 += (time.time() - t1)
        # EE += [data['models'][b,best]] # add the current best model to the pool, so that we are guaranteed not to worsen the score TODO: reimplemnt explicitly
        if len(EE) == 0:
            # ARO returned no new candidate mdoels (too few points survived threshold)
            continue
        EE = np.stack(EE).astype(np.float32)  # [m 3 3]
        # form the new data struct for evaluation of ARO models
        data1 = dict()
        for k in set(data.keys()) - set(['residuals', 'errorrs', 'models']):
            data1[k] = data[k][b:b+1]
        data1['correspondences'] = data1['correspondences'][:, :N]
        data1['models'] = torch.tensor(EE).unsqueeze(0).float()  # [1 M 3 3]
        compute_residuals(data1)
        # compute errors to GT R, t
        gt_E = data['models'][b, -1].cpu().numpy()
        gt_R = data['gt_R'][b].cpu().numpy()
        gt_t = data['gt_t'][b].cpu().numpy()
        t2 = time.time()
        # changed to evaluate on the GT R, t
        errs1, errs_r1, errs_t1 = torch.tensor(
            pose_error_EERT(EE, gt_R, gt_t, C, True))  # [M]
        # errs1 = torch.tensor(pose_error_EE(EE, gt_E, C)) # [M]
        dt2 += (time.time() - t2)
        # chose the best of ARO models
        ss = sufficient_statistic(
            data1['residuals'], W.N_bins, max_distance=W.max_distance, pow=W.pow)  # [B M K]
        best_s1, best_idx1 = best_score(ss, M)
        # best_ss1[b] = best_s1
        # best_ee1[b] = errs1[best_idx1.item()]
        # best_models[b] = data1['models'][:, best_idx1.item()]
        # best_eer1[b] = errs_r1[best_idx1.item()]
        # best_eet1[b] = errs_t1[best_idx1.item()]

        # sampson_errors = Sampson(
        #     torch.concat((data1['correspondences'][0, :, :2], torch.ones_like(data1['correspondences'][0, :, :2])[:, 0, None]), dim=-1),
        #     torch.concat((data1['correspondences'][0, :, 2:4], torch.ones_like(data1['correspondences'][0, :, 2:4])[ :, 0, None]), dim=-1),
        #     data1['models'][:, best_idx1.item()][0]
        # )
        # residuals = data1['residuals'][:, best_idx1.item()].cpu().numpy()[0]
        # inliers = np.arange(residuals.shape[0])[residuals < threshold]
        # EE = numerical_optimization(C, data['K1'][b].cpu().numpy(), data['K2'][b].cpu().numpy(), inliers,
        #                             data1['models'][:, best_idx1.item()]
        # .cpu().numpy(), threshold, 0)

        best_ee1[b], best_eer1[b],  best_eet1[b], best_ss1[b], best_models[b] = best_BA(
            data1, best_idx1, M, W, threshold, masks[best_idx1.item()])
    # best_ss1 = torch.hstack(best_ss1)
    # best_ee1 = torch.hstack(best_ee1)
    # best_models = torch.vstack(best_models)
    end = time.time()
    # print(f'ARO total:  {end - start} (ARO_candidates: {dt1}, pose error: {dt2})')
    # [B], [B], [B 3 3]
    return best_ee1, best_eer1, best_eet1, best_ss1, best_models
