# %%
import matplotlib.colors as mcolors
from .. import local_optimization as LO
from .aro import ARO_candidates, pose_error_EE, pose_error_EERT
from ..load_data import *
from ..score_weights import *
from types import SimpleNamespace
import os
import sys
from pathlib import Path
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

if __name__ == "__main__":
    __name__ = 'score_learn.evaluate.py'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False
# %%
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import math
import time
#!%matplotlib inline

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


#!%matplotlib inline
# some defaults
batch_size = 32
is_F = False
# ARO_polish = True
# BA_polish = False

polish = 0
# 0 - no polish
# 1 - BA
# 2 - LMeDs
# 3 - ARO
# 4 - ARO + BA
# 5 - LMeds + BA
# 7 - our polish (EM-LMA)
# datasets
PhotoTourism_test = [
    'florence_cathedral_side',
    'british_museum',
    'lincoln_memorial_statue',
    'london_bridge',
    'milan_cathedral',
    'mount_rushmore',
    'piazza_san_marco',
    'reichstag',
    'sagrada_familia',
    'st_pauls_cathedral',
    'united_states_capitol'
]

scannet_train = ['scannet/train']
scannet_test = ['scannet/test']

PhotoTourism_val = [
    'buckingham_palace',
    'brandenburg_gate',
    'colosseum_exterior',
    'grand_place_brussels',
    'notre_dame_front_facade',
    'palace_of_westminster',
    'pantheon_exterior',
    'prague_old_town_square',
    'sacre_coeur',
    'taj_mahal',
    'trevi_fountain',
    'westminster_abbey'
]

Lamar_train = ['cab_train', 'lin_train', 'hge_train']
Lamar_test = ['cab_test', 'lin_test', 'hge_test']

eth3d_test = ['eth3d']
eth3d_train = ['eth3d_train']

# %% Validation / Test config:

# KITTI setup
# val_scenes = ['KITTI/train/']
# test_scenes = ['KITTI/test']
# is_F = True
# pth = "./models/KITTI/"
# batch_size = 8

# PhotoTourism
val_scenes = PhotoTourism_val
test_scenes = PhotoTourism_test
pth = "./models/SPSG/"


# eth3d
# val_scenes =eth3d_train
# test_scenes =eth3d_test
# pth = "./models/SPSG/"


# # Lamar generalization
# val_scenes = PhotoTourism_val #  Lamar_train#
# test_scenes = Lamar_train + Lamar_test
# pth = "./models/3/"


# %%
def best_score(ss, M):
    scores = ss.counts @ M  # [B M K] @ [K T] -> [B M T]
    best_s, best_idx = scores.max(dim=1)  # [B T] / [B]
    best_s = best_s.cpu()
    best_idx = best_idx.cpu()
    return best_s, best_idx  # [B T] / [B]


def select_dim1(source, best_idx):
    B = best_idx.shape[0]
    best_e = source[torch.arange(B).view(
        [-1] + [1]*(best_idx.dim()-1)).expand(best_idx.shape), best_idx]  # [B T]
    return best_e  # [B T] / [B]


# def error_of_the_best(ss, errors, M):
#     scores = ss.counts @ M  # [B M K] @ [K T] -> [B M T]
#     best_idx = scores.argmax(dim=1).cpu()  # [B T] / [B]
#     B = best_idx.shape[0]
#     best_e = errors[torch.arange(B).view([-1] + [1]*(best_idx.dim()-1)).expand(best_idx.shape), best_idx]  # [B T]
#     return best_e, best_idx, scores  # [B T] / [B]


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


if polish == 1:
    polish_func = best_BA
elif polish == 2:
    polish_func = best_LMEDS
elif polish == 3:
    polish_func = best_ARO
elif polish == 4:
    polish_func = best_ARO_BA
else:
    print("undefined polishing method!")


def local_optimization_ours(data, models):
    """
    data - dict with batched data:
        correspondences [B, N, 4]
        K1,
        K2
    models [B, 3, 3] -- current best models
    return:
     rmodels [B, 3, 3] -- optimized best models
    """
    x, y = normalized_points(data)
    KX = data['K1'].to(x)
    KY = data['K2'].to(x)
    models = models.to(x)

    def score_f(rr):
        s = WGU.score_residuals(rr, reduction=None)
        return s

    def weight_f(rr):
        s = WGU.IRLS_weight(rr)
        return s
    E1 = LO.local_optimization(
        x, y, KX, KY, LO.E_parameterization(models), score_f, weight_f)
    return E1


def evaluate(loader, mode):
    global Oe
    global Or
    global Ot
    global Me
    global methods

    for idx, data in enumerate(loader):
        # F = data['models'][:, :-1].cuda()  # all models [B M]
        compute_residuals(data)
        R = data['residuals'].cuda()
        R = R[:, :-1, :]  # select all but GT model [B M N]
        errors = data['errors'][:, :-1]  # select all but GT model [B M]
        hash = dict()
        for i, models in enumerate(data['models']):
            nan_mask = np.unique(np.where(np.isnan(models))[0])
            if len(nan_mask) != 0:
                data['models'][i][nan_mask] = torch.eye(3)
                data['errors'][i][nan_mask] = 180.
        # find the oracle model by the minimal errors
        ora_idx = torch.argmin(errors, dim=-1)
        best_ora, best_idx_ora = errors.min(dim=-1)
        if mode == 'test':
            C = data['correspondences'].numpy()
            best_ora_models = []
            best_ora_r = []
            best_ora_t = []

            for i, b_i in enumerate(best_idx_ora):
                best_ora_models += [data['models'][i, b_i.item()]]
            best_ora_models = torch.stack(best_ora_models).numpy()
            gt_R = data['gt_R'].numpy()
            gt_t = data['gt_t'].numpy()

            for b in range(best_idx_ora.shape[0]):
                try:
                    _, best_ora_r_b, best_ora_t_b = pose_error_EERT(
                        best_ora_models[b][None, :], gt_R[b], gt_t[b], C[b], combine=True)  # [M]
                except:
                    import pdb
                    pdb.set_trace()
                best_ora_r += [best_ora_r_b]
                best_ora_t += [best_ora_t_b]

        for W in methods:
            W.to(R)
            key = (W.N_bins, W.max_distance, W.pow)
            if key not in hash:
                ss = sufficient_statistic(
                    R, W.N_bins, max_distance=W.max_distance, pow=W.pow)  # [B M K]
                hash[key] = ss
            else:
                ss = hash[key]
            if mode == 'val' and not hasattr(W, 'locked'):
                M = W.M.T
            else:
                M = W.val_w
            best_s, best_idx = best_score(ss, M)

            best_e = select_dim1(errors, best_idx)  # used in validation

            # oracle errors

            if mode == 'test' and polish == 7:
                # best_R = select_dim1(R, best_idx)
                # score0 = WGU.score_residuals(best_R)
                # print(score0[0:10])
                E = select_dim1(data['models'], best_idx).cuda()
                E1, best_scores = local_optimization_ours(data, E)
                best_models = E1.cpu().numpy()
                best_e1 = []
                best_r = []
                best_t = []
                for b in range(best_models.shape[0]):
                    best_e_b, best_r_b, best_t_b = pose_error_EERT(
                        best_models[b][None, :], gt_R[b], gt_t[b], C[b], combine=True)  # [M]
                    best_e1 += [best_e_b]
                    best_r += [best_r_b]
                    best_t += [best_t_b]
                best_e1 = np.stack(best_e1)[:, 0]
                best_r = np.stack(best_r)[:, 0]
                best_t = np.stack(best_t)[:, 0]
                best_e = best_e1

                # compute error of optimized models

            if mode == 'test' and polish != 7:

                # if polish == 0:
                best_models = []
                best_e = []
                best_r = []
                best_t = []
                for i, b_i in enumerate(best_idx):
                    best_models += [data['models'][i, b_i.item()]]
                best_models = torch.stack(best_models).numpy()

                for b in range(best_idx.shape[0]):
                    best_e_b, best_r_b, best_t_b = pose_error_EERT(
                        best_models[b][None, :], gt_R[b], gt_t[b], C[b], combine=True)  # [M]
                    best_e += [best_e_b]
                    best_r += [best_r_b]
                    best_t += [best_t_b]
                best_e = np.stack(best_e)[:, 0]
                best_r = np.stack(best_r)[:, 0]
                best_t = np.stack(best_t)[:, 0]

                # best_models = data['models'][torch.concat((torch.arange(32)[:, None], best_idx[:, None]), dim=-1).tolist()]
                # [data['models'][i] for i in torch.concat((torch.arange(32)[:, None], best_idx[:, None]), dim=-1).tolist()]
                # 0 - no polish
                # 1 - BA
                # 2 - LMeDs
                # 3 - ARO
                # 4 - ARO + BA
                # 5 - LMeds + BA
                if polish != 0:
                    best_ee1, best_eer1, best_eet1, best_ss1, best_models1 = polish_func(
                        data, best_idx, M, W)
                    ora_ee1, best_eer1, best_eet1, ora_ss1, ora_models1 = polish_func(
                        data, ora_idx, M, W)

                    m = best_ss1 > best_s  # where ARO models are improving the score
                    best_e[m] = best_ee1[m]  # record their GT error
                    # import pdb; pdb.set_trace()
                    best_r[m] = best_eer1[m]
                    best_t[m] = best_eet1[m]

                    ora_s = torch.as_tensor(
                        [s[ora_idx[i]].item() for i, s in enumerate(ss.counts @ M)])
                    m_ora = ora_ss1 > ora_s  # where ARO models are improving the score
                    best_ora[m_ora] = ora_ee1[m_ora]  # record their GT error
            # if mode == 'test' and ARO_polish and W.name == 'ML   pi=30.0':
            #     import pdb; pdb.set_trace()
            #     best_ee1, best_ss1, best_models1 = best_ARO_BA(data, best_idx, M, W)
            #     m = best_ss1 > best_s # where ARO models are improving the score
            #     best_e[m] = best_ee1[m] # record their GT error
            #     # best_models[m] = best_models1

            #     # do the same thing on oracle model
            #     ora_ee1, ora_ss1, ora_models1 = best_ARO_BA(data, ora_idx, M, W)
            #     ora_s = torch.as_tensor([s[ora_idx[i]].item() for i, s in enumerate(ss.counts @ M )])
            #     m_ora = ora_ss1 > ora_s # where ARO models are improving the score
            #     best_ora[m_ora] = ora_ee1[m_ora] # record their GT error

            # elif mode == 'test' and BA_polish:
            #     if pymagsac_available:
            #         best_ee1, best_ss1, best_models1 = best_BA(data, best_idx, M, W)
            #         m = best_ss1 > best_s # where ARO models are improving the score
            #         best_e[m] = best_ee1[m] # record their GT error

            #         # do the same thing on oracle model
            #         ora_ee1, ora_ss1, ora_models1 = best_BA(data, ora_idx, M, W)
            #         ora_s = torch.as_tensor([s[ora_idx[i]].item() for i, s in enumerate(ss.counts @ M )])
            #         m_ora = ora_ss1 > ora_s # where ARO models are improving the score
            #         best_ora[m_ora] = ora_ee1[m_ora]

            #     else: # use
            #         pass

            W.best_e += [best_e]
            if mode == 'test':
                W.best_r += [best_r]
                W.best_t += [best_t]

        # import pdb; pdb.set_trace()
        Oe += [best_ora.numpy()]
        if mode == 'test':
            Or += [np.stack(best_ora_r)[:, 0]]
            Ot += [np.stack(best_ora_t)[:, 0]]
        Me += [errors.mean(dim=-1).numpy()]
        if idx % 10 == 0:
            print(idx)
        # if idx >= 3000:
        #     break


def test_ARO():
    global Oe
    global Me
    global methods
    W = ScoreWeightsMAGSAC(maximum_threshold=10, N_bins=500, max_distance=10)
    W.pow = 1
    W.val_w = W.score_weights_normalized().cuda()
    W.locked = True
    W.name = 'MAGSAC++ (tau=10)'
    W.best_e = []
    Oe = []
    Me = []
    dataset = ResidualData(
        val_scenes[0], padding=True, sqrt=True, F=is_F)  # padding the
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    for idx, data in enumerate(loader):
        EE = data['models'][0, :-1].cpu().numpy()  # all models but GT [B M]
        gt_E = data['models'][0, -1].cpu().numpy()
        N = data['num_pts'][0]
        C = data['correspondences'][0, :N].cpu().numpy()
        errs0 = data['errors'][0, :-1].cpu().numpy()
        errs = pose_error_EE(EE, gt_E, C)
        # print(errs0)
        # print(errs)
        # assert (np.abs(errs0 - errs).max()< 0.1)
        break
    methods = [W]
    evaluate(loader, 'test')

    # for idx, data in enumerate(loader):
    #     F = data['models'][:, :-1].cuda()  # all models [B M 3 3]
    #     compute_residuals(data)
    #     R = data['residuals'].cuda()
    #     R = R[:, :-1, :]  # select all but GT model [B M N]
    #     E = data['errors'][:, :-1]  # select all but GT model [B M]
    #     num_pts = data['num_pts']

    #     best_idx = -1
    #     best_M = F[:, best_idx, ...]

    #     for b in range(best_M.shape[0]):
    #         # F0 = best_M[b].cpu() #
    #         C = data['correspondences'][b][0:num_pts[b], :].cpu().numpy()
    #         R0 = data['residuals'][b, best_idx,:num_pts[b]].cpu().numpy()
    #         EE = ARO(C, R0)

    #     break

# test_ARO()

# exit(0)

# src = '/tmp/RANSAC/KITTI/train/'; F_matrix = True
# pth = "./models/eth3d/"


# _________Load models______________
# %% Load modes
print('________Loading models___________')
max_distance = 10
N_bins = 500  # filter to select models to evaluate
methods = []
# pth = "./models/RS/"
# pth = "./models/KITTI/"
files = os.listdir(pth)
# print(files)
# files = ['monotone_tau=10.npz']
# files = ['monotone_tau=10_bins=2000.npz']
# files = ['msac_tau=3.0.npz']
mdict = dict()
for file in files:
    W = None
    f = os.path.join(pth, file)
    if file.endswith(".pkl"):
        W = torch.load(f)
        if isinstance(W, ScoreWeightsMonotoneMix):
            continue
        if isinstance(W, ScoreWeightsTZ):
            continue
        # if isinstance(W, ScoreWeightsMonotoneMix) and W.alpha > 0.1 and not hasattr(W, 'M'):
            # continue
        if not hasattr(W, 'pow'):
            W.pow = 1
        W.name = file.replace('.pkl', '').replace('magsac', 'MAGSAC++').replace('ransac', 'RANSAC').replace(
            'msac', 'MSAC').replace('_', ' ').replace('bins=500', '').replace('tau=10.0', '').replace('alpha', 'pi')
        print(W.name + ',\t max_distance=' + str(W.max_distance))
        if not hasattr(W, 'M'):
            W.gen_hyperparams(100)
            M = W.score_matrix()
            W.register_buffer('M', M)
        else:
            print('loaded M for ' + W.name)
        W.cuda()
        mdict[W.name] = W
        methods.append(W)

methods = []
# methods += [mdict['RANSAC']]
# methods += [mdict['MSAC']]
# methods += [mdict['MAGSAC++']]

# Add MAGSAC-10
# W = ScoreWeightsMAGSAC(maximum_threshold=10, N_bins=methods[0].N_bins, max_distance=methods[0].max_distance)
# W.pow = 1
# W.val_w = W.score_weights_normalized().cuda()
# W.locked = True
# W.name = 'MAGSAC++ (tau=10)'
# methods += [W]

# # Add TZ
# W = ScoreWeightsTZ(N_bins=methods[0].N_bins, max_distance=max_distance, alpha=None, max_outlier_dist=100.0, pow=1)
# W.name = 'TZ'
# W.cuda()
# W.gen_hyperparams(100)
# M = W.score_matrix()
# W.register_buffer('M', M)
# methods += [W]

# Add GU
W = ScoreWeightsGU(N_bins=N_bins, max_distance=max_distance, pow=1)
W.name = 'GU'
W.cuda()
W.gen_hyperparams(100)
M = W.score_matrix()
W.register_buffer('M', M)
WGU = W
methods += [W]


for k in mdict.keys():
    if 'ML' in k:
        methods += [mdict[k]]

# add remaining
for k in mdict.keys():
    if mdict[k] not in methods:
        methods += [mdict[k]]


# %%
results = dotdict()
results['method'] = []
vresults = dotdict()
# %% ____________Validation______________________________________
for M in methods:
    M.best_e = []
Oe = []
Me = []
# validate on all val scenes
for val_src in val_scenes:
    val_src_name = val_src.replace('/', '_')
    print(f'__Validation on {val_src}___________')
    dataset = ResidualData(val_src, padding=True,
                           sqrt=True, F=is_F)  # padding the
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    evaluate(loader, 'val')

for M in methods:
    M.best_e = np.concatenate(M.best_e, axis=0)  # [N T]
Oe = np.concatenate(Oe, axis=0)
Me = np.concatenate(Me, axis=0)

# %%
cc = list(mcolors.TABLEAU_COLORS)
cc = cc + cc

stat = np.median
# stat = np.mean
#
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
for (i, W) in enumerate(methods):
    if hasattr(W, 'locked'):
        continue
    if 'ML' in W.name:  # or 'TZ' in W.name: #  and not 'mult' in W.name:
        ax = ax2
    else:
        ax = ax1
        # continue
    v = stat(W.best_e, axis=0)
    v_besti = np.argmin(v)
    W.set_hyperparam(W.hyperparams[v_besti])
    W.best_hyperparam = W.hyperparams[v_besti]
    W.hbest_i = v_besti
    ax.plot(W.hyperparams, v, label=W.name, color=cc[i])
    ax.plot(W.best_hyperparam, v[v_besti], 'o', label=None, color=cc[i])
# plt.axhline(stat(Oe),0,1, color = "k", label= 'Oracle')
plt.legend()
ax1.legend(loc=1)
ax2.legend(loc=2)
ax1.set_xlabel('Threshold $\\tau$ [px]')
ax1.locator_params(axis='x', nbins=10)
ax2.set_xlabel('Inliers prior $\\pi$')
ax2.locator_params(axis='x', nbins=10)
ax1.locator_params(axis='y', nbins=10)
ax2.set_xscale('logit')
ax1.set_ylabel(f'{stat.__name__} pose error')
plt.draw()
Path('fig').mkdir(exist_ok=True)
plt.savefig(f'fig/val_{val_src_name}.pdf')
plt.show()

# %%
# assert(False)
plt.figure()
for (i, W) in enumerate(methods):
    if hasattr(W, 'locked'):
        continue
    # if 'ML' in W.name and 'mult' in W.name:
        # continue
    # w = W.score_weights_normalized()
    w = W.M[W.hbest_i]
    W.val_w = w
    pow = 1
    xx = (torch.arange(w.shape[0])/w.shape[0]*W.max_distance**pow)**(1/pow)
    # or isinstance(W, ScoreWeightsTZ):
    if isinstance(W, ScoreWeightsMonotoneMix):
        hparam = f'$\\pi={W.best_hyperparam*100:3.1f}$%'
    else:
        hparam = f'$\\tau={W.best_hyperparam:3.2f}$'
    plt.plot(xx, w.cpu().detach(), label=W.name + f' ({hparam})', color=cc[i])
plt.legend()
plt.title('Selected kernels')
# plt.xlim(0,7)
plt.xlabel('Residual [px]')
plt.draw()
plt.savefig(f'fig/val_{val_src_name}_kernels.pdf')

# %%
# assert(False)
# %% TEST
# ________________________ TEST________________________________________________
print('________TEST___________')
results[val_src] = []
if 'average' in results.keys():
    del results['average']
# results['average'] = []

for M in methods:
    M.best_e = []
    M.best_r = []
    M.best_t = []
Oe = []
Or = []
Ot = []

Me = []

for src in test_scenes:
    print(src)
    dataset = ResidualData(src, padding=True, sqrt=True, F=is_F)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    evaluate(loader, 'test')


# joint results for all test scenes
for M in methods:
    M.best_e = np.concatenate(M.best_e, axis=0)  # [N T]
    M.best_r = np.concatenate(M.best_r, axis=0)  # [N T]
    M.best_t = np.concatenate(M.best_t, axis=0)  # [N T]
Oe = np.concatenate(Oe, axis=0)
Or = np.concatenate(Or, axis=0)
Ot = np.concatenate(Ot, axis=0)

Me = np.concatenate(Me, axis=0)

# concatenate oracle results
methods += [SimpleNamespace(best_e=Oe, best_r=Or, best_t=Ot, name='Oracle')]
# methods += [SimpleNamespace(best_e=Me, best_r = Or, best_t = Ot, name='Random')] # wait to modify it

if len(results.method) == 0:
    for M in methods:
        results['method'].append(M.name)

print('Error stats of method-selected model')
stats = [np.median]  # np.mean,
maxl = 0
for M in methods:
    maxl = max(maxl, len(M.name))
print("results on pose ")

for (mi, M) in enumerate(methods):
    print(M.name.ljust(maxl), end=': ')
    for stat in stats:
        try:
            res = scipy.stats.bootstrap(
                (M.best_e,), stat, confidence_level=0.95, method='BCa', n_resamples=10000)
        except:
            print(np.median(M.best_e))
            continue
        ci = res.confidence_interval
        # d = (ci.high - ci.low)/2
        v = stat(M.best_e)
        d = max(ci.high-v, v-ci.low)
        formatted = format_std(v, d)
        print(f'\t {stat.__name__}={formatted}', end='')
        if stat == np.median:
            # results['method'].append(M.name)
            results[val_src].append(formatted)
            if vresults[M.name] is None:
                vresults[M.name] = []
            vresults[M.name].append(v)
            # results['average'].append(np.array(vresults[M.name]).mean())
    print('')
    # print(f' std={np.std(M.best_e)}')
# %%
results['hyperparam'] = []
for W in methods:
    if isinstance(W, ScoreWeightsMonotoneMix) or isinstance(W, ScoreWeightsMonotoneMix):
        hparam = f'$\\pi={W.best_hyperparam*100:3.1f}$%'
    elif hasattr(W, 'best_hyperparam'):
        hparam = f'$\\tau={W.best_hyperparam:3.2f}$'
    else:
        hparam = ''
    results['hyperparam'].append(hparam)

# %%

df = pd.DataFrame.from_dict(results)
mnames = [m.name for m in methods]
df['method'] = pd.Categorical(df['method'], mnames)

# display(df)
ss = df.to_latex(index=True)
print(ss.replace('±', '$\pm$'))

# %%

# remove random and oracle resutls
del methods[-2:]

# print(df.to_csv(index=False))

# df2 = df.groupby(by=['data', 'batch_size'], axis=1).mean()
# df1 = df.groupby(['method', 'zbits','data']).mean()
# df1 = df.groupby(list(set(o1.keys())-{'seed', 'L'})).mean()
# df2 = df1.drop('seed', axis=1)
# dfs = dict(tuple(df.groupby('batch_size')))
# for k in dfs.keys():
#     print(f"batch_size={k}")
#     d = dfs[k]
#     df3 = d.drop(['lr', 'batch_size'], axis=1)
#     df4 = df3.set_index(['data', 'zbits', 'method'])
#     df5 = df4.unstack(level=[0, 1])  # .unstack(level=0)
#     # df5 = df5.unstack(level=0)  # .unstack(level=0)
#     # df5 = df5.style.set_properties(**{'text-align': 'left'})
#     display(df5)
#     ss = df5.to_latex(index=True)
#     print(ss.replace('±', '$\pm$'))
#     print(df5.to_csv(index=False))


# %%
f = plt.figure()
for M in methods[0:5]:
    y, bine = np.histogram(M.best_e, 20, range=(0, 50), density=True)
    plt.plot(bine[:-1], y, label=M.name)
    # plt.hist(M.best_e)
plt.legend()
plt.show()
plt.close(f)
# %%
