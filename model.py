import torch

__name__ = 'score_learn.model_H.py'
__package__ = 'score_learn'

import numpy as np

from .functional import *
import kornia
import cv_utils
import time
import math

def normalize_points(x, K1):
    """
    x [... N 3]
    y [... N 3]
    K1 [... 3 3]
    """
    K1I = K1.inverse()
    x = torch.einsum('...ij, ...nj -> ...ni', K1I, x)
    return x

def unnormalize_points(x, K):
    """
    x [... N 3]
    K [... 3 3]
    """
    x = torch.einsum('...ij, ...nj -> ...ni', K, x)
    return x


def new_minimal_models(data, m_batch_size, max_average_sol=None, include_GT=False, solver = None, min_sample=None, validation_fn=None):
    """
    Create new minimal models using minimal solver
    Correspondences:
    x [B, max_N, 3]
    y [B, max_N, 3] 
    """
    C = data['correspondences'] # [B, max_N, 4]
    if C.shape[-1] ==6:
        xx = C[..., :3]
        xx = xx / xx[..., 2:3]
        yy = C[..., 3:]
        yy = yy / yy[..., 2:3]
    elif C.shape[-1] == 4:
        xx = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
        yy = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
    n_points = data['num_pts']
    max_models = 0
    models = [[] for b in range(C.shape[0])]
    for b in range(C.shape[0]):
        n = n_points[b]
        mx = xx[b,:n]
        my = yy[b,:n]
        # log_p = (C.new_ones((n,))/n).log() #uniform sampling
        # sampling probabilities based on SNN ratio, small SNN is better -> more likely to draw
        snn = data['snn'][b,:n]
        # what is the sigma of Normal distribution such that 95% of values are within [0,0.95]?
        sigma = 1/8.0
        p = torch.exp(-snn**2/(2*sigma**2)) + 1e-6
        # p = snn < 0.7 + 0.001 # hard thresholding
        log_p = (p / p.sum()).log() # convert to categorical distribution over all points and take log
        EE = []
        II = []
        start_time = time.time()
        max_iters = 10
        iter = 0
        if False:
            mask = snn < 0.7
            mx = mx[mask]
            my = my[mask]
            n = mx.shape[0]
            log_p = (C.new_ones((n,))/n).log() #uniform sampling
        
        while len(EE) < m_batch_size and iter < max_iters and n >= min_sample:
            ii = sample_visible_subsets(min_sample, log_p, m_batch_size, mx, my) # [M, min_sample]
            x1 = mx[ii,:].cpu().numpy().astype(float) # [M, min_sample, 3] coordinates at minimal sample
            y1 = my[ii,:].cpu().numpy().astype(float) # [M, min_sample, 3]
            E, I = solver(x1,y1) # E: list of [3,3] numpy, I: list of indices in ii
            E = torch.tensor(np.array(E))
            I = torch.tensor(np.array(I))
            # check cheirality of the points we used to estimate the models
            if E.shape[0] == 0:
                iter += 1
                continue
            if validation_fn is not None:
                mask = validation_fn(E, mx, my, ii[I])
            else:
                mask = torch.ones(len(E), dtype=torch.bool, device=E.device)
            EE.extend(E[mask].tolist())
            II.extend(I[mask].tolist())
            iter += 1
        # print(f"Loop for batch {b} took {elapsed_time:.4f} seconds")

        # get the required number of valid models
        if len(EE) > m_batch_size:
            EE = EE[:m_batch_size]
            II = II[:m_batch_size]

        if len(EE) < m_batch_size: # this is to prevent errors in decomposition, etc. for the whole batch
            num_missing = m_batch_size - len(EE)
            EE.extend([torch.eye(3, 3, dtype=torch.float32)] * num_missing)
            II.extend([0] * num_missing)

        if False and b==0: # DEBUG TEST
            print('Indices of first model:', II[0])
            x1 = mx[ii[II[0]]].cpu()
            y1 = my[ii[II[0]]].cpu()
            M = torch.tensor(EE[0], dtype = x1.dtype)
            reprojected_y1 = (M @ x1.T).T
            reprojected_y1 = reprojected_y1 / reprojected_y1[:,2:3]
            err = (reprojected_y1 - y1).norm(dim=1)
            print('Reprojection errors of first model (px):', err)
            r = SampsonBM(x1[None,:, :], y1[None,:, :], M[None, None, :, :])[0,0]
            print('Sampson errors of first model in normalized coordinates:', r)
            K1 = data['K1'][b].cpu()
            K2 = data['K2'][b].cpu()
            x1 = unnormalize_points(x1, K1)
            y1 = unnormalize_points(y1, K2)
            M = unnormalize_models(M[None,:,:], K1, K2)[0]
            reprojected_y1 = (M @ x1.T).T
            reprojected_y1 = reprojected_y1 / reprojected_y1[:,2:3]
            err_px = (reprojected_y1 - y1).norm(dim=1)
            print('Reprojection errors of first model (px):', err_px)
            r = SampsonBM(x1[None,:, :], y1[None,:, :], M[None, None, :, :])[0,0]
            print('Sampson errors of first model in pixel coordinates:', r)
        
        n_models = len(EE)
        max_models = max(max_models, n_models)
        models[b] = EE
    # assert(max_models > 1000)
    if max_average_sol is not None:
        max_models = min(max_models, m_batch_size*max_average_sol)
    for b in range(C.shape[0]):
        if len(models[b]) > max_models:
            models[b] = models[b][:max_models]
        else:
            models[b] = np.concatenate([np.stack(models[b]), np.zeros((max_models - len(models[b]), 3, 3))]) # pad models with zeros -- Ok for E but maybe not for H?
    models = np.stack(models) # stack along batch dim
    models = torch.tensor(models).to(dtype= torch.float32).cpu()
    #
    # data['models'][:,:1000] = models # use 1K models
    if include_GT:
        GTmodels = data['models'][:,-1:].to(models)
        models = torch.cat([models, GTmodels], dim = 1)
    data['models'] = models


# __________________________________________________________


def R_error(R: Tensor, R_gt: Tensor) -> Tensor:
    """
    calculate angle between provided rotations
    R [..., 3, 3]
    R_gt [..., 3, 3]
    Return: [...]
    """
    if False: 
        """REFERENCE SOLUTION"""
        s = list(R.shape)[:-2]
        r = R.new_zeros(s)
        for b in range(R.shape[0]):
            R2 = R_gt[b][0]
            for i, R1 in enumerate(R[b]):
                r[b, i] = np.linalg.norm(TR.from_matrix((R1 @ R2.T).cpu().numpy()).as_rotvec())
                # r[b,i] = torch.arccos(torch.max(torch.min((torch.trace(R1 @ R2.transpose(0, 1)) - 1) * 0.5, torch.tensor(1.0, device=R.device)), torch.tensor(-1.0, device=R.device)))
        e_R = r / math.pi * 180  # [...]
    else:
        dR = torch.einsum('...ij, ...kj -> ...ik', R, R_gt)  # R R_gt^T
        r = kornia.geometry.conversions.rotation_matrix_to_axis_angle(dR.view(-1, 3, 3)).view(dR.shape[:-1])  # [..., 3]
        e_R = r.norm(dim=-1) / math.pi * 180  # [...]    
        e_R = torch.nan_to_num(e_R, nan=float('180')) # for padded models
    return e_R

def t_error(T: Tensor, T_gt: Tensor) -> Tensor:
    """
    R [..., 3]
    R_gt [..., 3]
    Return: [...]
    """
    cos = (T*T_gt).sum(dim=-1)/(T.norm(dim=-1)*T_gt.norm(dim=-1))  # [*]
    cos = cos.clip(min=-1, max=1)  # numerical fix
    e_t = torch.acos(cos) / math.pi * 180
    e_t = torch.nan_to_num(e_t, nan=float('180')) # for padded models
    return e_t


def auc_(data, threshold):
    # return cv_utils.AUC(data, thresholds=[threshold])[0]
    return cv_utils.pose_auc(data, thresholds=[threshold])[0]

def AUC_10(data, axis=-1):
    threshold = 10
    r = []
    if data.ndim == 2:
        assert (axis == -1)
        for i in range(data.shape[0]):
            r += [auc_(data[i], threshold)]
        r = np.array(r)
    else:
        assert (data.ndim == 1)
        assert (axis == 0 or axis == -1)
        r = auc_(data, threshold)
    return r
    # return cv_utils.AUC(data, thresholds=[10], binsize=50)


