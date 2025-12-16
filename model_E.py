import torch

__name__ = 'score_learn.model_H.py'
__package__ = 'score_learn'

import numpy as np

from .functional import *
import poselib
import kornia
import cv_utils
import cv2

from . import model
from .model import R_error, t_error

#______________________ Sampson Error _____________________________________
def Sampson(x: Tensor, y: Tensor, F: Tensor) -> Tensor:
    """
    x [n x 3]
    y [n x 3]
    F [3 x 3]
    """
    xF = x @ F # [n, 3]
    Fy = y @ F.T # [n,3 ]
    numerator = (xF * y).sum(dim=-1) # [n]
    denom = ((xF[:, 0:2]**2 + Fy[:, 0:2]**2).sum(dim=-1))**0.5
    return numerator/denom


def SampsonBM(x: Tensor, y: Tensor, F: Tensor) -> Tensor:
    """
    Computes (x'Fy) / ((x'F)[0:2]^2 + (Fy)[0:2]^2)^0.5
    x [B x n x 3]
    y [B x n x 3]
    F [B x M x 3 x 3]
    """
    xF = torch.einsum('bni, bmij -> bmnj', x, F[..., 0:2])  # [B, M, n, 2]
    d1 = (xF**2).sum(dim=-1)  # [B, M, n]
    del xF
    Fy = torch.einsum('bnj, bmij -> bmni', y, F[...,0:2,:])  # [B, M, n, 2]
    d2 = (Fy**2).sum(dim=-1)  # [B, M, n]
    del Fy
    denom = (d1 + d2)**0.5 + 1e-15
    numerator = torch.einsum('bni, bnj, bmij -> bmn', x, y, F) + 1e-5  # [B, M, n]
    return numerator/denom  # [B, M, n]

def SampsonM(x: Tensor, y: Tensor, F: Tensor) -> Tensor:
    """
    x [n x 3]
    y [n x 3]
    F [M x 3 x 3]
    """
    xF = torch.einsum('ni, mij -> mnj',x,F)  # [M, n, 3]
    Fy = torch.einsum('nj, mij -> mni',y,F)  # [M, n, 3]
    numerator = torch.einsum('ni, nj, mij -> mn',x,y,F)  # [M, n]
    denom = ((xF[..., 0:2]**2 + Fy[..., 0:2]**2).sum(dim=-1))**0.5
    return numerator/denom # [M, n]

def SampsonJJ(x: Tensor, y: Tensor, F: Tensor) -> Tensor:
    """
    Computes (x'Fy) / ((x'F)[0:2]^2 + (Fy)[0:2]^2)^0.5
    J = 
    x [N x 3]
    y [N x 3]
    F [M x 3 x 3]
    """
    M = F.shape[0]
    N = x.shape[0]
    x = x.unsqueeze(0).tile([M, 1, 1]).detach().requires_grad_(True)  # [M N 3]
    y = y.unsqueeze(0).tile([M, 1, 1]).detach().requires_grad_(True)  # [M N 3]
    xF = torch.einsum('mni, mij -> mnj',x,F)  # [M, N, 3]
    Fy = torch.einsum('mnj, mij -> mni',y,F)  # [M, N, 3]
    numerator = torch.einsum('mni, mnj, mij -> mn',x,y,F)  # [M, N]
    denom = ((xF[..., 0:2]**2 + Fy[..., 0:2]**2).sum(dim=-1))**0.5 # [M, N]
    residuals = numerator/denom  # [M, N]
    residuals.sum().backward()
    G1 = x.grad[...,:2] # grad except in homogenous 1
    G2 = y.grad[...,:2]
    JJ = ((G1**2).sum(dim=-1) + (G2**2).sum(dim=-1))**0.5  # [M, N]
    logJJ = torch.log(JJ).sum(dim=-1) #[M]
    # assert JJ.shape == residuals.shape
    return residuals.detach(), logJJ

def estimate_volume(F, K1, K2,  estimate_distance=50.0, n_points=1_000, repeats=10):
    vol = 0
    sz1 = 2*K1[:, :-1, -1] # [B 2]
    sz2 = 2*K2[:, :-1, -1] # [B 2]
    N = 500
    for rep in range(repeats):
        with torch.no_grad():
            # normalized coordinates in [-1,1]
            x = (torch.rand([2*n_points, 2], device='cuda'))
            y = x[:n_points, :]
            x = x[n_points:, :]
            x = torch.einsum('ni, bi -> bni', x, sz1) # scale by (2*cx, 2*cy)
            y = torch.einsum('ni, bi -> bni', y, sz2) # scale by (2*cx, 2*cy)
            x = torch.cat([x, torch.ones_like(x[...,:1])], dim=-1)
            y = torch.cat([y, torch.ones_like(x[...,:1])], dim=-1)
            # unnormalized coordinates
            # x = torch.einsum('bij, nj -> bni', K1, x) # (K1)x in the format [n,3]
            # y = torch.einsum('bij, nj -> bni', K2, y) # (K2)y in the format [n,3]
            # scale = (K1[:,0, 0] + K1[:,1, 1] + K2[:,0, 0] + K2[:,1, 1])/4
            r = SampsonBM(y, x, F).abs() #*scale.view([-1,1,1]) #[B M N]
            vol = vol + (r < estimate_distance).sum(dim=-1)/n_points  # [B M]
    vol /= repeats
    return vol # [B M]


def score_vol(ss, vol, estimate_distance, W):
    """
    ss.counts [B M N]
    vol [B M]
    """
    vol_bin = vol / estimate_distance * W.bin_size() # (x,y) volume per bin [B, M]
    logV_bin = torch.log(vol_bin)  # [B M]
    vol_out = 1 - vol_bin*W.N_bins # (x,y) volum for r> max_distance
    logV_out = torch.log(vol_out)  # [B M]
    #
    n_in = ss.n_inside.squeeze(-1)  # [B M]
    n_pts = ss.n_points.squeeze(-1)  # [B 1]
    n_out = n_pts - n_in # [B M]
    #
    w = W.score_weights().float()
    # w = W.c.postfix_cumsum().float()
    # W1 = torch.cat([w.view([1, 1, -1]) + logV_bin.unsqueeze(-1), W.score_out + logV_out.unsqueeze(-1)], -1) # [B M N+1]
    # logZ = torch.logsumexp(W1, dim=-1, keepdim=False) # [B, M]
    # scores = ss.counts @ w + n_out*W.score_out  - logZ*n_pts # [B, M]
    # scores = ss.counts @ w + n_out*W.score_out # - logZ*n_pts  # [B, M]
    #    
    logZ = W.logZ()
    scores = ss.counts @ (w - logZ) - n_in*logV_bin + n_out*(-logZ - logV_out) # [B, M]
    # scores = ss.counts @ (w - logZ) + n_out*(-logZ - math.log(100/10*200)) # [B, M] DEBUG: check against density
    return scores # [B, M]

#_______________________ residuals ________________________________________

def compute_residuals(data):
    with torch.no_grad():
        correspondences = data['correspondences']
        C = to_tensor(correspondences).cuda()  # [B, N, 4]
        x = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
        y = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
        models = to_tensor(data['models']).to(device='cuda', dtype=torch.float32)  # [B, M, 3, 3]        
        K1 = data['K1'].float().cuda() # [B, 3, 3] -- paired with x
        K2 = data['K2'].float().cuda() # [B, 3, 3] -- paired with y               
        if data['is_F'][0]:
            F = models
            assert('Are we sure?')
        else:
            # convert E to F
            K1I = K1.inverse()
            K2I = K2.inverse()
            F = torch.einsum('bij, bmik, bkl -> bmjl', K2I, models, K1I)  # K2^{-T} F K1^{-1}
            # unnormalize points
            x = torch.einsum('bij, bnj -> bni', K1, x) # (K1)x in the format [b,n,3]
            y = torch.einsum('bij, bnj -> bni', K2, y) # (K2)y in the format [b,n,3]            
        r = SampsonBM(y, x, F)
        # r = x.new_ones(F.shape[0:2] + x.shape[1:2]) # [B, M N]
        # logJJ = r_new.new_zeros(r_new.shape[0]) # [M]
        # models = F.cpu().numpy()
        r = torch.nan_to_num(r, nan=float('inf'))
        # residuals = r # DEBUG
        residuals = r.abs()
        data['residuals'] = residuals#, nan=float('inf'))


#_______________________ minimal solvers___________________________________

def solve_epipolar(x1, x2):
        """
        x1,x2: [B, 4, 2] numpy
        """
       
        m_batch_size = x1.shape[0]
        EE = []
        II = []
        for i in range(m_batch_size):
            E = poselib.essential_matrix_5pt(x1[i], x2[i])
            EE.extend(E)
            II.extend([i]*len(E)) # index i has created multiple solutions
        return EE, II

def new_minimal_models(data, m_batch_size, max_average_sol=None, include_GT=False):
    model.new_minimal_models(data, m_batch_size, max_average_sol=max_average_sol, include_GT=include_GT,solver=solve_epipolar,min_sample=5, validation_fn=None)

# _____________________ errors _____________________________________________
def pose_error_batch_torch(models, data):
    """
    models [B, M, 3, 3]
    return:
    err_e, err_R, err_t [B, M]
    """
    models = models.cuda().double()
    
    # Projection on E manifold is redundant
    # EE = models.view([-1,3,3])
    # U,s,Vh = torch.linalg.svd(EE) # [n,3,3]
    # s[:, -1] = 0
    # s[:, 0:2] = 1
    # S = torch.diag_embed(s)
    # EE = U @ S @ Vh
    # models = EE.view(models.shape)
    
    if False:
        """REFERENCE SOLUTION"""
        C = data['correspondences']
        data['correspondences']
        R1 = torch.zeros_like(models)
        R2 = torch.zeros_like(models)
        tt = R1[...,0].clone()
        for b in range(models.shape[0]):
            n = data['num_pts'][b]
            p1n = C[b, :n, :2]
            p2n = C[b, :n, 2:]
            for i in range(models.shape[1]):
                E = models[b,i]
                # OPenCV reference sol
                if True:
                    _, R, t1, _ = cv2.recoverPose(E.cpu().detach().numpy().astype(np.float64), p1n.cpu().numpy(), p2n.cpu().numpy())
                    R1[b, i] = torch.tensor(R)
                    R2[b, i] = torch.tensor(R)
                    tt[b, i] = torch.tensor(t1).squeeze(-1)
                else:
                    # cv_util reference sol
                    r1, r2, t = cv_utils.decompose_E(E)
                    R1[b,i] = torch.tensor(r1)
                    R2[b,i] = torch.tensor(r2)
                    tt[b,i] = torch.tensor(t).squeeze(-1)
        T = tt
    else:
        (R1, R2, T) = kornia.geometry.decompose_essential_matrix(models.double())
    #
    assert((torch.linalg.det(R1) > 0).all())
    assert((torch.linalg.det(R2) > 0).all())
    T = T.squeeze(-1)
    #
    gt_R = to_tensor(data['gt_R']).to(R1).unsqueeze(1)
    gt_T = to_tensor(data['gt_t']).to(T).unsqueeze(1).squeeze(-1)
    # shortcut the chairality check knowing the GT
    err_R = torch.min(R_error(R1, gt_R), R_error(R2, gt_R))
    err_t = torch.min(t_error(T, gt_T), t_error(-T, gt_T))
    err_e = torch.max(err_R, err_t)
    return err_e, err_R, err_t


def compose_essential_matrix(R, t):
    Tx = kornia.geometry.epipolar.cross_product_matrix(t)
    return Tx @ R
