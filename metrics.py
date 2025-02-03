import numpy as np
import torch
import math
from torch import Tensor
import kornia
import cv2
import cv_utils
from scipy.spatial.transform import Rotation as TR

from .functional import *

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
    return e_t


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
