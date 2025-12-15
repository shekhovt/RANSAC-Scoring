import numpy as np
import torch
import math
from torch import Tensor
import kornia
import cv2
import cv_utils
from scipy.spatial.transform import Rotation as TR

from .functional import *

# def R_error(R: Tensor, R_gt: Tensor) -> Tensor:
#     """
#     calculate angle between provided rotations
#     R [..., 3, 3]
#     R_gt [..., 3, 3]
#     Return: [...]
#     """
#     if False: 
#         """REFERENCE SOLUTION"""
#         s = list(R.shape)[:-2]
#         r = R.new_zeros(s)
#         for b in range(R.shape[0]):
#             R2 = R_gt[b][0]
#             for i, R1 in enumerate(R[b]):
#                 r[b, i] = np.linalg.norm(TR.from_matrix((R1 @ R2.T).cpu().numpy()).as_rotvec())
#                 # r[b,i] = torch.arccos(torch.max(torch.min((torch.trace(R1 @ R2.transpose(0, 1)) - 1) * 0.5, torch.tensor(1.0, device=R.device)), torch.tensor(-1.0, device=R.device)))
#         e_R = r / math.pi * 180  # [...]
#     else:
#         dR = torch.einsum('...ij, ...kj -> ...ik', R, R_gt)  # R R_gt^T
#         r = kornia.geometry.conversions.rotation_matrix_to_axis_angle(dR.view(-1, 3, 3)).view(dR.shape[:-1])  # [..., 3]
#         e_R = r.norm(dim=-1) / math.pi * 180  # [...]    
#         e_R = torch.nan_to_num(e_R, nan=float('180')) # for padded models
#     return e_R

# def t_error(T: Tensor, T_gt: Tensor) -> Tensor:
#     """
#     R [..., 3]
#     R_gt [..., 3]
#     Return: [...]
#     """
#     cos = (T*T_gt).sum(dim=-1)/(T.norm(dim=-1)*T_gt.norm(dim=-1))  # [*]
#     cos = cos.clip(min=-1, max=1)  # numerical fix
#     e_t = torch.acos(cos) / math.pi * 180
#     e_t = torch.nan_to_num(e_t, nan=float('180')) # for padded models
#     return e_t


# def auc_(data, threshold):
#     # return cv_utils.AUC(data, thresholds=[threshold])[0]
#     return cv_utils.pose_auc(data, thresholds=[threshold])[0]

# def AUC_10(data, axis=-1):
#     threshold = 10
#     r = []
#     if data.ndim == 2:
#         assert (axis == -1)
#         for i in range(data.shape[0]):
#             r += [auc_(data[i], threshold)]
#         r = np.array(r)
#     else:
#         assert (data.ndim == 1)
#         assert (axis == 0 or axis == -1)
#         r = auc_(data, threshold)
#     return r
#     # return cv_utils.AUC(data, thresholds=[10], binsize=50)
