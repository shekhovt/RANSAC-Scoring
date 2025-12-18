import torch

import model

__name__ = 'score_learn.model_H.py'
__package__ = 'score_learn'

import numpy as np

from .functional import *
import poselib
import kornia
import cv_utils
import cv2

from .decompose_homography import decompose_homography_mat, is_homography_ill_conditioned, validate_homography_cheirality
from .import model
from .model import normalize_points, unnormalize_points, R_error, t_error

#_______________________ Sampson error (batched/unbatched) _____________________________

def Sampson(x, y, H):
    """
    Computes Sampson error for a homography and a single correspondence.

    Args:
        H: (3,3) homography matrix (torch tensor)
        x: (2,) or (3,) source point (x1, x2 [,1])
        y: (2,) or (3,) destination point (y1, y2 [,1])

    Returns:
        scalar Sampson error (not squared)
    """

    # Normalize homogeneous coordinates to guard against arbitrary scale
    if x.shape[0] == 3:
        x = x / (x[2] + 1e-12)
    if y.shape[0] == 3:
        y = y / (y[2] + 1e-12)

    # Project x using H
    Hx = H @ x  # 3-vector
    u = Hx[0] / Hx[2]
    v = Hx[1] / Hx[2]

    # Residual (two constraint components)
    g = torch.stack([
        y[0] - u,
        y[1] - v
    ])  # shape (2,)

    # Jacobian wrt (y1,y2,x1,x2)
    # ∂g/∂y = identity
    J_y = torch.eye(2, dtype=H.dtype, device=H.device)

    # ∂u/∂x  and ∂v/∂x  (x is homogeneous)
    dHx_dx = H[:, :2]  # derivatives of Hx wrt x1,x2 (3x2)

    d_denom_dx = dHx_dx[2]  # d(Hx[2])/dx
    d_u_dx = (dHx_dx[0] * Hx[2] - Hx[0] * d_denom_dx) / (Hx[2] ** 2)
    d_v_dx = (dHx_dx[1] * Hx[2] - Hx[1] * d_denom_dx) / (Hx[2] ** 2)

    J_x = torch.stack([ -d_u_dx, -d_v_dx ], dim=0)  # (2,2)

    # Complete Jacobian wrt 4-vector (y1,y2,x1,x2)
    J = torch.cat([J_y, J_x], dim=1)  # shape (2,4)

    # Compute Sampson squared error
    JJt_inv = torch.inverse(J @ J.T)
    sampson_sq = g @ JJt_inv @ g

    return (sampson_sq + 1e-12)**0.5

import torch
import time


def SampsonBM(x, y, H, eps=1e-12):
    """
    Vectorized Sampson error for homographies.

    Args:
        H:  [B, M, 3, 3]   homography hypotheses
        x:  [B, n, 3]      source points (homogeneous)
        y:  [B, n, 3]      destination points (homogeneous)

    Returns:
        Sampson errors: [B, M, n]
    """

    B, M, _, _ = H.shape
    _, n, _ = x.shape

    # Normalize homogeneous coordinates to keep scale consistent
    x_norm = x / (x[..., 2:3] + eps)
    y_norm = y / (y[..., 2:3] + eps)

    # Expand x,y to match H:  [B, 1, n, 3] → [B, M, n, 3]
    x_exp = x_norm[:, None, :, :].expand(B, M, n, 3)
    y_exp = y_norm[:, None, :, :].expand(B, M, n, 3)

    # Homography projection: H @ x
    # H: [B, M, 3, 3], x_exp: [B, M, n, 3]
    # Reshape for batched matrix-vector multiplication
    # H[:,:,None,:,:] -> [B, M, 1, 3, 3], then broadcast to [B, M, n, 3, 3]
    # x_exp[..., None] -> [B, M, n, 3, 1]
    Hx = (H[:, :, None, :, :] @ x_exp.unsqueeze(-1)).squeeze(-1)  # [B, M, n, 3]

    # Normalize Hx → (u,v)
    u = Hx[..., 0] / Hx[..., 2]
    v = Hx[..., 1] / Hx[..., 2]

    # Residual g = [y1 - u, y2 - v]
    g = torch.stack([y_exp[..., 0] - u, 
                     y_exp[..., 1] - v], dim=-1)  # [B, M, n, 2]

    # === Jacobian wrt (y1,y2,x1,x2) ===

    # J_y = identity for y1,y2 part
    # Shape: [B,M,n,2,2]
    J_y = torch.eye(2, device=H.device, dtype=H.dtype).view(1,1,1,2,2)
    J_y = J_y.expand(B, M, n, 2, 2)

    # dHx/dx for x1,x2
    # Extract H[ :, :, :, :2 ] → [B,M,3,2]
    dHx_dx = H[..., :2]  # [B, M, 3, 2]

    # Repeat for n points
    dHx_dx = dHx_dx[:, :, None, :, :].expand(B, M, n, 3, 2)

    # Derivatives of denom Hx[2]
    d_denom = dHx_dx[..., 2, :]  # [B,M,n,2]

    # d(u)/dx_j and d(v)/dx_j
    Hx0 = Hx[..., 0]
    Hx1 = Hx[..., 1]
    Hx2 = Hx[..., 2]

    d_u_dx = (dHx_dx[..., 0, :] * Hx2.unsqueeze(-1)
              - Hx0.unsqueeze(-1) * d_denom) / (Hx2.unsqueeze(-1) ** 2 + eps)

    d_v_dx = (dHx_dx[..., 1, :] * Hx2.unsqueeze(-1)
              - Hx1.unsqueeze(-1) * d_denom) / (Hx2.unsqueeze(-1) ** 2 + eps)

    # J_x = - [ du/dx ; dv/dx ]  → [B,M,n,2,2]
    J_x = -torch.stack([d_u_dx, d_v_dx], dim=-2)

    # Full Jacobian: concat over y,x directions
    # [B,M,n,2,4]
    J = torch.cat([J_y, J_x], dim=-1)

    # Compute JJ^T  [B,M,n,2,2]
    JJt = J @ J.transpose(-1, -2)

    # Solve (JJt) * z = g  for z
    # g is [B,M,n,2], treat as column
    g_vec = g.unsqueeze(-1)

    # z = (JJt)^{-1} g  but using linear solve
    z = torch.linalg.solve(JJt + eps * torch.eye(2, device=H.device), g_vec)

    # Sampson squared error = g^T * z
    sampson_sq = (g_vec.transpose(-1, -2) @ z).squeeze(-1).squeeze(-1)

    return (sampson_sq + eps)**0.5  # [B, M, n]


def SymmetricReprojectionError(x, y, H, eps=1e-12):
    """
    Vectorized symmetric reprojection error for homographies.
    
    Computes: sqrt(||y - H*x||^2 + ||x - H^{-1}*y||^2)
    
    This is the standard symmetric transfer error used for homography estimation.
    
    Args:
        H:  [B, M, 3, 3]   homography hypotheses
        x:  [B, n, 3]      source points (homogeneous)
        y:  [B, n, 3]      destination points (homogeneous)

    Returns:
        Symmetric reprojection errors: [B, M, n]
    """
    
    B, M, _, _ = H.shape
    _, n, _ = x.shape
    
    # Expand x,y to match H:  [B, 1, n, 3] → [B, M, n, 3]
    x_exp = x[:, None, :, :].expand(B, M, n, 3)
    y_exp = y[:, None, :, :].expand(B, M, n, 3)
    
    # Normalize to inhomogeneous coordinates
    x_norm = x_exp / (x_exp[..., 2:3] + eps)
    y_norm = y_exp / (y_exp[..., 2:3] + eps)
    
    # Forward: y_proj = H * x
    # H: [B, M, 3, 3], x_exp: [B, M, n, 3]
    Hx = (H[:, :, None, :, :] @ x_exp.unsqueeze(-1)).squeeze(-1)  # [B, M, n, 3]
    y_proj = Hx / (Hx[..., 2:3] + eps)  # Normalize to inhomogeneous
    
    # Forward error: ||y - H*x||^2
    forward_error_sq = ((y_norm[..., 0] - y_proj[..., 0])**2 + 
                        (y_norm[..., 1] - y_proj[..., 1])**2)
    
    # Backward: x_proj = H^{-1} * y
    # Compute H^{-1} for all homographies
    try:
        H_inv = torch.linalg.inv(H)  # [B, M, 3, 3]
    except:
        # Handle singular matrices
        H_inv = torch.linalg.pinv(H)  # Use pseudo-inverse as fallback
    
    Hinv_y = (H_inv[:, :, None, :, :] @ y_exp.unsqueeze(-1)).squeeze(-1)  # [B, M, n, 3]
    x_proj = Hinv_y / (Hinv_y[..., 2:3] + eps)  # Normalize to inhomogeneous
    
    # Backward error: ||x - H^{-1}*y||^2
    backward_error_sq = ((x_norm[..., 0] - x_proj[..., 0])**2 + 
                         (x_norm[..., 1] - x_proj[..., 1])**2)
    
    # Symmetric error: sqrt of sum of squared errors
    symmetric_error = torch.sqrt(forward_error_sq + backward_error_sq + eps)
    
    return symmetric_error  # [B, M, n]


#_______________________ minimal solvers___________________________________

def solve_homography(x1, x2):
        """
        x1,x2: [B, 4, 2] numpy
        """
        m_batch_size = x1.shape[0]
        HH = []
        II = []
        for i in range(m_batch_size):
            H = poselib.homography_4pt(x1[i], x2[i], check_cheirality = True) # Does 4pt always has at most one solution?
            if H is not None:
                HH.extend([H])
                II.extend([i])
        return HH, II

def validate_homography(E, mx, my, iii):
    """
    E [N, 3 x 3] -- models
    mx, my [n, 3] -- points
    iii -- [N, 4] -- minimal samples corresponding to the models
    """
    valid, E, _ = validate_homography_cheirality(E, mx.double(), my.double(), iii)
    # check ill-conditioning
    ill, _ = is_homography_ill_conditioned(E, threshold = 100)
    mask = ~ill & valid
    return mask


# def new_minimal_models(data, m_batch_size, max_average_sol=None, include_GT=False, solver = solve_homography, min_sample=4):
#     """
#     Create new minimal models using minimal solver
#     Correspondences:
#     x [B, max_N, 3]
#     y [B, max_N, 3] 
#     """
#     C = data['correspondences'] # [B, max_N, 4]
#     xx = C[..., :3]
#     xx = xx / xx[..., 2:3]
#     yy = C[..., 3:]
#     yy = yy / yy[..., 2:3]
#     # xx = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
#     # yy = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
#     n_points = data['num_pts']
#     max_models = 0
#     models = [[] for b in range(C.shape[0])]
#     for b in range(C.shape[0]):
#         n = n_points[b]
#         mx = xx[b,:n]
#         my = yy[b,:n]
#         # log_p = (C.new_ones((n,))/n).log() #uniform sampling
#         # sampling probabilities based on SNN ratio, small SNN is better -> more likely to draw
#         snn = data['snn'][b,:n]
#         # what is the sigma of Normal distribution such that 95% of values are within [0,0.95]?
#         sigma = 1/8.0
#         p = torch.exp(-snn**2/(2*sigma**2)) + 1e-6
#         # p = snn < 0.7 + 0.001 # hard thresholding
#         log_p = (p / p.sum()).log() # convert to categorical distribution over all points and take log
#         EE = []
#         II = []
#         start_time = time.time()
#         max_iters = 10
#         iter = 0
#         if False:
#             mask = snn < 0.7
#             mx = mx[mask]
#             my = my[mask]
#             n = mx.shape[0]
#             log_p = (C.new_ones((n,))/n).log() #uniform sampling
        
#         while len(EE) < m_batch_size and iter < max_iters and n >= min_sample:
#             ii = sample_visible_subsets(min_sample, log_p, m_batch_size, mx, my) # [M, min_sample]
#             x1 = mx[ii,:].cpu().numpy().astype(float) # [M, min_sample, 3] coordinates at minimal sample
#             y1 = my[ii,:].cpu().numpy().astype(float) # [M, min_sample, 3]
#             E, I = solver(x1,y1) # E: list of [3,3] numpy, I: list of indices in ii
#             E = torch.tensor(np.array(E))
#             I = torch.tensor(np.array(I))
#             # check cheirality of the points we used to estimate the models
#             if E.shape[0] == 0:
#                 iter += 1
#                 continue
#             valid, E, _ = validate_homography_cheirality(E, mx.double(), my.double(), ii[I])
#             # check ill-conditioning
#             ill, _ = is_homography_ill_conditioned(E, threshold = 100)
#             mask = ~ill & valid
#             EE.extend(E[mask].tolist())
#             II.extend(I[mask].tolist())
#             iter += 1
#         # print(f"Loop for batch {b} took {elapsed_time:.4f} seconds")

#         # get the required number of valid models
#         if len(EE) > m_batch_size:
#             EE = EE[:m_batch_size]
#             II = II[:m_batch_size]

#         if len(EE) < m_batch_size: # this is to prevent errors in decomposition, etc. for the whole batch
#             num_missing = m_batch_size - len(EE)
#             EE.extend([torch.eye(3, 3, dtype=torch.float32)] * num_missing)
#             II.extend([0] * num_missing)

#         if False and b==0: # DEBUG TEST
#             print('Indices of first model:', II[0])
#             x1 = mx[ii[II[0]]].cpu()
#             y1 = my[ii[II[0]]].cpu()
#             M = torch.tensor(EE[0], dtype = x1.dtype)
#             reprojected_y1 = (M @ x1.T).T
#             reprojected_y1 = reprojected_y1 / reprojected_y1[:,2:3]
#             err = (reprojected_y1 - y1).norm(dim=1)
#             print('Reprojection errors of first model (px):', err)
#             r = SampsonBM(x1[None,:, :], y1[None,:, :], M[None, None, :, :])[0,0]
#             print('Sampson errors of first model in normalized coordinates:', r)
#             K1 = data['K1'][b].cpu()
#             K2 = data['K2'][b].cpu()
#             x1 = unnormalize_points(x1, K1)
#             y1 = unnormalize_points(y1, K2)
#             M = unnormalize_models(M[None,:,:], K1, K2)[0]
#             reprojected_y1 = (M @ x1.T).T
#             reprojected_y1 = reprojected_y1 / reprojected_y1[:,2:3]
#             err_px = (reprojected_y1 - y1).norm(dim=1)
#             print('Reprojection errors of first model (px):', err_px)
#             r = SampsonBM(x1[None,:, :], y1[None,:, :], M[None, None, :, :])[0,0]
#             print('Sampson errors of first model in pixel coordinates:', r)
        
#         n_models = len(EE)
#         max_models = max(max_models, n_models)
#         models[b] = EE
#     # assert(max_models > 1000)
#     if max_average_sol is not None:
#         max_models = min(max_models, m_batch_size*max_average_sol)
#     for b in range(C.shape[0]):
#         if len(models[b]) > max_models:
#             models[b] = models[b][:max_models]
#         else:
#             models[b] = np.concatenate([np.stack(models[b]), np.zeros((max_models - len(models[b]), 3, 3))]) # pad models with zeros -- Ok for E but maybe not for H?
#     models = np.stack(models) # stack along batch dim
#     models = torch.tensor(models).to(dtype= torch.float32).cpu()
#     #
#     # data['models'][:,:1000] = models # use 1K models
#     if include_GT:
#         GTmodels = data['models'][:,-1:].to(models)
#         models = torch.cat([models, GTmodels], dim = 1)
#     data['models'] = models


def normalize_models(models, K1, K2):
    K2I = K2.inverse()
    M_px = torch.einsum('...ij, ...mjk, ...kl -> ...mil', K2I, models, K1)  # K2^{-1} F K1
    return M_px

def unnormalize_models(models, K1, K2):
    K1I = K1.inverse()
    M_px = torch.einsum('...ij, ...mjk, ...kl -> ...mil', K2, models, K1I)  # K2 F K1^{-1}
    return M_px

def unnormalize(data):
    correspondences = data['correspondences']
    C = to_tensor(correspondences).cuda()  # [B, N, 4]
    x = C[..., :3]
    y = C[..., 3:]
    models = to_tensor(data['models']).to(device='cuda', dtype=torch.float32)  # [B, M, 3, 3]        
    K1 = data['K1'].float().cuda() # [B, 3, 3] -- paired with x
    K2 = data['K2'].float().cuda() # [B, 3, 3] -- paired with y
    # unnormalize points
    x = unnormalize_points(x, K1)
    y = unnormalize_points(y, K2)           
    # unnormalize models
    M_px = unnormalize_models(models, K1, K2)
    return x, y, M_px

def compute_residuals(data):
    """
    assume data has normalized cooridnates
    models are in normalized coordinates
    computing residuals in pixel coordinates
    """
    with torch.no_grad():
        x, y, M = unnormalize(data)
        r = SampsonBM(x, y, M)
        # r = SymmetricReprojectionError(x, y, M) # DEBUG
        r = torch.nan_to_num(r, nan=float('inf'))
        residuals = r.abs()
        data['residuals'] = residuals



def new_minimal_models(data, m_batch_size, max_average_sol=None, include_GT=False):
    return model.new_minimal_models(data, m_batch_size, max_average_sol=max_average_sol, include_GT=include_GT, solver = solve_homography, min_sample=4, validation_fn=validate_homography)
    
# def normalized_homography(H, K1, K2):
#     normalizedHomography = np.linalg.inv(K2).dot(H1to2).dot(K1)


def pose_error_batch_torch(models, data, method = 'torch'):
    """
    Docstring for pose_error
    
    :param models: [B, M, 3, 3] homography matrices in normalized coordinates
    :param GT: [3 4] [R t] matrix
     # TODO: cheirality check instead of best combination with GT
    """
    # Decompose homographies into R and t using Kornia
    if method == 'torch':
        models = models.cuda()
    elif method == 'cv2':
        models = models.cpu().double()
    gt_R = to_tensor(data['gt_R']).to(models) # [B, 3, 3]
    gt_T = to_tensor(data['gt_t']).to(models) # [B, 3]

    B, M = models.shape[:2]
    Err_R = torch.zeros((B, M), dtype=models.dtype, device=models.device)
    Err_t = torch.zeros((B, M), dtype=models.dtype, device=models.device)
    Err_e = torch.zeros((B, M), dtype=models.dtype, device=models.device)
    # Decompose using torch
    if method == 'torch':
        BRs, Bts, Bnormals = decompose_homography_mat(models) # [B, M, num, 3, 3], [B, M, num, 3], [B, M, num, 3]
        num = 4
        for b in range(B):
            err_R_sm = torch.zeros((num,M), dtype=models.dtype, device=models.device)
            err_t_sm = torch.zeros((num,M), dtype=models.dtype, device=models.device)
            for s in range(num):
                err_R_sm[s] = R_error(to_tensor(BRs[b,:,s]), gt_R[b][None,:,:])
                err_t_sm[s] = t_error(to_tensor(Bts[b,:,s]), gt_T[b][None,:])
            Err_R[b] = err_R_sm.min(dim=0).values
            Err_t[b] = err_t_sm.min(dim=0).values
            Err_e[b] = torch.max(Err_R[b], Err_t[b])
        Err_e = Err_R # DEBUG
        return Err_e, Err_R, Err_t

    elif method == 'cv2':
        # models: [B, M, 3, 3]
        Err_R = torch.zeros((B, M), dtype=models.dtype, device=models.device)
        Err_t = torch.zeros((B, M), dtype=models.dtype, device=models.device)
        Err_e = torch.zeros((B, M), dtype=models.dtype, device=models.device)
        for b in range(B):
            for m in range(M):

                num, Rs, ts, normals = cv2.decomposeHomographyMat(models[b,m].cpu().numpy(), np.identity(3))
                err_R = 180.0
                err_t = 180.0
                for s in range(num):
                    e_R = R_error(to_tensor(Rs[s]), gt_R[b]).item()
                    e_t = t_error(to_tensor(ts[s].flatten()), gt_T[b]).item()
                    err_R = min(err_R, e_R)
                    err_t = min(err_t, e_t)

                if False: # DEBUG of decompose_homography_mat (more convenient in this loop):
                    Rs1, ts1, normals1 = decompose_homography_mat(models[b,m].cpu().double())
                    num = 4
                    err_R1 = 180.0
                    err_t1 = 180.0
                    for s in range(num):
                        e_R1 = R_error(to_tensor(Rs1[s]), gt_R[b]).item()
                        e_t1 = t_error(to_tensor(ts1[s].flatten()), gt_T[b]).item()
                        err_R1 = min(err_R1, e_R1)
                        err_t1 = min(err_t1, e_t1)
                    diff_R = abs(err_R - err_R1)
                    diff_t = abs(err_t - err_t1)
                    if diff_R > 1e-4 or diff_t > 1e-4:
                        print(f"Large difference between cv2 and torch decomposition for batch {b} model {m}: err_R diff = {diff_R}, err_t diff = {diff_t}")
                        print('homography candidate:', models[b,m].cpu().numpy())
                        print('GT:', gt_R[b].cpu().numpy())
                        f,c = is_homography_ill_conditioned(models[b,m].cpu().double(), threshold = 100)
                        print(f'Condition number: {c}')
                        print("---")

                err_e = max(err_R, err_t)
                Err_R[b,m] = err_R
                Err_t[b,m] = err_t
                Err_e[b,m] = err_e
    Err_e = Err_R # DEBUG
    return Err_e, Err_R, Err_t

