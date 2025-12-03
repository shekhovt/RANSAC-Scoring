import torch

import numpy as np

from .functional import *
from .metrics import R_error, t_error
import poselib
import kornia
import cv_utils
import cv2

from .decompose_homography import decompose_homography_mat, decompose_homography_stable, is_decomposition_numerically_unstable, is_homography_ill_conditioned, decompose_homography_robust, validate_homography_cheirality

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

    # Expand x,y to match H:  [B, 1, n, 3] → [B, M, n, 3]
    x_exp = x[:, None, :, :].expand(B, M, n, 3)
    
    # Normalize y to get inhomogeneous coordinates
    y = y / y[..., 2:3]
    # Expand y:  [B, 1, n, 3] → [B, M, n, 3]
    y_exp = y[:, None, :, :].expand(B, M, n, 3)

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
              - Hx0.unsqueeze(-1) * d_denom) / (Hx2.unsqueeze(-1) ** 2)

    d_v_dx = (dHx_dx[..., 1, :] * Hx2.unsqueeze(-1)
              - Hx1.unsqueeze(-1) * d_denom) / (Hx2.unsqueeze(-1) ** 2)

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


def new_minimal_models(data, m_batch_size, max_average_sol=None, include_GT=False, solver = solve_homography, min_sample=4):
    """
    Create new minimal models using minimal solver
    Correspondences:
    x [B, max_N, 3]
    y [B, max_N, 3] 
    """
    C = data['correspondences'] # [B, max_N, 4]
    xx = C[..., :3]
    xx = xx / xx[..., 2:3]
    yy = C[..., 3:]
    yy = yy / yy[..., 2:3]
    # xx = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
    # yy = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
    n_points = data['num_pts']
    max_models = 0
    models = [[] for b in range(C.shape[0])]
    for b in range(C.shape[0]):
        n = n_points[b]
        log_p = (C.new_ones((n,))/n).log()
        EE = []
        II = []
        
        start_time = time.time()
        while len(EE) < m_batch_size:
            ii = sample_visible_subsets(min_sample, log_p, m_batch_size, xx[b,:n,:2], yy[b,:n,:2]) # [M, min_sample]
            x1 = xx[b,:n][ii,:].cpu().numpy().astype(float)
            y1 = yy[b,:n][ii, :].cpu().numpy().astype(float)
            E, I = solver(x1,y1)
            E = torch.tensor(E)
            I = torch.tensor(I)
            valid, E = validate_homography_cheirality(E, xx[b,:n,:2], yy[b,:n,:2], threshold = 0.01)
            ill, _ = is_homography_ill_conditioned(E, threshold = 100)
            mask = ~ill & valid
            EE.extend(E[mask].tolist())
            II.extend(I[mask].tolist())
        # print(f"Loop for batch {b} took {elapsed_time:.4f} seconds")

        # get the required number of valid models
        EE = EE[:m_batch_size]
        II = II[:m_batch_size]

        if False and b==0: # DEBUG TEST
            print('Indices of first model:', II[0])
            x1 = xx[b,:n][ii[II[0]]].cpu()
            y1 = yy[b,:n][ii[II[0]]].cpu()
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


def normalize_models(models, K1, K2):
    K2I = K2.inverse()
    M_px = torch.einsum('...ij, ...mjk, ...kl -> ...mil', K2I, models, K1)  # K2^{-1} F K1
    return M_px

def unnormalize_models(models, K1, K2):
    K1I = K1.inverse()
    M_px = torch.einsum('...ij, ...mjk, ...kl -> ...mil', K2, models, K1I)  # K2 F K1^{-1}
    return M_px

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
        r = torch.nan_to_num(r, nan=float('inf'))
        residuals = r.abs()
        data['residuals'] = residuals



def new_minimal_models_H(data, m_batch_size, max_average_sol=None, include_GT=False):
    return new_minimal_models(data, m_batch_size, max_average_sol=max_average_sol, include_GT=include_GT, solver = solve_homography, min_sample=4)

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
    # Decompose using torch
    if method == 'torch':
        BRs, Bts, Bnormals, _ = decompose_homography_robust(models, s3_threshold = 0.01) # [B, M, num, 3, 3], [B, M, num, 3], [B, M, num, 3]
        num = 4
        for b in range(B):
            err_R_sm = torch.zeros((num,M), dtype=models.dtype, device=models.device)
            for s in range(num):
                err_R_sm[s] = R_error(to_tensor(BRs[b,:,s]), gt_R[b][None,:,:])
            Err_R[b] = err_R_sm.min(dim=0).values
            if False and b == 0: # DEBUG
                Err_R_ref = torch.zeros((M,), dtype=models.dtype, device=models.device)
                for m in range(100):
                    num, Rs, ts, normals = cv2.decomposeHomographyMat(models[b,m].cpu().numpy(), np.identity(3))
                    Rs1, ts1, normals1 = decompose_homography_mat(models[b,m].cpu().double())
                    err_R = 180.0
                    err_t = 180.0
                    for s in range(num):
                        e_R = R_error(to_tensor(Rs[s]), gt_R[b].cpu().double()).item()
                        e_t = t_error(to_tensor(ts[s].flatten()), gt_T[b].cpu().double()).item()
                        err_R = min(err_R, e_R)
                        err_t = min(err_t, e_t)
                    Err_R_ref[m] = err_R
                print('Max difference in Err_R between torch and cv2 for first 100 models:', (Err_R[b,:100] - Err_R_ref[:100]).abs().max().item())
                print('Mean difference in Err_R between torch and cv2 for first 100 models:', (Err_R[b,:100] - Err_R_ref[:100]).abs().mean().item())
                print('Std difference in Err_R between torch and cv2 for first 100 models:', (Err_R[b,:100] - Err_R_ref[:100]).abs().std().item())

        # Err_t = R_error(to_tensor(BRs[:,0]), gt_R)
        return Err_R, Err_R, Err_R # DEBUG

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

                if True: # DEBUG:
                    Rs1, ts1, normals1, _ = decompose_homography_robust(models[b,m].cpu().double())
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
                        _, s3 = is_decomposition_numerically_unstable(models[b,m].cpu().double())
                        print(f'Condition number: {c} s3={s3}')
                        print("---")

                err_e = max(err_R, err_t)
                Err_R[b,m] = err_R
                Err_t[b,m] = err_t
                Err_e[b,m] = err_e
    Err_e = Err_R # DEBUG
    return Err_e, Err_R, Err_t


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
