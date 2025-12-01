# translation rotation
# %%
import os, sys
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
# import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import math
import time
import shlex
from argparse import ArgumentParser
import warnings
import itertools
import poselib
from scipy.spatial.transform import Rotation as TR
#!%matplotlib inline

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from types import SimpleNamespace

from .score_weights import *
from .load_data import *
from .metrics import * #ARO_candidates, pose_error_EE, pose_error_EERT
from .aro import * #ARO_candidates, pose_error_EE, pose_error_EERT
from . import local_optimization as LO
from .drawing import *


op = ArgumentParser()
op.add_argument("--batch_size", type=int, default=8, help="number of pairs processed in parallel")
op.add_argument("--val_samples", type=int, default=1000, help="number of minimal samples used for validation")
op.add_argument("--val_pairs", type=int, default=1000, help="number of image pairings per scene used for validation")
op.add_argument("--val_thresholds", type=int, default=200, help="grid of thresholds subdividing [0.1 10] for validation")
op.add_argument("--N_bins", type=int, default=500, help="histogram size for residuals")
op.add_argument("--max_distance", type=float, default=10.0, help="max distance for the histogram")
op.add_argument("--polish", default=0, type=str, help="0 - no polish, 1 - BA, 2 - LMeDs, 3 - ARO, 4 - ARO + BA, 5 - LMeds + BA, GaU - our polish (EM-LMA GaU) , MSAC - our polish (EM-LMA MSAC)")
# op.add_argument("--data", type=str, default='PhotoTourismSPSG', help ="dataset")
op.add_argument("--data", type=str, default='PhotoTourismRootSIFT', help="dataset")
# op.add_argument("--data", type=str, default='KITTI', help="dataset")
op.add_argument("-V", "--validate", action='store_true', default=False, help="recompute validation")
op.add_argument("-R", "--recompute", action='store_true', default=False, help="recompute test")
op.add_argument("--running", action='store_true', default=False, help="running test")
op.add_argument("--kde", action='store_true', default=False, help="running test")
op.add_argument("--geom", action='store_true', default=False, help="error geometry analysis")
op.add_argument("--inliers", action='store_true', default=False, help="inliers statistics experiment")
op.add_argument("--static", action='store_true', default=False, help="static 1K test, DEPRICATED")
op.add_argument("--var", type=bool, default=True, help="variance test")
op.add_argument("--F", action='store_true', default=False, help="use solver for F matrix (regardless of the dataset)")
op.add_argument("--new_models", type=bool, default=True, help="sample new models for validation instead of saved ones")

args_str = ' '.join(sys.argv[1:])
ops, args = op.parse_known_args(shlex.split(args_str))
o = SimpleNamespace(**vars(ops))
## 
## For interactove model
o.validate = True
# o.F = True
# o.geom = True
# o.inliers = True
# o.R = True
# o.var = True
o.var = False
##
##

print(o)
batch_size = o.batch_size
try:
    o.polish = int(o.polish)
except:
    pass
polish = o.polish
if polish == 0:
    polish_func = None
elif polish == 1:
    polish_func = best_BA
elif polish == 2:
    polish_func = best_LMEDS
elif polish == 3:
    polish_func = best_ARO
elif polish == 4:
    polish_func = best_ARO_BA
elif not (polish == 'GaU' or polish == 'MSAC'):
    print("undefined polishing method!")

for dataset_info in datasets:
    if dataset_info.name == o.data:
        break
if dataset_info.name != o.data:
    print(f'cannot find dataset {o.data}')
    exit(1)

o.F = (dataset_info.Fundamental or o.F)

Eval_GCMAGSAC = False

# val_scenes = dataset_info.val
# val_scenes = dataset_info.test
val_scenes = dataset_info.val[3:4]
# test_scenes = dataset_info.test #[7:8]
test_scenes = dataset_info.test
res_root = f'results/{dataset_info.name}/'
results_file0 = res_root + f'polish={polish}/' + f'test_results.pkl'
model_path = res_root + 'models/'
if Eval_GCMAGSAC:
    res_root += 'GCMAGSAC/'
results_path = res_root + f'polish={polish}/'
val_file = res_root + f'val_results.pkl'

def set_results_paths(folder = ''):
    global results_file
    global results_running_file
    global table_file
    global table_out
    if folder != '':
        folder += '/'
    results_file = results_path + f'{folder}test_results.pkl'
    results_running_file = results_path + f'{folder}test_results_running.pkl'
    table_file = results_path + f'{folder}test_table.pkl'
    table_out = results_path + f'{folder}test_table.txt'

# set_results_paths(test_scenes[0])
set_results_paths()
# results_file = f'results/PhotoTourism(0-2)_polish={polish}.pkl'

# %%
def best_score(ss, M):
    scores = ss.counts @ M  # [B M K] @ [K T] -> [B M T]
    best_s, best_idx = scores.max(dim=1)  # [B T] / [B]
    best_s = best_s.cpu()
    best_idx = best_idx.cpu()
    return best_s, best_idx  # [B T] / [B]


def local_optimization_ours(data, models, W):
    """
    data - dict with batched data:
        correspondences [B, N, 4]
        K1,
        K2
    models [B, 3, 3] -- current best models
    W - ScoreWeights with functions score_residuals and IRLS_weights
    return:
     rmodels [B, 3, 3] -- optimized best models
    """
    x, y = normalized_points(data)
    KX = data['K1'].to(x)
    KY = data['K2'].to(x)
    models = models.to(x)
    def score_f(rr):
        s = W.score_residuals(rr, reduction=None)
        return s
    def weight_f(rr):
        s = W.IRLS_weight(rr)
        return s
    E1 = LO.local_optimization(x, y, KX, KY, LO.E_parameterization(models),score_f, weight_f, iterations = 25)
    return E1

def local_optimization_PoseLib(data, models, loss_type):
    """
    models [B, 3, 3]
    return:
     rmodels [B, 3, 3] -- optimized best models
    """
    def camera_dict_from_matrix(K):
        return {'model': 'SIMPLE_PINHOLE', 'width': int(K[0, 2]*2), 'height': int(K[1, 2]*2), 'params': [float((K[0, 0] + K[1, 1])/2), float(K[0, 2]), float(K[1, 2])]}
    
    models = copy.deepcopy(models)
    
    bo = {'max_iterations': 25, 'loss_type': loss_type, 'loss_scale': WMSAC.tau}
    X1, X2 = unnormalized_points(data)
    X1 = X1.cpu().double().numpy()
    X2 = X2.cpu().double().numpy()
    R1, R2, t1 = kornia.geometry.epipolar.decompose_essential_matrix(models)
    for b in range(models.shape[0]):
        K1 = data['K1'][b].double().numpy()
        K2 = data['K2'][b].double().numpy()
        C1 = camera_dict_from_matrix(K1)
        C2 = camera_dict_from_matrix(K2)
        P = poselib.CameraPose()
        P.R = R1[b]
        P.t = t1[b]
        # check can recompose
        # E = LO.compose_essential_matrix(torch.tensor(P.R), torch.tensor(P.t))
        n = data['num_pts'][b]
        x1 = X1[b,:n,:2]
        x2 = X2[b,:n,:2]
        P,B = poselib.refine_relative_pose(x1, x2, P, C1, C2, bo)
        E = LO.compose_essential_matrix(torch.tensor(P.R), torch.tensor(P.t))
        models[b] = E
    return models
    

def local_optimization(data, models, polish):
    if polish == 'GaU':
        return local_optimization_ours(data, models, WGU)
    elif polish == 'MSAC':
        return local_optimization_ours(data, models, WMSAC)
    elif polish in ['TRIVIAL', 'TRUNCATED', 'HUBER', 'CAUCHY', 'TRUNCATED_LE_ZACH']:
        return local_optimization_PoseLib(data, models, polish), 0
    else:
        raise AttributeError(f'Polish method {polish} unrecognized')


def new_minimal_models(data, m_batch_size, max_average_sol=None, include_GT=False):
    C = data['correspondences'] # [B, max_N, 4]
    xx = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
    yy = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
    n_points = data['num_pts']
    max_models = 0
    models = [[] for b in range(C.shape[0])]
    for b in range(C.shape[0]):
        n = n_points[b]
        log_p = (C.new_ones((n,))/n).log()
        ii = sample_subsets(5, log_p, m_batch_size)
        x1 = xx[b,:n][ii,:].cpu().numpy().astype(float)
        y1 = yy[b,:n][ii, :].cpu().numpy().astype(float)
        EE = solve_epipolar(x1,y1)
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
            models[b] = np.concatenate([np.stack(models[b]), np.zeros((max_models - len(models[b]), 3, 3))])
    models = np.stack(models) # stack along batch dim
    models = torch.tensor(models).to(dtype= torch.float32).cpu()
    #
    # data['models'][:,:1000] = models # use 1K models
    if include_GT:
        GTmodels = data['models'][:,-1:].to(models)
        models = torch.cat([models, GTmodels], dim = 1)
    data['models'] = models
    # errors, _, _ = pose_error_batch_torch(data['models'][:,:-1], data)
    # data['errors'][:,:1000] = errors


def new_minimal_models_F(data, m_batch_size, max_average_sol=None, include_GT=False):
    # C = data['correspondences'].double() # [B, max_N, 4]
    # X1 = C[..., :2] # normalized points
    # X2 = C[..., 2:]
    X1, X2 = unnormalized_points(data)
    X1 = X1[:,:,:2].double() # [B, max_N, 3] chop off homogenous 1
    X2 = X2[:,:,:2].double()
    B = X1.shape[0]
    n_points = data['num_pts']
    max_models = 0
    models = [[] for b in range(B)]
    for b in range(B):
        K1 = data['K1'][b]
        K2 = data['K2'][b]
        n = n_points[b]
        if n>=7:
            log_p = (X1.new_ones((n,))/n).log()
            ii = sample_subsets(7, log_p, m_batch_size)
            x1 = X1[b,:n][ii,:] # [mbatch, 7, 2]
            y1 = X2[b,:n][ii,:]
            FF = []
            for i in range(x1.shape[0]):
                F = cv2.findFundamentalMat(x1[i].cpu().numpy(),y1[i].cpu().numpy(),cv2.FM_7POINT)[0]
                if F is not None and isinstance(F,np.ndarray) and F.shape[0] > 0:
                    FF += [torch.tensor(F).view([-1,3,3])]
            FF = torch.cat(FF, dim = 0).cuda()
            # FF = kornia.geometry.epipolar.find_fundamental(x1, y1, weights=None, method='7POINT')
            FF = FF.view([-1,3,3])
            EE = torch.einsum('ij, mik, kl -> mjl', K2.to(FF), FF, K1.to(FF))  # K2^{T} F K1
            # EE = FF
        else:
            EE = torch.zeros((0,3,3)).float().cuda()
        #
        n_models = len(EE)
        max_models = max(max_models, n_models)
        models[b] = EE.cpu().numpy()
    # max_models = m_batch_size*3
    if max_average_sol is not None:
        max_models = min(max_models, m_batch_size*max_average_sol)
    for b in range(B):
        if len(models[b]) > max_models:
            models[b] = models[b][:max_models]
        else:
            models[b] = np.concatenate([models[b], np.zeros((max_models - len(models[b]), 3, 3))])
    models = np.stack(models) # stack along batch dim
    models = torch.tensor(models).to(dtype= torch.float32).cpu()
    if include_GT:
        GTmodels = data['models'][:,-1:].to(models)
        models = torch.cat([models, GTmodels], dim = 1)
    data['models'] = models
    data['is_F'][:] = False


def decompose_SVDZ(EE):
    U,s,Vh = torch.linalg.svd(EE) # [n,3,3]
    s[-1] = 0
    s[0:2] = 1
    S = torch.diag_embed(s)
    EE = U @ S @ Vh
    if torch.linalg.det(U) < 0:
        U[:,-1] = - U[:,-1]
    if torch.linalg.det(Vh) < 0:
        Vh[-1,:] = - Vh[-1,:]
    assert (torch.linalg.det(U) > 0 )
    assert (torch.linalg.det(Vh) > 0 )
    # assert ((EE - U @ S @ Vh).abs().max() < 1e-3)
    #
    def decompose_Z(U, transpose = False):
        if transpose:
            U = U.T
        q = TR.from_matrix(U).as_quat()
        gamma = math.atan2(q[-2], q[-1])
        theta = 2 * gamma
        Rz = TR.from_euler('z', theta).as_matrix()
        # qz = TR.from_euler('z', theta).as_quat()
        Uz = U @ Rz.T
        alpha = TR.from_matrix(Uz).magnitude()
        assert(TR.from_matrix(Uz).magnitude() <= TR.from_matrix(U).magnitude())
        if transpose:
            return Rz.T, Uz.T, alpha
        else:
            return Uz, Rz, alpha
    if False:
        def decompose_Z(U):
            a = TR.from_matrix(U.numpy()).as_euler('zxy')
            R1 = TR.from_euler('xy', a[1:]).as_matrix()
            R2 = TR.from_euler('z', a[0]).as_matrix()
            U1 = R1 @ R2
            assert ((U - U1).abs().max() < 1e-3)
            return R1, R2
    a_min = 1000
    for i in range(3):
        if i == 0:
            d = [1,1,1]
        elif i==1:
            d = [-1,1,-1]
        else:
            d = [1,-1,-1]
        D = np.diag(d)
        R1, R2, a1 = decompose_Z(U.numpy() @ D )
        R3, R4, a2 = decompose_Z(D @ Vh.numpy(), transpose=True)
        a = abs(a1) + abs(a2)
        if a < a_min:
            a_min= a
            S_m = R2 @ S.numpy() @ R3
            R1_m = R1
            R4_m = R4
            # a = TR.from_matrix(Vh.numpy()).as_euler('xyz')
            # R3 = TR.from_euler('z', a[2]).as_matrix()
            # R4 = TR.from_euler('xy', a[0:2]).as_matrix()
        # Vh1 = R3 @ R4
        # assert ((Vh - Vh1).abs().max() < 1e-3)
        # recompose
        # S1 = R2 @ S.numpy() @ R3
    # assert ((EE - R1 @ S1 @ R4).abs().max() < 1e-3)
    M  = R1_m @ S_m @ R4_m
    assert ((EE - M).abs().max() < 1e-3)
    return R1_m, S_m, R4_m

def random_rotation(n, deg, sampling = 'uniform', t = None):
    if deg is not None:
        assert(n == len(deg))
    d = torch.normal(torch.zeros(n, 3)) # [n, 3]
    if t is not None: # project orthogonal to t
        tn = F.normalize(t, dim=-1).unsqueeze(0)
        d = d - (d*tn).sum(dim=-1, keepdim=True)*tn
    d = F.normalize(d, dim=-1) # approx uniform direction vector
            
    if sampling == 'uniform':
        theta = (torch.rand(n, 1) - 0.5)*deg/180*math.pi # amount of rotation in radians
    elif sampling == None or sampling == 'bernoulli': 
        if len(deg) > 1:
            theta = (deg/180*math.pi).view([-1, 1])
        else:
            theta = torch.ones(n, 1)*deg/180*math.pi
    # l = torch.tan(theta / 4)
    l = theta
    d = d * l
    # R = TR.from_mrp(d.numpy()).as_matrix()
    R = TR.from_rotvec(d.numpy()).as_matrix()
    return torch.tensor(R)


def new_perturbed_models(data, n=1000, deg = 10):
    C = data['correspondences'] # [B, max_N, 4]
    models = [[] for b in range(C.shape[0])]
    for b in range(C.shape[0]):
        # GTE = data['models'][b,-1] # GT model
        # (R1, R2, T) = kornia.geometry.decompose_essential_matrix(GTE)
        # R1 = R1.squeeze(dim=0)
        # T = T.squeeze(dim=-1)
        gt_R = to_tensor(data['gt_R'])[b]
        gt_T = to_tensor(data['gt_t'])[b].squeeze(-1)
        dR = random_rotation(n, deg)
        Rp = dR @ gt_R
        dR = random_rotation(n, deg)
        Tp = dR @ gt_T
        E = LO.compose_essential_matrix(torch.tensor(Rp), torch.tensor(Tp))
        models[b] = E
        # (R1, R2, T) = kornia.geometry.decompose_essential_matrix(E)
        # T = T.squeeze(-1)
        # err_R = torch.min(R_error(R1, gt_R), R_error(R2, gt_R))
        # err_t = torch.min(t_error(T, gt_T), t_error(-T, gt_T))
        pass
    models = np.stack(models) # stack along batch dim
    models = torch.tensor(models)
    # data['models'][:,:n] = models # replace first n models
    data['models'] = torch.cat([data['models'][:,:-1], models, data['models'][:,-1:]], dim = 1 ).cuda()


def augment_models(data, ref_models, n=1000, deg = 10):
    B = ref_models.shape[0]
    models = [[] for b in range(B)]
    for b in range(B):
        GTE = ref_models[b].double() # GT model
        (R1, R2, T) = kornia.geometry.decompose_essential_matrix(GTE)
        gt_R = R1.squeeze(dim=0)
        gt_T = T.squeeze(dim=-1)
        # gt_R = to_tensor(data['gt_R'])[b]
        # gt_T = to_tensor(data['gt_t'])[b].squeeze(-1)
        tp = 'uniform'
        # tp = None
        dR = random_rotation(n, deg, tp)
        Rp = dR @ gt_R
        dR = random_rotation(n, deg, tp)
        Tp = dR @ gt_T
        E = LO.compose_essential_matrix(torch.tensor(Rp), torch.tensor(Tp))
        models[b] = E
        # (R1, R2, T) = kornia.geometry.decompose_essential_matrix(E)
        # T = T.squeeze(-1)
        # err_R = torch.min(R_error(R1, gt_R), R_error(R2, gt_R))
        # err_t = torch.min(t_error(T, gt_T), t_error(-T, gt_T))
        pass
    models = np.stack(models) # stack along batch dim
    models = torch.tensor(models)
    # data['models'][:,:n] = models # replace first n models
    data['models'] = torch.cat([data['models'][:,:-1], models, data['models'][:,-1:]], dim = 1 ).cuda()

def new_errors(data):
    if data['is_F'][0]:
        raise NotImplementedError('Need to convert F to E for the erorr metric first')
    errors, errors_r, errors_t = pose_error_batch_torch(data['models'], data)
    data['errors'] = errors.cpu()
    data['errors_r'] = errors_r.cpu()
    data['errors_t'] = errors_t.cpu()
    return errors, errors_r, errors_t
    
    
# kde_xval = np.logspace(np.log10(0.1), np.log10(20), 20)
kde_xval = np.logspace(np.log10(0.001), np.log10(0.1), 20)
# print(kde_xval)

def evaluate(loader, mode):
    global methods
    eval_results = dict()
    for m in methods:
        eval_results[m] = dotdict()
        eval_results[m].best_e = []
        eval_results[m].best_r = []
        eval_results[m].best_t = []

    for idx, data in enumerate(loader):
        if idx*o.batch_size>o.val_pairs and mode=='val':
            break
        # F = data['models'][:, :-1].cuda()  # all models [B M]
        # C = data['correspondences'].numpy()
        if o.new_models: # sample new models            
            if o.F:
                new_minimal_models_F(data, o.val_samples)
            else:
                new_minimal_models(data, o.val_samples, max_average_sol=5)
            # errors, errors_r, errors_t = new_errors(data)
        else:
            data['models'] = data['models'][:, :-1, :] # select all but GT model residuals [B M N]
            # errors = data['errors'][:, :-1].cuda()
            # err_new, errors_r, errors_t = new_errors(data)
            # data['errors'] = errors
            # print(np.median((errors-err_new).cpu()))
        # compute errors anew
        data['models'] = data['models'].cuda()
        errors_saved = data['errors'][:, :-1].cuda().clone()
        errors, errors_r, errors_t = new_errors(data)
        if not o.new_models:
            for b in range(errors.shape[0]):
                diff = (torch.logical_and(errors[b] < 5, (errors[b] - errors_saved[b]).abs() > 0.1))
                ndiff = diff.sum()
                if ndiff >0:
                    id = torch.argwhere(diff).cpu().numpy().squeeze(-1)
                    print(data['files'][b])
                    print('model indices:', id)
                    np.set_printoptions(precision=5)
                    print('saved errors:', errors_saved[b][id].cpu().numpy())
                    print('computed errors:', errors[b][id].cpu().numpy())
                assert(ndiff == 0)
        
        if False:
            for i, models in enumerate(data['models']):
                nan_mask = np.unique(np.where(np.isnan(models.cpu().numpy()))[0])
                if len(nan_mask) != 0:
                    data['models'][i][nan_mask] = torch.eye(3)
                    data['errors'][i][nan_mask] = 180
        
        if False: #
            best_idx = data['errors'][:,:-1].argmin(dim=-1) # oracle has access to GT error function
            best_models = select_dim1(data['models'], best_idx)
            
            # new_perturbed_models(data, 500, 2.5)
            augment_models(data, best_models, 500, 0.7)
            new_errors(data)
        
        if mode == 'test':
            pass
            # doctor 0'th model to be the selected from GC-RANCAS-MAGSAC++
            # data['models'][:,0,:,:] = data['magsac_selected'] # replace at index 0
        compute_residuals(data)
        R = data['residuals']
        # R = data['residuals'][:, :-1, :].cuda() # select all but GT model residuals [B M N]
        # errors = data['errors'][:, :-1]  # select all but GT model errors [B M]
        assert((errors <= 180).all())
        exclude_mask = errors > 1000
        # hash = dict()
        if mode == 'kde':
            SS = []
            unnormalized_points(data)
            for bw in kde_xval:
                compute_kde_weights(data, bw)
                weights = data['kde_weights'].cuda().unsqueeze(-2)
                ss = sufficient_statistic(R, N_bins, max_distance=max_distance, weights = weights)  # [B M K]
                SS.append(ss.counts)
            SS = torch.stack(SS, dim=0) # [w B M K]
              
        for W in methods:
            if isinstance(W, MethodGT):
                continue
            if isinstance(W, Oracle): # oracle errors
                best_idx = errors.argmin(dim=-1) # oracle has access to GT error function
                best_models = select_dim1(data['models'], best_idx)
            # elif isinstance(W, GCMAGSAC):
            #     best_idx = errors.argmin(dim=-1) # does not matter
            #     best_models = data['magsac_selected']
            else:
                W.to(R)
                if mode == 'kde':
                    M = W.val_w
                    # scores = torch.einsum('BMKw, K ->BMw',SS, M) # [B M w]
                    scores = (SS @ M).permute(1,2,0) # [B M w]
                    best_s, best_idx = scores.max(dim=1)
                    best_s = best_s.cpu()
                    best_idx = best_idx.cpu()
                else:
                    ss = sufficient_statistic(R, W.N_bins, max_distance=W.max_distance, pow=W.pow)  # [B M K]
                    if mode == 'val' and not hasattr(W, 'locked'):
                        # M = W.M.T
                        # scores = ss.counts @ M  # [B M K] @ [K T] -> [B M T]
                        scores = torch.einsum('bmK, ...K ->bm...',ss.counts, W.M) # [B M ...]
                        if scores.ndim > 3:
                            scores = scores.logsumexp(dim = -1)
                        scores[exclude_mask,:] = 0
                        best_s, best_idx = scores.max(dim=1)  # [B T] / [B]
                        best_s = best_s.cpu()
                        best_idx = best_idx.cpu()
                    else:
                        M = W.val_w
                        best_s, best_idx = best_score(ss, M)
                # if W.name=='RANSAC(3)' and idx ==0:
                #     for b in range(best_idx.shape[0]):
                #         print('files:', data['files'][b], end='')
                #         print(f' score: {best_s[b]:3.2f}', end='')
                #         print(f' idx: {best_idx[b]:3.2f}', end='')
                #         print(f' idx: {best_idx[b]:3.2f}', end='')
                best_models = select_dim1(data['models'], best_idx)
                
            # compute solution errors
            if mode == 'val' or mode == 'kde':
                best_e = select_dim1(errors, best_idx)  # error of the best model [B, T]
                best_r = select_dim1(errors_r, best_idx)  # error of the best model [B, T]
                best_t = select_dim1(errors_t, best_idx)  # error of the best model [B, T]
            else: # mode == 'test':
                # LO of selected models:
                if polish == 0:
                    best_models = best_models.cpu().numpy()
                elif polish == 'GaU':
                    best_models1, best_scores = local_optimization_ours(data, best_models.cuda(), WGU)
                    best_models = best_models1.cpu().numpy()
                elif polish == 'GaU1':
                    WP = copy.deepcopy(WGU)
                    WP.set_hyperparam(WP.tau/1.1)
                    best_models1, best_scores = local_optimization_ours(data, best_models.cuda(), WP)
                    best_models = best_models1.cpu().numpy()                    
                elif polish == 'MSAC':
                    best_models1, best_scores = local_optimization_ours(data, best_models.cuda(), WMSAC)
                    best_models = best_models1.cpu().numpy()
                elif polish >0:
                    best_ee1, best_eer1, best_eet1, best_ss1, best_models1 = polish_func(data, best_idx, M, W)
                    # TODO: since the polish function knows the score, it can make the choice of whether to keep the model internally
                    m = best_ss1 > best_s  # where polished models are improving the current score
                    best_e[m] = best_ee1[m]  # record their GT error
                    # import pdb; pdb.set_trace()
                    best_r[m] = best_eer1[m]
                    best_t[m] = best_eet1[m]
                    pass               
                # compute test error of found models
                # assume best_models [b, 3, 3] -- selected models by optimization or not
                best_e, best_r, best_t = pose_error_batch(best_models, data)
                    
            eval_results[W].best_e += [best_e.cpu().numpy()]
            eval_results[W].best_r += [best_r.cpu().numpy()]
            eval_results[W].best_t += [best_t.cpu().numpy()]

        if idx % 10 == 0:
            print(idx*o.batch_size)
    # concatenate all batch results
    for m in methods:
        for k in eval_results[m].keys():
            if len(eval_results[m][k])> 0:
                eval_results[m][k] = np.concatenate(eval_results[m][k], axis=0)
    return eval_results


def evaluate_T(loader, n_pairs):
    """
    method:
    for each image pair and threshold compute error of the best model [B, T]
    analysis: can select subset of images: 
     - compute their median error vs threshold, select threshold
     - evaluate their median error for any given threshold
    """
    global methods
    eval_results = dict()
    for M in methods:
        n = M.name
        eval_results[n] = dotdict()
        eval_results[n].best_e = []
        eval_results[n].best_r = []
        eval_results[n].best_t = []
    for idx, data in enumerate(loader):
        if idx*o.batch_size>n_pairs:
            break
        if o.F:
            new_minimal_models_F(data, o.val_samples)
        else:
            new_minimal_models(data, o.val_samples, max_average_sol=5)
        data['models'] = data['models'].cuda()
        errors, errors_r, errors_t = new_errors(data)
        compute_residuals(data)
        R = data['residuals']
        ss = sufficient_statistic(R, N_bins, max_distance=max_distance, pow=1)  # [B M K]
        for M in methods:
            if isinstance(M, MethodGT) or isinstance(M, Oracle) or hasattr(M, 'locked'):
                continue
            M.to(R)
            scores = torch.einsum('bmK, ...K ->bm...',ss.counts, M.M) # [B M T]
            best_s, best_idx = scores.max(dim=1)  # [B T] / [B]
            best_s = best_s.cpu()
            best_idx = best_idx.cpu()
            best_e = select_dim1(errors, best_idx)  # error of the best model [B, T]
            best_r = select_dim1(errors_r, best_idx)  # error of the best model [B, T]
            best_t = select_dim1(errors_t, best_idx)  # error of the best model [B, T]
            n = M.name
            eval_results[n].best_e += [best_e.cpu().numpy()]
            eval_results[n].best_r += [best_r.cpu().numpy()]
            eval_results[n].best_t += [best_t.cpu().numpy()]
        if idx % 10 == 0:
            print(idx*o.batch_size)
    # concatenate all batch results
    for M in methods:
        n = M.name
        for k in eval_results[n].keys():
            if len(eval_results[n][k])> 0:
                eval_results[n][k] = np.concatenate(eval_results[n][k], axis=0) # [B, T]
    return eval_results


MAX_SOLUTIONS = 10


if o.polish == 'all':
    polishes = [0, 'GaU', 'TRUNCATED', 'CAUCHY', 'TRUNCATED_LE_ZACH']
elif o.polish == 0:
    polishes = [0]
else:
    polishes = [o.polish]
    
# polishes = [0, 'GaU', 'TRUNCATED', 'CAUCHY', 'TRUNCATED_LE_ZACH']
# polishes = [0, 'TRUNCATED']

def npstack(arrays):
    return np.stack(arrays) if len(arrays)>0 else np.zeros(shape =(0,0))

def solve_epipolar(x1, x2):
        m_batch_size = x1.shape[0]
        EE = []
        for i in range(m_batch_size):
            E = poselib.essential_matrix_5pt(x1[i], x2[i])
            EE.extend(E)
        return EE

def test_running(loader):
    global methods
    global polishes
    eval_results = dict()
    for mp in itertools.product(methods, polishes):
        eval_results[mp] = dotdict()
        eval_results[mp].best_M = []
        eval_results[mp].running_s = []
        eval_results[mp].running_e = []
        eval_results[mp].running_r = []
        eval_results[mp].running_t = []
        eval_results[mp].running_M = []

    files = []
    torch.manual_seed(0)
    m_batch_size = 100
    for idx, data in enumerate(loader):
        files += data['files']
        if idx>1000: # 
            break
        C = data['correspondences'] # [B, max_N, 4]
        xx = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
        yy = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
        n_points = data['num_pts']
        res = dict()
        if o.kde: compute_kde_weights(data)
        for mp in itertools.product(methods, polishes):
            res[mp] = dotdict()
            res[mp].running_s = []
            res[mp].running_e = []
            res[mp].running_r = []
            res[mp].running_t = []
            res[mp].running_M = []
            res[mp].best_s = C.new_empty(C.shape[0]).fill_(-torch.inf) # best scores
            res[mp].best_M = C.new_zeros(C.shape[0],3,3)  # best models

        start0 = torch.cuda.Event(enable_timing=True)
        end0 = torch.cuda.Event(enable_timing=True)
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)
        dt1 = 0; dt2 = 0; dt3=0; dtS=0; dtst=0; dtg=0; dtp=0
        for m_batch in range(40):  # rounds of sampling, m_batch_size models in each round # 100for each
            # save GT model from data
            #
            GTEs = data['models'][:, -1].clone().double() #[B,3,3]
            # for each image in the batch generate a batch of models
            # start1.record()
            # max_models = 0
            # models = [[] for b in range(C.shape[0])]
            # for b in range(C.shape[0]):
            #     n = n_points[b]
            #     log_p = (C.new_ones((n,))/n).log()
            #     ii = sample_subsets(5, log_p, m_batch_size)
            #     # sample = points[ii, :]  # [n_samples, k, D]
            #     x1 = xx[b,:n][ii,:].cpu().numpy().astype(float)
            #     y1 = yy[b,:n][ii, :].cpu().numpy().astype(float)
            #     EE = solve_epipolar(x1,y1)
            #     n_models = len(EE)
            #     max_models = max(max_models, n_models)
            #     models[b] = EE
            # end1.record()
            # torch.cuda.synchronize()
            # dt1+= start1.elapsed_time(end1)/1000
            # for b in range(C.shape[0]):
            #     models[b] = np.concatenate([np.stack(models[b]), np.zeros((max_models - len(models[b]), 3, 3))])
            # models = np.stack(models) # stack along batch dim
            # # print(max_models)
            # models = torch.tensor(models).to(dtype= torch.float32).cuda()
            # data['models'] = models
            # start1.record()
            # errors, _, _ = pose_error_batch_torch(data['models'], data)
            # end1.record()
            # torch.cuda.synchronize()
            # dtg += start1.elapsed_time(end1)/1000
            
            start1.record()
            if o.F:
                new_minimal_models_F(data, m_batch_size)
            else:
                new_minimal_models(data, m_batch_size)
            end1.record()
            torch.cuda.synchronize()
            dt1+= start1.elapsed_time(end1)/1000
            start1.record()
            new_errors(data)
            end1.record()
            torch.cuda.synchronize()
            dtg += start1.elapsed_time(end1)/1000
            models = data['models'].float().cuda()
            data['models'] = models
            errors = data['errors'].float().cuda()
            
            if False:
                # pre-filter models
                models = [None] * C.shape[0]
                max_models = 0
                # theta = np.linspace(-2,2,9,endpoint=True)/180*math.pi
                theta = np.linspace(-0.5,0.5,9,endpoint=True)/180*math.pi
                RR = torch.tensor(TR.from_euler('Z', theta).as_matrix()).float().cuda()
                #    
                for b in range(C.shape[0]):
                    # extract SVDZ from GT
                    # GTE = GTEs[b]
                    gt_R = to_tensor(data['gt_R'])[b]
                    gt_T = to_tensor(data['gt_t'])[b].squeeze(-1)
                    GTE = LO.compose_essential_matrix(gt_R, gt_T)
                    # Tp = gt_T
                    # Rp = gt_R
                    # if a == 'SVD-Z':
                    U_GT, S1_GT,V_GT = decompose_SVDZ(GTE.cpu())
                    #
                    mask = errors[b] < 20
                    mask[0] = True 
                    m = data['models'][b][mask].cpu().double()
                    # augment
                    if False:
                        U,s,Vh = torch.linalg.svd(m) # [3,3]
                        S = torch.diag_embed(s)
                        mm = []
                        for R in RR:
                            # recompose
                            m = U @ R @ S @ Vh
                            mm.append(m)
                        mm = torch.cat(mm, axis=0)
                        models[b] = mm
                    # decompose each model
                    mm = []
                    for mi in m:
                        U, S1_mi, V = decompose_SVDZ(mi)
                        mhat = U @ S1_GT @ V
                        # mhat = U_GT @ S1_GT @ R2
                        # mm.append(torch.tensor(mi))
                        mm.append(torch.tensor(mhat))
                    mm = torch.stack(mm, axis = 0)
                    models[b] = mm
                    max_models = max(max_models, models[b].shape[0])
                for b in range(C.shape[0]):
                    models[b] = torch.cat([models[b], m.new_zeros((max_models - models[b].shape[0], 3, 3))])
                models = torch.stack(models) # stack along batch dim
                # models = torch.tensor(models).to(dtype=torch.float32).cuda()
                data['models'] = models.cuda()
                errors, _, _ = pose_error_batch_torch(data['models'], data)
            
            # residuals
            start1.record()
            compute_residuals(data)
            end1.record()
            torch.cuda.synchronize()
            dtS += start1.elapsed_time(end1)/1000
            R = data['residuals']
            for W in methods:
                start0.record()
                if isinstance(W, MethodGT):  # GT
                    best_models = GTEs # GT [B,3,3]
                    best_s = best_models.new_zeros(best_models.shape[0]) # [B]
                else:
                    if isinstance(W, Oracle):  # oracle errors
                        # oracle has access to GT error function
                        best_s, best_idx = (-errors).max(dim=-1)
                    else:
                        if o.kde: 
                            scores = W.score_residuals(R, reduction=None)
                            scores = (scores*data['kde_weights'].to(scores)).sum(dim=-1)
                        else:
                            scores = W.score_residuals(R)
                        best_s, best_idx = scores.max(dim=1) # [B]
                        
                    best_s = best_s.cpu()
                    best_idx = best_idx.cpu()
                    best_models = select_dim1(data['models'], best_idx) 
                # checker = Failure()
                # pre_score_count, selection_failure, degenerate = checker.check(data['correspondences'], data['errors'][:, :-1], data['num_pts'], best_idx)
                # print(pre_score_count)
                # import pdb; pdb.set_trace()  
                end0.record()
                torch.cuda.synchronize()
                dt2 += start0.elapsed_time(end0)/1000
                # so-far-the-best
                for b in range(C.shape[0]):
                    if best_s[b] > res[(W,0)].best_s[b]:
                        res[(W,0)].best_s[b] = best_s[b]
                        res[(W,0)].best_M[b] = best_models[b]
                # polish selected so-far-the-best with polish methods
                for p in polishes:
                    record = False
                    if p == 0:
                        best_models = res[(W,0)].best_M
                        record = True
                    elif (m_batch == 0 or (m_batch+1) % 5 == 0): # apply polish sparsely
                        if W.name  == 'GaU' or (W.name  == 'Oracle' and p == 'GaU') or (W.name  == 'GT' and (m_batch == 0 or p == 'GaU')):
                            # print(W.name, p, m_batch)
                            start0.record()
                            best_models, best_s = local_optimization(data, res[(W,0)].best_M, p)
                            end0.record()
                            torch.cuda.synchronize()
                            dtp += start0.elapsed_time(end0)/1000
                            # Evaluate LO optimized so-far-the-best
                            # end0.record()
                            record = True
                    
                    if record:
                        start0.record()
                        # compute test error of the best found models
                        best_e, best_r, best_t = pose_error_batch_torch(best_models.unsqueeze(1), data)  # [B]
                        best_e = best_e.squeeze(-1).cpu().numpy()
                        best_r = best_r.squeeze(-1).cpu().numpy()
                        best_t = best_t.squeeze(-1).cpu().numpy()
                        end0.record()
                    
                        mp = (W,p)
                        res[mp].running_s += [res[mp].best_s + 0]  # copy
                        res[mp].running_e += [best_e]
                        res[mp].running_r += [best_r]
                        res[mp].running_t += [best_t]
                        res[mp].running_M += [res[mp].best_M.cpu().numpy()]
                        torch.cuda.synchronize()
                        dt3 += start0.elapsed_time(end0)/1000
                # Polish
                # if polish != 0 and W.name in ['GaU','Oracle'] and (m_batch == 0 or (m_batch+1) % 5 == 0): # only occasionally reoptimize current best_M
                #     start0.record()
                #     best_models, best_s = local_optimization(data, res[W].best_M, polish)
                #     # res[W].best_M = best_models # avoid repeatedly re-optimizing
                #     end0.record()
                #     torch.cuda.synchronize()
                #     dtp += start0.elapsed_time(end0)/1000
                #     # Evaluate LO optimized so-far-the-best
                #     start0.record()
                #     best_e, best_r, best_t = pose_error_batch_torch(best_models.unsqueeze(1), data)  # [B]
                #     # end0.record()
                # else:
                #     # Evaluate selected so-far-the-best
                #     start0.record()
                #     best_e, best_r, best_t = pose_error_batch_torch(res[W].best_M.unsqueeze(1), data)  # [B]
                #     # end0.record()
                #
                # best_e = best_e.squeeze(-1).cpu().numpy()
                # best_r = best_r.squeeze(-1).cpu().numpy()
                # best_t = best_t.squeeze(-1).cpu().numpy()
                
                # if polish == 0 and False:
                #     # DEBUG
                #     best_e1, best_r1, best_t1 = pose_error_batch(res[W].best_M.unsqueeze(1).cpu().numpy(), data)  # [B]
                #     assert(np.abs(best_r1 - best_r).max() < 0.1)
                #     assert(np.abs(best_t1 - best_t).max() < 0.1)
                #     assert(np.abs(best_e1 - best_e).max() < 0.1)
                # end0.record()
                
                
        print(f'Timing: Solver: {dt1:3.2f}', f'Residuals: {dtS:3.2f}', f'Oracle: {dtg:3.2f}', f'Scoring: {dt2:3.2f}', f'Ploish: {dtp:3.2f}', f'Eval: {dt3:3.2f}')
        if idx % 10 == 0:
            print(idx)
        # concatenate runing results
        for mp in itertools.product(methods, polishes):
            running_s = npstack(res[mp].running_s).swapaxes(0,1)
            running_e = npstack(res[mp].running_e).swapaxes(0,1)
            running_r = npstack(res[mp].running_r).swapaxes(0,1)
            running_t = npstack(res[mp].running_t).swapaxes(0,1)
            running_M = npstack(res[mp].running_M).swapaxes(0,1)
            eval_results[mp].running_s += [running_s]
            eval_results[mp].running_e += [running_e]
            eval_results[mp].running_r += [running_r]
            eval_results[mp].running_t += [running_t]
            eval_results[mp].running_M += [running_M]
        # if idx > 0:
        #     break
    # concatenate over batches
    for mp in itertools.product(methods, polishes):
        for k in eval_results[mp].keys():
            if len(eval_results[mp][k]) > 0:
                eval_results[mp][k] = np.concatenate(eval_results[mp][k], axis=0)
    return eval_results, files


# _________Load models______________
# %% Load modes
print('________Loading models___________')
max_distance = 10; N_bins = 500 # filter to select models to evaluate
methods = []
files = os.listdir(model_path)
mdict = dict()
for file in files:
    W = None
    f = os.path.abspath(os.path.join(model_path, file))
    if file.endswith(".pkl"):
        W = torch.load(f, weights_only=False)
        # if isinstance(W, ScoreWeightsMonotoneMix):
            # continue
        if isinstance(W, ScoreWeightsTZ):
            continue
        if isinstance(W, ScoreWeightsRANSAC): # will add manually
            continue        
        if isinstance(W, ScoreWeightsMSAC): # will add manually
            continue
        # if isinstance(W, ScoreWeightsMonotoneMix): # skip
            # continue
        if isinstance(W, ScoreWeightsMAGSAC): # will add manually
            continue
        # if isinstance(W, ScoreWeightsMonotoneMix) and W.alpha > 0.1 and not hasattr(W, 'M'):
            # continue        
        if not hasattr(W, 'pow'):
            W.pow = 1
        W.name = file.replace('.pkl', '').replace('magsac', 'MAGSAC++').replace('ransac', 'RANSAC').replace(
            'msac', 'MSAC').replace('_', ' ').replace('bins=500','').replace('tau=10.0','').replace('alpha','gamma')
        # W.name = W.name.split(' ')[0]
        W.file = file
        print(W.name + ',\t max_distance=' + str(W.max_distance))
        if not hasattr(W,'M'):
            W.gen_hyperparams(o.val_thresholds)
            M = W.score_matrix()
            W.register_buffer('M', M)
        else:
            print('loaded M for ' + W.name)
        W.cuda()
        mdict[W.name] = W
        methods.append(W)

methods = []

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
# W.gen_hyperparams(o.val_thresholds)
# M = W.score_matrix()
# W.register_buffer('M', M)
# methods += [W]

# Add GU
W = ScoreWeightsGU(N_bins=N_bins, max_distance=max_distance, pow=1)
W.name = 'GaU'
W.cuda()
W.gen_hyperparams(o.val_thresholds)
M = W.score_matrix()
W.register_buffer('M', M)
WGU = W
methods += [W]

# Add GaU
# W = ScoreWeightsGaU(N_bins=N_bins, max_distance=max_distance, pow=1)
# W.name = 'Marg-GaU'
# W.cuda()
# W.gen_hyperparams(o.val_thresholds)
# M = W.score_matrix()
# W.register_buffer('M', M)
# # methods += [W]

# # Add MSAC
W = ScoreWeightsMSAC(N_bins=N_bins, max_distance=max_distance, pow=1)
W.name = 'MSAC'
W.cuda()
W.gen_hyperparams(o.val_thresholds)
M = W.score_matrix()
W.register_buffer('M', M)
methods += [W]
WMSAC = W

# # Add RANSAC-locked
# W = ScoreWeightsRANSAC(N_bins=N_bins, max_distance=max_distance, pow=1, tau = 3.0)
# W.pow = 1
# W.gen_hyperparams(o.val_thresholds)
# W.val_w = W.score_weights_normalized().cuda()
# W.locked = True
# W.name = 'RANSAC(3)'
# methods += [W]

# # Add MAGSAC++
W = ScoreWeightsMAGSAC(N_bins=N_bins, max_distance=max_distance)
W.name = 'MAGSAC++'
W.pow = 1
W.cuda()
W.gen_hyperparams(o.val_thresholds)
M = W.score_matrix()
W.register_buffer('M', M)
methods += [W]

W = ScoreWeightsRANSAC(N_bins=N_bins, max_distance=max_distance)
W.name = 'RANSAC'
W.pow = 1
W.cuda()
W.gen_hyperparams(o.val_thresholds)
M = W.score_matrix()
W.register_buffer('M', M)
methods += [W]
#
# #
for k in mdict.keys():
    if 'ML' in k:
        methods += [mdict[k]]

# add remaining
for k in mdict.keys():
    if mdict[k] not in methods:
        methods += [mdict[k]]

# Add Oracle and GT
methods += [Oracle(N_bins=N_bins, max_distance=max_distance, pow=1), MethodGT(N_bins=N_bins, max_distance=max_distance, pow=1)]

# if Eval_GCMAGSAC:
#     # need validated WMSAC and WGU models
#     load_methods = load_object(results_file0)
#     for M in load_methods:
#         if M.name == 'MSAC':
#             WMSAC = M
#         if M.name == 'GaU':
#             WGU = M
#     W = GCMAGSAC(N_bins=N_bins, max_distance=max_distance, pow=1)
#     methods = [W]
#     val_scenes = []

[print(m.name) for m in methods]

results = dotdict()
results['method'] = []
vresults = dotdict()

# %% ____________Validation______________________________________
if os.path.exists(val_file) and not o.validate:
    print('Loaded validated methods')
    methods = load_object(val_file)
    WGU = [m for m in methods if isinstance(m, ScoreWeightsGU)][0]
    WMSAC = [m for m in methods if isinstance(m, ScoreWeightsMSAC)][0]
    validate = False
else:
    validate = True
# %%
if validate:
    for M in methods:
        # M.results = []
        M.best_e = []
        M.best_r = []
        M.best_t = []
    # Oe = []
    # Me = []
    # validate on all val scenes
    for val_src in val_scenes:
        val_src_name = val_src.replace('/','_')
        print(f'__Validation on {val_src}___________')
        torch.manual_seed(1)
        dataset = ResidualData(dataset_info, val_src, padding=True) # padding the 
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        eval_results = evaluate(loader, 'val')
        for M in methods:
            M.best_e += [eval_results[M].best_e]
            M.best_r += [eval_results[M].best_r]
            M.best_t += [eval_results[M].best_t]

    # concatenate over validation scenes
    for M in methods:
        if len(M.best_e) > 0:
            M.best_e = np.concatenate(M.best_e, axis=0)  # [N T]
            M.best_r = np.concatenate(M.best_r, axis=0)  # [N T]
            M.best_t = np.concatenate(M.best_t, axis=0)  # [N T]


# %%
def NmAA(data ,axis=-1):
    return 1-AUC_10(data, axis)

if validate:
    import matplotlib.colors as mcolors
    cc =  list(mcolors.TABLEAU_COLORS)
    cc = cc + cc
    stat = np.median
    # stat = np.mean
    # stat = NmAA
    #
    for aligned in [False]:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # ax2 = ax1.twiny()
        bot = 10        
        for (i,W) in enumerate(methods):
            if W.name == 'GT':
                continue
            if 'gamma=30.0' in W.name:
                continue
            # if hasattr(W,'locked'):
                # continue
            # if 'ML' in W.name: # or 'TZ' in W.name: #  and not 'mult' in W.name:
            #     continue
            #     ax = ax2
            # else:
            ax = ax1
                # continue
            has_hyperparam = hasattr(W, 'set_hyperparam') and not (hasattr(W, 'locked') and W.locked)
            # for err_key in ['best_r', 'best_e']:
            for err_key in ['best_e']:
                v = np.array(stat(getattr(W,err_key).T, axis=-1))
                v = v.reshape(v.size)
                v_besti = np.argmin(v)
                if err_key == 'best_e':
                    style = '-'
                    label=W.name
                    if has_hyperparam:
                        W.set_hyperparam(W.hyperparams[v_besti]) # chose hyperparam only is has_hyperparam, not locked, and using best_e metric
                        W.best_hyperparam = W.hyperparams[v_besti]
                        W.hbest_i = v_besti
                        x = W.hyperparams
                        xb = W.best_hyperparam
                        if aligned:
                            x = x/W.best_hyperparam
                            xb = xb/W.best_hyperparam
                        name = W.name + r' ($\tau{=}'+f'{W.best_hyperparam:2.2f}$)'
                        name = name.replace('gamma=30.0', '').replace('gamma=10.0', '')
                        label = name
                        ax.plot(xb, v[v_besti], 'o', label=None, color=cc[i])
                        ax.plot(x, v, style, label=label, color=cc[i])
                    else:
                        ax.axhline(v, label=label, color = 'k', linestyle=style)
                    bot = min(bot, v[v_besti])
                else: # err_key != best_e
                    style = '--'
                    label = None
                    if has_hyperparam:
                        ax.plot(x, v, style, label=label, color=cc[i])
                    else:
                        ax.axhline(v, label=label, color = 'k', linestyle=style)
                
        #plt.axhline(stat(Oe),0,1, color = "k", label= 'Oracle')
        plt.legend()
        ax1.legend(loc=1)
        # bot = v[v_besti]*0.99 - 0.1
        # bot = max(0, bot*0.99 - 0.1)
        bot = 0
        v_best = np.min(stat(methods[0].best_e.T, axis=-1))
        top = v_best*1.5 + 1.0
        if stat == NmAA:
            bot = v[v_besti]*0.99
            top = min(1.0, v_best*1.2)
        if not aligned:
            plt.ylim([bot, top])
            pass
        else:
            plt.ylim([bot, top])
            pass
            # plt.ylim([1.8, 1.8 + 2.2 - 1.5])
            plt.xlim([0,3])
        if aligned:
            ax1.set_xlabel('Aligned by min')
        else:
            ax1.set_xlabel('Hyperparameter $\\tau$ [px]')
        ax1.locator_params(axis='x', nbins=10)
        if False:
            ax2.legend(loc=2)
            ax2.set_xlabel('Inliers ratio $\\gamma$')
            ax2.locator_params(axis='x', nbins=10)
            ax2.set_xscale('logit')
        plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())            
        ax1.locator_params(axis='y', nbins=10)
        ax1.set_ylabel(f'{stat.__name__} error')
        plt.draw()
        # Path('fig').mkdir(exist_ok=True)
        fig_path = results_path + f'/fig/'
        # force_path(fig_path)
        savefig(fig_path + f'validation{"-aligned" if aligned else ""}.pdf')
        plt.show()
        plt.close(fig)

#%%
if validate:
    pass
    # assert(False)
#%%
if validate:
    print('Plotting validation kernels')
    plt.figure()
    for (i,W) in enumerate(methods):
        if hasattr(W, 'locked'):
            continue
        if 'gamma=30.0' in W.name:
                continue
        # if 'ML' in W.name and 'mult' in W.name:
            # continue
        # w = W.score_weights_normalized()
        w = W.M[W.hbest_i]
        W.val_w = w
        pow = 1
        xx = (torch.arange(w.shape[0])/w.shape[0]*W.max_distance**pow)**(1/pow)
        # if isinstance(W, ScoreWeightsMonotoneMix): # or isinstance(W, ScoreWeightsTZ):
            # hparam = f'$\\gamma={W.best_hyperparam*100:3.1f}$%'
        # else:
        # hparam = f'$\\tau={W.best_hyperparam:3.2f}$'
        name = W.name + f'($\\tau={W.best_hyperparam:2.2f}$)'   
        name = name.replace('gamma=30.0','').replace('gamma=10.0','')
        plt.plot(xx, w.cpu().detach(), label=name, color = cc[i], marker = markers[i], markevery = [35+i*20])
    plt.legend()
    # plt.title('Selected kernels')
    # plt.xlim(0,7)
    plt.xlabel('Residual [px]')
    plt.show()
    plt.draw()
    force_path(fig_path)
    savefig(fig_path + f'validation_kernels.pdf')

    force_path(val_file)
    save_object(val_file, methods)
    

# %% ____________KDE Validation______________________________________
if o.kde:
    for M in methods:
        M.best_e = []
    for val_src in val_scenes:
        val_src_name = val_src.replace('/','_')
        print(f'__Validation on {val_src}___________')
        dataset = ResidualData(dataset_info, val_src, padding=True) # padding the 
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
        eval_results = evaluate(loader, 'kde')
        for M in methods:
            M.best_e += [eval_results[M].best_e]
    # concatenate over validation scenes
    for M in methods:
        if len(M.best_e) > 0:
            M.best_e = np.concatenate(M.best_e, axis=0)  # [N T]

# %%
if o.kde:
    import matplotlib.colors as mcolors
    cc =  list(mcolors.TABLEAU_COLORS)
    cc = cc + cc
    stat = np.median
    # stat = np.mean

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()
    for (i,W) in enumerate(methods):
        # if hasattr(W,'locked'):
            # continue
        if 'ML' in W.name: # or 'TZ' in W.name: #  and not 'mult' in W.name:
            continue
            ax = ax2
        else:
            ax = ax1
            # continue
        v = np.array(stat(W.best_e, axis=0))
        v = v.reshape(v.size)
        v_besti = np.argmin(v)
        if hasattr(W, 'set_hyperparam'):
            x = kde_xval
            xb = x[v_besti]
            name = W.name + f'($bw={xb:3.3f}$)'
            ax.plot(x, v, label=name, color=cc[i])
            ax.plot(xb, v[v_besti], 'o', label=None, color=cc[i])
        else:
            ax.axhline(v, label=W.name, color = 'k', linestyle='--')

    plt.legend()
    # ax1.legend(loc=1)
    bot = v[v_besti]*0.99 - 0.1
    v_best = np.min(stat(methods[0].best_e, axis=0))
    top = v_best*1.5 + 1.0
    # plt.ylim([bot, top])
    # plt.xlim([0,3])
    ax1.set_xlabel('KDE bandwidth [normalized]')
    ax1.locator_params(axis='x', nbins=10)
    ax1.locator_params(axis='y', nbins=10)
    ax1.set_ylabel(f'{stat.__name__} pose error')
    plt.draw()
    fig_path = results_path + f'/fig/'
    savefig(fig_path + f'validation-KDE.pdf')
    plt.show()
    plt.close(fig)

# %%
# %%

# if o.running:
#     if os.path.exists(results_running_file) and not o.recompute:
#         methods = load_object(results_running_file)
#     else:
        #assert(False)
        # ________________________ RUNNING TEST_________________________________________________
if o.running:
    print('________RUNNING TEST___________')            
    for src in test_scenes:
        print(src)
        set_results_paths(src)
        if os.path.exists(results_running_file) and not o.recompute:
            pass
        else:
            # results[val_src] = []
            for M in methods:
                M.res = dict()    
            for (M, p) in itertools.product(methods, polishes):
                M.res[p] = dotdict()
                M.res[p].running_s = []
                M.res[p].running_e = []
                M.res[p].running_r = []
                M.res[p].running_t = []
                M.res[p].running_M = []
            # for (i,m) in enumerate(methods):
                # if isinstance(m,Oracle):
                    # del methods[i]
            
            dataset = ResidualData(dataset_info, src, padding=True)
            torch.manual_seed(1)  # for data shuffling
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            results, files = test_running(loader)
            for (M,p) in itertools.product(methods, polishes):
                mp = (M,p)
                M.res[p].running_s += [results[mp].running_s]
                M.res[p].running_e += [results[mp].running_e]
                M.res[p].running_r += [results[mp].running_r]
                M.res[p].running_t += [results[mp].running_t]
                M.res[p].running_M += [results[mp].running_M]
                # M.best_r += [eval_results[M].best_r]
                # M.best_t += [eval_results[M].best_t]

            force_path(results_running_file)
            print('saving:', results_running_file)
            res = dict(methods = methods, files = files)
            save_object(results_running_file, res)


    # plot resutls

    # f = plt.figure()
    # for M in methods:
    #     y = np.array([np.median(v, axis=0) for v in M.running_e]).mean(axis=0)
    #     x = np.arange(len(y))*100
    #     plt.plot(x,y, label=M.name)
    # plt.legend()
    # plt.show()

    # fig_path = results_path + f'/fig/'
    # force_path(fig_path)
    # plt.savefig(fig_path + f'running.pdf')

    # plt.close(f)
    
    # assert(False)
    #exit(0)
if o.static:
    # ________________________1K TEST_________________________________________________
    if os.path.exists(results_file) and not o.recompute:
        methods = load_object(results_file)
    else:
        print('________TEST___________')
        # results[val_src] = []

        for M in methods:
            M.best_e = []
            M.best_r = []
            M.best_t = []

        for src in test_scenes:
            print(src)
            dataset = ResidualData(dataset_info, src, padding=True)
            loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=0,shuffle=False)
            eval_results = evaluate(loader, 'test')
            for M in methods:
                M.best_e += [eval_results[M].best_e]
                M.best_r += [eval_results[M].best_r]
                M.best_t += [eval_results[M].best_t]

        force_path(results_file)
        save_object(results_file, methods)

    # _______________________ ANALYSIS______________________________________________ 
    # Statistics: 
    # 1 median of e,r,t per dataset, averaged
    # 2 AUC of e,r,t per dataset, averaged
    # same for flat data

    stats = [np.median, AUC_10]
    # stats = [AUC_10]

    # joints = [True, False]
    joints = [False]

    maxl = 0
    for M in methods:
        maxl = max(maxl, len(M.name))
    confidence = 0.95
    nn_resamples = {AUC_10: 10, np.median: 10000}

    res_list = []

    print(f'____{dataset_info.name}_polish={polish}___')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for joint in joints:
            for stat in stats:
                n_resamples = nn_resamples[stat]
                print(f'____{"Mean" if not joint else "Total"} {stat.__name__}_____')
                for M in methods:
                    print(M.name.ljust(maxl), end=': ')
                    for err_key in ['best_e', 'best_r', 'best_t']:
                        Data = getattr(M, err_key)
                        # print(f'______{stat.__name__} {err_key}_____')
                        if joint:
                            data = np.concatenate(Data, axis = 0)
                            res = scipy.stats.bootstrap((data,), stat, confidence_level=confidence, method='BCa', n_resamples=n_resamples)
                            ci = res.confidence_interval
                            v = stat(data)
                            d = (ci.high - ci.low)/2
                        else:
                            v = []
                            d = []
                            for data in Data:
                                res = scipy.stats.bootstrap((data,), stat, confidence_level=confidence, method='BCa', n_resamples=n_resamples)
                                ci = res.confidence_interval
                                v += [stat(data)]
                                d += [(ci.high - ci.low)/2]
                            v = np.array(v).mean()
                            d = np.array(d).mean()
                            
                        formatted = format_std(v, d)
                        k = err_key.replace('best_', '')
                        print(f'\t ({k}): {formatted}', end='')
                        rec = dict(method=M.name, stat = stat.__name__, joint = joint, err_key = err_key, val = formatted)
                        res_list += [rec]
                    print('')

    force_path(table_file)
    save_object(table_file, res_list)

#%%
nmethods = {m.name: m for m in methods}
XYZ = ['X', 'Y', 'Z']
axes = XYZ + ['rand', 't', 'SVD-Z']
def compute_geom_stats():
    outf = res_root + f'score_vs_error.pkl'
    if os.path.exists(outf) and not o.R:
        scores, pose_err = load_object(outf)
    else:
        # r_scores = {a: [] for a in axes}
        pose_err = {a:[] for a in axes}
        scores = {a:[] for a in axes}
        for src in test_scenes:
            W = WGU
            # W = nmethods['RANSAC']
            src_name = src.replace('/','_')
            print(f'{src}')
            dataset = ResidualData(dataset_info, src, padding=True) # padding the 
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
            Ntheta = 200
            deg = math.pi/180
            theta = np.linspace(0,20*deg, Ntheta)
            RR = {a: None for a in axes}
            nRR = {a: None for a in axes}
            for a in XYZ:
                RR[a] = torch.tensor(TR.from_euler(a , theta).as_matrix())
                nRR[a] = torch.tensor(TR.from_euler(a , -theta).as_matrix())
            for idx, data in enumerate(loader):
                if idx %10 == 0:
                    print(idx)
                GTEs = data['models'][:,-1].clone().double()
                for loop in range(1):
                    for a in axes:
                        models = []
                        for b in range(GTEs.shape[0]):
                            # take GT model
                            GTE = GTEs[b]
                            gt_R = to_tensor(data['gt_R'])[b]
                            gt_T = to_tensor(data['gt_t'])[b].squeeze(-1)
                            Tp = gt_T
                            Rp = gt_R
                            if a == 'SVD-Z':
                                EE = LO.compose_essential_matrix(torch.tensor(Rp), torch.tensor(Tp))
                                R1, S1, R4 = decompose_SVDZ(EE)
                                dR = TR.from_euler('Z', theta).as_matrix()
                                # perturbed
                                EE = torch.tensor(R1) @ torch.tensor(dR) @ torch.tensor(S1 @ R4)
                            else:
                                if a == 't':
                                    dR = random_rotation(n=Ntheta, deg=torch.tensor(theta)/deg, sampling=None, t=gt_T)
                                    Tp = dR @ gt_T
                                elif a == 'rand':
                                    dR = random_rotation(n=Ntheta, deg=torch.tensor(theta)/deg, sampling = None)
                                    Rp = dR @ gt_R
                                else:
                                    dR = RR[a]  # X/Y/Z rotation
                                    Rp = dR @ gt_R
                                EE = LO.compose_essential_matrix(torch.tensor(Rp), torch.tensor(Tp))
                            models += [EE.float()]
                        data['models'] = torch.stack(models)
                        new_errors(data)
                        # compute its score
                        compute_residuals(data)
                        R = data['residuals']
                        score  = W.score_residuals(R)
                        scores[a] += [score.cpu()]
                        pose_err[a] += [data['errors'].cpu()]
        for a in axes:
            # r_scores[a] = torch.cat(r_scores[a], dim=0)
            scores[a] = torch.cat(scores[a], dim=0)
            pose_err[a] = torch.cat(pose_err[a], dim=0)
        save_object(outf, [scores, pose_err])
    return scores, pose_err
# %%
if o.geom:
    scores, pose_err = compute_geom_stats()
    fig = plt.figure()
    labels = {'X': 'Pitch', 'Y':'Yaw', 'Z':'Roll', 'rand':'Random Rot', 't':'Translation', 'SVD-Z':'SVD-Z'}
    prop_cycle = plt.rcParams['axes.prop_cycle']
    cc = prop_cycle.by_key()['color']
    for i,a in enumerate(axes):
        # s = r_scores[a].std(dim=0).numpy()
        xx = pose_err[a].numpy()
        immask = scores[a][:,0] >= 5
        frac_retained = immask.sum()/immask.shape[0]
        print('Fraction of rejected images:', 1-frac_retained)
        sc = scores[a][immask,:]
        data = (sc/sc[:,0:1]).numpy()
        y = np.mean(data, axis=0)
        # data = r_scores[a].clone().numpy()
        # data.sort(axis=0)
        # for q in [0.5, 0.7]:
            # y = np.median(data, axis=0)
        # nth = int(data.shape[0]*0.99)
        # y = np.partition(data, nth, axis=0)[nth]
        # y = data[nth,:]
        # y = np.mean(data[:nth,:], axis=0)
        # s = np.std(data[:nth,:], axis=0)
        # x = theta/math.pi*180
        x = xx.mean(axis=0)
        l = plt.plot(x, y, label = labels[a], color = cc[i])
        # nth = int(data.shape[0]*0.7)
        # y = data[nth,:]
        # l = plt.plot(x, y, ':', color = cc[i], alpha= 0.3)
            # nth = data.shape[0]*70//100
            # y = np.partition(data, nth, axis=0)[nth]
            # plt.plot(x, y, ":", color = l[0].get_color(), label=None)
        # y1 = y - s*0.5
        # y2 = y + s*0.5
        # plt.fill_between(x, y1, y2, alpha = 0.1)
        # med_err = np.median(pose_err[a].cpu().numpy(), axis=0)
        # plt.plot(theta, med_err, '--', label = None)
    plt.xlabel('Pose error, deg')
    plt.ylabel(f'Relative Score, Mean of {frac_retained*100:3.1f}%')
    legend = plt.legend(loc=1)
    if False:
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='k', linestyle='-', lw=1), Line2D([0], [0], color='k', linestyle=':', lw=1)]
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        handles.append(custom_lines[0])
        labels.append('50%')
        handles.append(custom_lines[1])
        labels.append('70%')   
        legend._legend_box = None
        legend._init_legend_box(handles, labels)
        legend._set_loc(legend._loc)
        legend.set_title(legend.get_title().get_text())    
    plt.yscale('log')
    plt.ylim(bottom=1e-2, top = 1.05)
    plt.xlim(left=0)
    plt.draw()
    outf = res_root + f'score_vs_error.pdf'
    savefig(outf)
    plt.show()
    plt.close(fig)
        # pass
        # break
            # evaluate 

#%%
# def compute_geom_stats_hist():
if o.inliers:
    nmethods = {m.name: m for m in methods}
    axes = ['rand']
    deg = math.pi/180
    thetas = np.array([0, 0.1, 0.2, 0.5, 1, 5, 10, 20])*deg
    outf = res_root + f'residuals_vs_error.pkl'
    if os.path.exists(outf) and False:
        hists, pose_err, bins = load_object(outf)
    else:
        # r_scores = {a: [] for a in axes}
        pose_err = {a:[] for a in axes}
        hists = {a:[] for a in axes}
        for src in test_scenes:
            print(f'{src}')
            dataset = ResidualData(dataset_info, src, padding=True) # padding the 
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
            for idx, data in enumerate(loader):
                if idx %10 == 0:
                    print(idx)
                GTEs = data['models'][:,-1].clone().double()
                for loop in range(1):
                    for a in axes:
                        models = []
                        for b in range(GTEs.shape[0]):
                            GTE = GTEs[b]
                            gt_R = to_tensor(data['gt_R'])[b]
                            gt_T = to_tensor(data['gt_t'])[b].squeeze(-1)
                            Tp = gt_T
                            Rp = gt_R
                            # t rotation
                            dR = random_rotation(n=len(thetas), deg=torch.tensor(thetas)/deg, sampling=None, t=gt_T)
                            Tp = dR @ gt_T
                            Tp = gt_T
                            # Rn rotation
                            dR = random_rotation(n=len(thetas), deg=torch.tensor(thetas)/deg, sampling = None)
                            Rp = dR @ gt_R
                            EE = LO.compose_essential_matrix(torch.tensor(Rp), torch.tensor(Tp))
                            models += [EE.float()]
                        data['models'] = torch.stack(models)
                        new_errors(data)
                        # compute residuals hists
                        compute_residuals(data)
                        R = data['residuals']
                        ss = sufficient_statistic(R, N_bins=100, max_distance=max_distance)
                        hists[a] += [ss.counts.cpu()]
                        pose_err[a] += [data['errors'].cpu()]
        for a in axes:
            # r_scores[a] = torch.cat(r_scores[a], dim=0)
            hists[a] = torch.cat(hists[a], dim=0)
            pose_err[a] = torch.cat(pose_err[a], dim=0)
        bins = ss.bins
        save_object(outf, [hists, pose_err, bins])
    # return hists, pose_err, bins
# %%
if o.inliers:
    # hists, pose_err, bins = compute_geom_stats_hist()
    for cum in [False, True]:
        fig = plt.figure()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        cc = prop_cycle.by_key()['color']
        a = axes[0]
        for i, theta in enumerate(thetas):
            # s = r_scores[a].std(dim=0).numpy()
            xx = pose_err[a][:,i].numpy()
            immask = hists[a][:,0,:50].sum(dim=-1) >= 5
            frac_retained = immask.sum()/immask.shape[0]
            # print('Fraction of rejected images:', 1-frac_retained.item())
            # sc = hists[a][immask,:]
            data = hists[a][:,i].numpy()
            # data = data/(data[:,:-1].sum(axis=1)[:,None]+1)
            y = np.mean(data, axis=0)
            if cum:
                y = np.cumsum(y)
            x = xx.mean(axis=0)
            l = plt.plot(bins[:-2].cpu(), y[:-1], label = f'$e={x:2.1f}^\circ$', color = cc[i])
        if cum:
            plt.ylabel(f'Average number of inliers')
            plt.xlabel('Threshold [px]')
        else:
            plt.xlabel('Residual value [px], 100 bins')
            plt.ylabel(f'Average bin count')
            plt.yscale('log')
        legend = plt.legend(loc=1)
        # plt.yscale('log')
        # plt.ylim(bottom=1e-2, top = 1.05)
        # plt.xlim(left=0)
        plt.draw()
        outf = res_root + f'residuals_vs_error{"-cum" if cum else ""}.pdf'
        savefig(outf)
        plt.show()
        plt.close(fig)
# %%

if o.var:
    print('Variance Test')
    vt_scenes = val_scenes + test_scenes
    res = dict()
    # load or compute T results
    for val_src in vt_scenes:
        val_src_name = val_src.replace('/','_')
        var_file = res_root + val_src + '/val_T.pkl'
        if os.path.exists(var_file) and True:
            print(f'loading {var_file}')
            eval_results = load_object(var_file)
        else:
            print(f'__Multithreshold on {val_src}___________')
            torch.manual_seed(1)
            dataset = ResidualData(dataset_info, val_src, padding=True)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            if val_src in val_scenes:
                pairs_boud = 10000/len(val_scenes)
            else:
                pairs_boud = 10000/len(test_scenes)
            eval_results = evaluate_T(loader, pairs_boud) # dict of results for each scene
            print(f'saving {var_file}')
            force_path(var_file)
            save_object(var_file, eval_results)
        res[val_src] = eval_results
    #
    MM = [M for M in methods if not(isinstance(M, MethodGT) or isinstance(M, Oracle) or hasattr(M, 'locked'))]
        
# %%
if o.var:
    print('Measuring Test Error Variance for fixed training sample size')
    # MM = [M for M in methods if not(isinstance(M, MethodGT) or isinstance(M, Oracle) or hasattr(M, 'locked'))]
    # concatenate all val results together and all test results together
    # e_val = dict()
    # e_test = dict()
    eres = {M.name:dotdict() for M in MM}
    for M in MM:
        name = M.name
        e_val = np.concatenate([res[s][name].best_e for s in val_scenes], axis=0) # [B, T] (i. e. [image pair, threshold])
        e_test = np.concatenate([res[s][name].best_e for s in test_scenes], axis=0)
        # sample bootstrap subsets
        val_samples = 10 # size of the validation sample
        bootstrap_samples = 5000 # how many times to re-draw the validation subset
        n = e_val.shape[0]
        ee = []
        for s in range(bootstrap_samples): #bootstrap samples
            index = np.random.choice(n, val_samples, replace=False) # choose a validation subset
            # statistic
            V = np.median(e_val[index],axis=0) # we use median pose error as the criterion, here median over the selected validation subset
            t_best = np.argmin(V) # select best hyperparameter
            # evaluate it on test set
            e = np.median(e_test[:, t_best]) # all test errors for the selected threshold.
            # This computes flat median over test scenes, don't we want to compute mean median statistic as usual?
            ee += [e]
        ee = np.array(ee) # list of test errors [bootstrap_samples]
        # Expected test error:
        Ee = np.mean(ee)
        # Get a confidence interval on the mean
        bres = scipy.stats.bootstrap((ee,), np.mean, confidence_level=0.95, method='BCa', n_resamples=10000)
        ci = bres.confidence_interval
        d = (ci.high - ci.low)/2
        eres[name].ee = ee
        eres[name].expected = Ee
        eres[name].expected_ci = [ci.low, ci.high]
        eres[name].expected_d = d
        print(M.name + f'\t Expected median pose error: {Ee:3.3f} +- {d:3.3f}') # 
        # Get a confidence interval on std
        bres = scipy.stats.bootstrap((ee,), np.std, confidence_level=0.95, method='BCa', n_resamples=10000)
        ci = bres.confidence_interval
        d = (ci.high - ci.low)/2
        eres[name].v = np.std(ee)
        eres[name].ci = [ci.low, ci.high]
        eres[name].d = d
# %% print out some examples of test error for different trianing samples
if o.var:
    for k in range(20):
        for M in MM:
            name = M.name
            if 'gamma=30' in name or 'RANSAC' in name or 'gamma' in name or 'MSAC' in name:
                continue
            name = name.replace('gamma=30.0', '').replace('gamma=10.0', '')
            if k == 0:
                print(name + '\t', end='')
            else:
                e = eres[M.name].ee[k]
                print(f'{e:3.2f}\t', end='')
        print('')
       

# %%
# %%
if o.var and False:
    f = plt.figure(figsize=(5,2))
    names = []
    vv = []
    err = []
    for M in MM:
        name = M.name
        if 'gamma=30' in name:
            continue
        v = eres[name].v
        vv += [v]
        err += [[v - eres[name].ci[0], eres[name].ci[1] - v]]
        name = name.replace('gamma=30.0', '').replace('gamma=10.0', '')
        names += [name]
    plt.gca().bar(names, vv, color=cc[0:len(names)], yerr = np.array(err).T, capsize=7)
    plt.gca().set_ylabel('std')
    # plt.title('Std of test error w.r.t. random validation subset')
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.draw()
    outf = res_root + f'sensitivity.pdf'
    savefig(outf)
    plt.show()
    plt.close(f)

#%%
# %% Expected test error vs training set size
if o.var:
    def expected_series(nn, sigmas, adjust_thresholds=False, subsample_MAGSAC=False):
        # torch.manual_seed(1) # DEBUG
        # np.random.seed(1)
        from scipy.ndimage import gaussian_filter
        MM = [M for M in methods if not(isinstance(M, MethodGT) or isinstance(M, Oracle) or hasattr(M, 'locked'))]
        # concatenate all val results together and all test results together
        # e_val = dict()
        # e_test = dict()
        eres = {M.name:dotdict() for M in MM}
        for M in MM:
            name = M.name
            e_val = np.concatenate([res[s][name].best_e for s in val_scenes], axis=0) # [B, T] (i. e. [image pair, threshold])
            e_test = np.concatenate([res[s][name].best_e for s in test_scenes], axis=0)
            # sample bootstrap subsets
            val_samples = 1000 # size of the validation sample
            # nn = np.array([2,4,8])
            bootstrap_samples = 1000 # how many times to re-draw the validation subset
            N = e_val.shape[0]
            ee = []
            for s in range(bootstrap_samples): #bootstrap samples
                # index = np.random.choice(N, val_samples, replace=False) # choose a validation subset
                # statistic
                ne = []
                if sigmas is None or sigmas == [None]:
                    sigmas = [M.best_sigma]
                    assert(not adjust_thresholds, 'not implemented correctly')
                    # range_adjust = True
                    pass
                else:
                    pass
                    # range_adjust = False                    
                for n,sigma in itertools.product(nn, sigmas):
                    # idx = index[:n+1]
                    idx = np.random.choice(N, n, replace=False) # choose a validation subset
                    V = np.median(e_val[idx],axis=0) # we use median pose error as the criterion, here median over the selected validation subset
                    nT = len(V)
                    # sigma=2
                    tt = M.hyperparams
                    if adjust_thresholds:
                        if 'MAGSAC' in name:
                            # drop_mask = np.ones(len(tt))
                            # drop_mask[4:200:3] = 0
                            # V[drop_mask>0] = 100
                            V[tt<0.3] = 1000 # range adjustment
                            if subsample_MAGSAC:
                                V[5::3] = 1000
                                V[6::3] = 1000
                        else:
                            V[tt>10/3] = 1000
                        # sigma *= 3
                    if sigma > 0:
                        assert(False)
                        weight = gaussian_filter(np.ones(nT), sigma=sigma, mode='constant')
                        smoothed_V = gaussian_filter(V, sigma=sigma, mode='constant')
                        smoothed_V = smoothed_V/weight
                    else:
                        smoothed_V = V
                    # smoothed_V = V
                    # smoothed_V += (np.arange(n) - n/2)
                    t_best = np.argmin(smoothed_V) # select best hyperparameter
                    # evaluate it on test set
                    e = np.median(e_test[:, t_best]) # all test errors for the selected threshold.
                    # This computes flat median over test scenes, don't we want to compute mean median statistic as usual?
                    ne += [e]
                e = np.array(ne)
                ee += [e]
            ee = np.vstack(ee) # [bootstrap_samples, nn]
            # Expected test error:
            Ee = np.mean(ee, axis = 0)
            std = np.std(ee, axis = 0)
            # print(M.name)
            # print(Ee)
            # Get confidence intervals on the mean
            cci = []
            cci_std = []
            for k in range(ee.shape[1]): # for each set of test errors in the series
                bres = scipy.stats.bootstrap((ee[:,k],), np.mean, confidence_level=0.95, method='BCa', n_resamples=10000)
                ci = bres.confidence_interval
                cci += [np.array([ci.low, ci.high])]
                bres = scipy.stats.bootstrap((ee[:,k],), np.std, confidence_level=0.95, method='BCa', n_resamples=10000)
                ci = bres.confidence_interval
                cci_std += [np.array([ci.low, ci.high])]
            cci = np.vstack(cci) # [nn, 2]
            cci_std = np.vstack(cci_std) # [nn, 2]
            eres[name].expected = Ee
            eres[name].expected_ci = cci
            eres[name].std = std
            eres[name].std_ci = cci_std
        return eres
    # nn = 2**np.arange(1,11,1)
# tune_sigma = 'SPSG' in dataset_info.name
    tune_sigma = False
# %%
if o.var and tune_sigma:
    nn = [2]
    sigmas = [0,0.5,1,2,3,4,5,6,7]
    eres_s = expected_series(nn, sigmas)
# %%
if o.var and tune_sigma:
    # smoothing parameter selection
    f = plt.figure(figsize=(8,4))
    names = []
    vv = []
    err = []
    i = 0
    for M in MM:
        name = M.name
        if 'gamma=30' in name:
            M.best_sigma = 0
            continue
        v = eres_s[name].expected
        cci = eres_s[name].expected_ci
        name = name.replace('gamma=30.0', '').replace('gamma=10.0', '')
        names += [name]
        plt.gca().plot(sigmas, v, '-', label=name, color = cc[i], marker = markers[i])
        plt.fill_between(sigmas,y1 = cci[:,0], y2 = cci[:,1], facecolor=cc[i], alpha=0.3, label=None)
        M.best_sigma =  sigmas[np.argmin(v)]
        i = i + 1
    plt.legend()
    plt.draw()
    plt.xlabel('Sigma')
    plt.title(f'Expected error versus smoothing parameter, val set size={nn[0]}')
    plt.ylabel(r'$\rm\mathbb{E}[e]$')
    plt.show()
    plt.close(f)
    [print(M.name, M.best_sigma) for M in MM]

# %%
if o.var:
    nn = 2**np.arange(1,11,1)
    eres = expected_series(nn, [0])
# %%    
if o.var:
    nn = 2**np.arange(1,11,1)
    eres_smooth = expected_series(nn, [0], adjust_thresholds=True)
    eres_s2 = expected_series(nn, [0], adjust_thresholds=True, subsample_MAGSAC=True)
    # eres_smooth = expected_series(nn, [0]) # repeatability

# %%
if o.var:
    f = plt.figure(figsize=(8,4))
    names = []
    vv = []
    err = []
    i = 0
    for M in MM:
        name = M.name
        if 'gamma=30' in name:
            continue
        v = eres[name].expected
        cci = eres[name].expected_ci
        name = name.replace('gamma=30.0', '').replace('gamma=10.0', '')
        names += [name]
        plt.gca().plot(nn, v, '-', label=name, color = cc[i], marker = markers[i])
        plt.fill_between(nn,y1 = cci[:,0], y2 = cci[:,1], facecolor=cc[i], alpha=0.3, label=None)
        if True:
            vs = eres_smooth[M.name].expected            
            cci_s = eres_smooth[M.name].std_ci
            plt.gca().plot(nn, vs, ':', color = cc[i], marker = markers[i], linewidth=2)        
        # if 'MAGSAC' in name:
        #     vs = eres_s2[M.name].expected
        #     cci_s = eres_s2[M.name].std_ci
        #     plt.gca().plot(nn, vs, '--', color = cc[i], marker = markers[i], linewidth=2)
        # sqrtn = 1000**0.5
        # d1 = (cci[:,1] - v)
        # d0 = (cci[:,0] - v)
        # plt.fill_between(nn,y1 = v+d0*sqrtn, y2 = v + d1*sqrtn, facecolor=cc[i], alpha=0.1, label=None)
        i = i + 1
   
    # plt.gca().set_ylabel('std')
    # plt.title('Std of test error w.r.t. random validation subset')
    # plt.gca().spines[['right', 'top']].set_visible(False)
    plt.legend()
    plt.draw()
    # outf = res_root + f'sensitivity.pdf'
    # savefig(outf)
    # plt.xlim(left=2)
    plt.xscale('log', base=2)
    plt.xlabel('Validatino set size')
    # plt.gca().set_xticklabels(nn)
    plt.ylabel(r'$\rm\mathbb{E}[e]$')
    # plt.title(f'Expected median test pose error vs. validation set size, {dataset_info.name}')
    outf = res_root + f'training_size_e_mean.pdf'
    savefig(outf)
    plt.show()
    plt.close(f)

# %%
if o.var:
    f = plt.figure(figsize=(8,4))
    names = []
    vv = []
    err = []
    i = 0
    for M in MM:
        name = M.name
        if 'gamma=30' in name:
            continue
        v = eres[name].std
        cci = eres[name].std_ci
        name = name.replace('gamma=30.0', '').replace('gamma=10.0', '')
        names += [name]
        plt.gca().plot(nn, v, '-', label=name, color = cc[i], marker = markers[i])
        plt.fill_between(nn,y1 = cci[:,0], y2 = cci[:,1], facecolor=cc[i], alpha=0.3, label=None)
        if True:
            v_s = eres_smooth[M.name].std
            cci_s = eres_smooth[M.name].std_ci
            plt.gca().plot(nn, v_s, ':', color = cc[i], marker = markers[i], label=None)
        # plt.fill_between(nn,y1 = cci_s[:,0], y2 = cci_s[:,1], facecolor=cc[i], alpha=0.3, label=None)
        i = i + 1   
    # plt.gca().set_ylabel('std')
    # plt.title('Std of test error w.r.t. random validation subset')
    # plt.gca().spines[['right', 'top']].set_visible(False)
    plt.legend()
    # outf = res_root + f'sensitivity.pdf'
    # savefig(outf)
    # plt.xlim(left=2)
    plt.xscale('log', base=2)
    plt.xlabel('Validatino set size')
    # plt.gca().set_xticklabels(nn)
    plt.ylabel(r'std[e]')
    # plt.ylim([0,2])
    plt.yscale('log')
    # plt.title(f'Std of test error with respect to random trining set, smoothed, {dataset_info.name}')
    plt.draw()
    outf = res_root + f'training_size_e_std.pdf'
    savefig(outf)
    plt.show()
    plt.close(f)


#%%



# %% Format Table
# res_list = load_object(table_file)
# df = pd.DataFrame.from_records(data, index=None, exclude=None, columns=None, coerce_float=False, nrows=None)
# ss = df.to_latex(index=True)
# ss = ss.replace('±', '$\pm$')
# f = open(table_out, "w")
# f.write(ss)
# f.close()

# # joint results for all test scenes
# for M in methods:
#     M.best_e = np.concatenate(M.best_e, axis=0)  # [N T]
#     M.best_r = np.concatenate(M.best_r, axis=0)  # [N T]
#     M.best_t = np.concatenate(M.best_t, axis=0)  # [N T]

# if len(results.method) == 0:
#     for M in methods:
#         results['method'].append(M.name)

# print('Error stats of method-selected model')
# stats = [np.median]#np.mean, 

# print("results on pose ")

# for (mi,M) in enumerate(methods):
#     print(M.name.ljust(maxl), end=': ')
#     for stat in stats:
#         try: 
#             res = scipy.stats.bootstrap((M.best_e,), stat, confidence_level=0.95, method='BCa', n_resamples=10000)
#         except:
#             print(np.median(M.best_e))
#             continue
#         ci = res.confidence_interval
#         #d = (ci.high - ci.low)/2
#         v = stat(M.best_e)
#         d = max(ci.high-v,v-ci.low)
#         formatted = format_std(v, d)
#         print(f'\t {stat.__name__}={formatted}',end='')
#         if stat == np.median:
#             # results['method'].append(M.name)
#             # results[val_src].append(formatted)
#             if vresults[M.name] is None:
#                 vresults[M.name] = []
#             vresults[M.name].append(v)
#             # results['average'].append(np.array(vresults[M.name]).mean())
#     print('')
#     # print(f' std={np.std(M.best_e)}')
# # %%
# results['hyperparam'] = []
# for W in methods:
#     if isinstance(W, ScoreWeightsMonotoneMix) or isinstance(W, ScoreWeightsMonotoneMix):
#         hparam = f'$\\pi={W.best_hyperparam*100:3.1f}$%'
#     elif hasattr(W, 'best_hyperparam'):
#         hparam = f'$\\tau={W.best_hyperparam:3.2f}$'
#     else:
#         hparam = ''
#     results['hyperparam'].append(hparam)

# %%
if False: # create latex table
    df = pd.DataFrame.from_dict(results)
    mnames = [m.name for m in methods]
    df['method'] = pd.Categorical(df['method'], mnames)

    # display(df)
    ss = df.to_latex(index=True)
    print(ss.replace('±', '$\pm$'))

# %%
# f = plt.figure()
# for M in methods[0:5]:
#     y, bine = np.histogram(M.best_e, 20, range=(0,50), density=True)
#     plt.plot(bine[:-1], y, label=M.name)
#     # plt.hist(M.best_e)
# plt.legend()
# plt.show()
# plt.close(f)
# %%