# translation rotation
# %%
import os, sys
if __name__ == "__main__":   
    __name__ = 'score_learn.evaluate_H.py'
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
from tqdm import tqdm
from scipy.spatial.transform import Rotation as TR
#!%matplotlib inline

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from types import SimpleNamespace

from .score_weights import *
from .load_data import *
from .metrics import *
from . import local_optimization as LO
from .drawing import *

from . import model_H

op = ArgumentParser()
op.add_argument("--batch_size", type=int, default=8, help="number of pairs processed in parallel")
op.add_argument("--val_samples", type=int, default=1000, help="number of minimal samples used for validation")
op.add_argument("--val_pairs", type=int, default=1000, help="number of image pairings per scene used for validation")
op.add_argument("--val_thresholds", type=int, default=200, help="grid of thresholds subdividing [0.1 10] for validation")
op.add_argument("--N_bins", type=int, default=500, help="histogram size for residuals")
op.add_argument("--max_distance", type=float, default=30.0, help="max distance for the histogram")
op.add_argument("--polish", default=0, type=str, help="0 - no polish, 1 - BA, 2 - LMeDs, GaU - our polish (EM-LMA GaU) , MSAC - our polish (EM-LMA MSAC)")
op.add_argument("--data", type=str, default='HEB', help="dataset")
op.add_argument("-V", "--validate", action='store_true', default=False, help="recompute validation")
op.add_argument("-R", "--recompute", action='store_true', default=False, help="recompute test")
op.add_argument("--running", action='store_true', default=False, help="running test")
op.add_argument("--kde", action='store_true', default=False, help="?")
op.add_argument("--geom", action='store_true', default=False, help="error geometry analysis")
op.add_argument("--inliers", action='store_true', default=False, help="inliers statistics experiment")
op.add_argument("--static", action='store_true', default=False, help="static 1K test, DEPRICATED")
# op.add_argument("--var", type=bool, default=False, help="variance test")
op.add_argument("--var", action='store_true', default=False, help="variance test")
op.add_argument("--largeval", action='store_true', default=False, help="variance test")
op.add_argument("--F", action='store_true', default=False, help="use solver for F matrix (regardless of the dataset)")
# DEPRICATED
# op.add_argument("--new_models", type=bool, default=True, help="sample new models, cannot be changed")

args_str = ' '.join(sys.argv[1:])
ops, args = op.parse_known_args(shlex.split(args_str))
o = SimpleNamespace(**vars(ops))
## 
## For interactove model
# o.validate = True
# o.F = True
# o.geom = True
# o.inliers = True
# o.R = True
o.var = True
# o.var = False
o.largeval = True
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
elif not (polish == 'GaU' or polish == 'MSAC'):
    print("undefined polishing method!")

if o.polish == 'all':
    polishes = [0, 'GaU', 'TRUNCATED', 'CAUCHY', 'TRUNCATED_LE_ZACH']
elif o.polish == 0:
    polishes = [0]
else:
    polishes = [o.polish]


for dataset_info in datasets:
    if dataset_info.name == o.data:
        break
if dataset_info.name != o.data:
    print(f'cannot find dataset {o.data}')
    exit(1)

o.type = dataset_info.type

if dataset_info.type == 'H':
    from .model_H import new_minimal_models, pose_error_batch_torch, AUC_10, compute_residuals
    o.minimal_sample = 4
    o.avg_solutions = 1
    o.min_solver = model_H.solve_homography
    o.MAGSAC_dof = 2

else:
    from .model_E import new_minimal_models, pose_error_batch_torch, AUC_10, compute_residuals
    o.minimal_sample = 5
    o.avg_solutions = 1
    o.MAGSAC_dof = 4


Eval_GCMAGSAC = False

val_scenes = dataset_info.val
# val_scenes = dataset_info.test[1:2]
# val_scenes = dataset_info.val + dataset_info.test[0:4]
# val_scenes = dataset_info.val + dataset_info.test
test_scenes = dataset_info.test
res_root = f'results/{dataset_info.name}/'
results_file0 = res_root + f'polish={polish}/' + f'test_results.pkl'
model_path = res_root + 'models/'
if Eval_GCMAGSAC:
    res_root += 'GCMAGSAC/'
results_path = res_root + f'polish={polish}/'
val_file = res_root + f'val_results.pkl'

def create_loader(val_src):
    if dataset_info.type == 'H':
        dataset = H_dataset(dataset_info, val_src, padding=True, snn_threshold=0.7)
    else:
        dataset = ResidualData(dataset_info, val_src, padding=True) # padding the 
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    return loader
    
torch.manual_seed(0)

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


def new_errors(data):
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
        start = time.time()
        new_minimal_models(data, o.val_samples, max_average_sol=o.avg_solutions, min_sample=o.minimal_sample, solver=o.min_solver)
        end = time.time()
        if idx == 0:
            print(f"new_minimal_models takes {end - start:.4f} seconds")
        # compute errors anew
        data['models'] = data['models'].cuda()
        start = time.time()
        errors, errors_r, errors_t = new_errors(data)
        end = time.time()
        if idx == 0:
            print(f"new_errors takes {end - start:.4f} seconds")
        assert((errors <= 180).all())
        if mode == 'test':
            pass
        start = time.time()
        compute_residuals(data)
        end = time.time()
        if idx == 0:
            print(f"compute_residuals takes {end - start:.4f} seconds")
        R = data['residuals']
        exclude_mask = errors > 1000
        # hash = dict()
        # if mode == 'kde':
        #     SS = []
        #     unnormalized_points(data)
        #     for bw in kde_xval:
        #         compute_kde_weights(data, bw)
        #         weights = data['kde_weights'].cuda().unsqueeze(-2)
        #         ss = sufficient_statistic(R, N_bins, max_distance=max_distance, weights = weights)  # [B M K]
        #         SS.append(ss.counts)
        #     SS = torch.stack(SS, dim=0) # [w B M K]
              
        for W in methods:
            if isinstance(W, MethodGT):
                continue
            if isinstance(W, Oracle): # oracle errors
                best_idx = errors.argmin(dim=-1) # oracle has access to GT error function
                best_models = select_dim1(data['models'], best_idx)
            else:
                W.to(R)
                if mode == 'kde':
                    raise NotImplementedError()
                    # M = W.val_w
                    # # scores = torch.einsum('BMKw, K ->BMw',SS, M) # [B M w]
                    # scores = (SS @ M).permute(1,2,0) # [B M w]
                    # best_s, best_idx = scores.max(dim=1)
                    # best_s = best_s.cpu()
                    # best_idx = best_idx.cpu()
                else:
                    ss = sufficient_statistic(R, W.N_bins, max_distance=W.max_distance, pow=W.pow)  # [B M K]
                    if mode == 'val' and not hasattr(W, 'locked'):
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
                    raise NotImplementedError()
                    best_models1, best_scores = local_optimization_ours(data, best_models.cuda(), WGU)
                    best_models = best_models1.cpu().numpy()
                elif polish == 'MSAC':
                    raise NotImplementedError()
                    best_models1, best_scores = local_optimization_ours(data, best_models.cuda(), WMSAC)
                    best_models = best_models1.cpu().numpy()
                elif polish >0:
                    raise NotImplementedError()
                    best_ee1, best_eer1, best_eet1, best_ss1, best_models1 = polish_func(data, best_idx, M, W)
                    m = best_ss1 > best_s  # where polished models are improving the current score
                    best_e[m] = best_ee1[m]  # record their GT error
                    # import pdb; pdb.set_trace()
                    best_r[m] = best_eer1[m]
                    best_t[m] = best_eet1[m]
                    pass               
                best_e, best_r, best_t = pose_error_batch_torch(best_models, data)
                    
            eval_results[W].best_e += [best_e.cpu().numpy()]
            eval_results[W].best_r += [best_r.cpu().numpy()]
            eval_results[W].best_t += [best_t.cpu().numpy()]

        if idx % 10 == 0:
            print(idx*o.batch_size)
        if idx > 500:
            break # DEBUG
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
        new_minimal_models(data, o.val_samples, max_average_sol=5)
        data['models'] = data['models'].cuda()
        errors, errors_r, errors_t = new_errors(data)
        compute_residuals(data)
        R = data['residuals']
        ss = sufficient_statistic(R, N_bins, max_distance=max_distance, pow=1)  # [B M K]
        for M in methods:
            if isinstance(M, MethodGT) or isinstance(M, Oracle) or hasattr(M, 'locked'):
                if isinstance(M, Oracle):
                    best_idx = errors.argmin(dim=-1) # oracle has access to GT error function
            else:
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

   
def npstack(arrays):
    return np.stack(arrays) if len(arrays)>0 else np.zeros(shape =(0,0))

# def test_running(loader):
#     global methods
#     global polishes
#     eval_results = dict()
#     for mp in itertools.product(methods, polishes):
#         eval_results[mp] = dotdict()
#         eval_results[mp].best_M = []
#         eval_results[mp].running_s = []
#         eval_results[mp].running_e = []
#         eval_results[mp].running_r = []
#         eval_results[mp].running_t = []
#         eval_results[mp].running_M = []

#     files = []
#     torch.manual_seed(0)
#     m_batch_size = 100
#     for idx, data in enumerate(loader):
#         files += data['files']
#         if idx>1000: # 
#             break
#         C = data['correspondences'] # [B, max_N, 4]
#         xx = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
#         yy = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
#         n_points = data['num_pts']
#         res = dict()
#         if o.kde: compute_kde_weights(data)
#         for mp in itertools.product(methods, polishes):
#             res[mp] = dotdict()
#             res[mp].running_s = []
#             res[mp].running_e = []
#             res[mp].running_r = []
#             res[mp].running_t = []
#             res[mp].running_M = []
#             res[mp].best_s = C.new_empty(C.shape[0]).fill_(-torch.inf) # best scores
#             res[mp].best_M = C.new_zeros(C.shape[0],3,3)  # best models

#         start0 = torch.cuda.Event(enable_timing=True)
#         end0 = torch.cuda.Event(enable_timing=True)
#         start1 = torch.cuda.Event(enable_timing=True)
#         end1 = torch.cuda.Event(enable_timing=True)
#         dt1 = 0; dt2 = 0; dt3=0; dtS=0; dtst=0; dtg=0; dtp=0
#         for m_batch in range(40):  # rounds of sampling, m_batch_size models in each round # 100for each
#             # save GT model from data
#             #
#             GTEs = data['models'][:, -1].clone().double() #[B,3,3]
#             # for each image in the batch generate a batch of models
#             # start1.record()
#             # max_models = 0
#             # models = [[] for b in range(C.shape[0])]
#             # for b in range(C.shape[0]):
#             #     n = n_points[b]
#             #     log_p = (C.new_ones((n,))/n).log()
#             #     ii = sample_subsets(5, log_p, m_batch_size)
#             #     # sample = points[ii, :]  # [n_samples, k, D]
#             #     x1 = xx[b,:n][ii,:].cpu().numpy().astype(float)
#             #     y1 = yy[b,:n][ii, :].cpu().numpy().astype(float)
#             #     EE = solve_epipolar(x1,y1)
#             #     n_models = len(EE)
#             #     max_models = max(max_models, n_models)
#             #     models[b] = EE
#             # end1.record()
#             # torch.cuda.synchronize()
#             # dt1+= start1.elapsed_time(end1)/1000
#             # for b in range(C.shape[0]):
#             #     models[b] = np.concatenate([np.stack(models[b]), np.zeros((max_models - len(models[b]), 3, 3))])
#             # models = np.stack(models) # stack along batch dim
#             # # print(max_models)
#             # models = torch.tensor(models).to(dtype= torch.float32).cuda()
#             # data['models'] = models
#             # start1.record()
#             # errors, _, _ = pose_error_batch_torch(data['models'], data)
#             # end1.record()
#             # torch.cuda.synchronize()
#             # dtg += start1.elapsed_time(end1)/1000
            
#             start1.record()
#             if o.F:
#                 new_minimal_models_F(data, m_batch_size)
#             else:
#                 new_minimal_models(data, m_batch_size)
#             end1.record()
#             torch.cuda.synchronize()
#             dt1+= start1.elapsed_time(end1)/1000
#             start1.record()
#             new_errors(data)
#             end1.record()
#             torch.cuda.synchronize()
#             dtg += start1.elapsed_time(end1)/1000
#             models = data['models'].float().cuda()
#             data['models'] = models
#             errors = data['errors'].float().cuda()
            
#             if False:
#                 # pre-filter models
#                 models = [None] * C.shape[0]
#                 max_models = 0
#                 # theta = np.linspace(-2,2,9,endpoint=True)/180*math.pi
#                 theta = np.linspace(-0.5,0.5,9,endpoint=True)/180*math.pi
#                 RR = torch.tensor(TR.from_euler('Z', theta).as_matrix()).float().cuda()
#                 #    
#                 for b in range(C.shape[0]):
#                     # extract SVDZ from GT
#                     # GTE = GTEs[b]
#                     gt_R = to_tensor(data['gt_R'])[b]
#                     gt_T = to_tensor(data['gt_t'])[b].squeeze(-1)
#                     GTE = LO.compose_essential_matrix(gt_R, gt_T)
#                     # Tp = gt_T
#                     # Rp = gt_R
#                     # if a == 'SVD-Z':
#                     U_GT, S1_GT,V_GT = decompose_SVDZ(GTE.cpu())
#                     #
#                     mask = errors[b] < 20
#                     mask[0] = True 
#                     m = data['models'][b][mask].cpu().double()
#                     # augment
#                     if False:
#                         U,s,Vh = torch.linalg.svd(m) # [3,3]
#                         S = torch.diag_embed(s)
#                         mm = []
#                         for R in RR:
#                             # recompose
#                             m = U @ R @ S @ Vh
#                             mm.append(m)
#                         mm = torch.cat(mm, axis=0)
#                         models[b] = mm
#                     # decompose each model
#                     mm = []
#                     for mi in m:
#                         U, S1_mi, V = decompose_SVDZ(mi)
#                         mhat = U @ S1_GT @ V
#                         # mhat = U_GT @ S1_GT @ R2
#                         # mm.append(torch.tensor(mi))
#                         mm.append(torch.tensor(mhat))
#                     mm = torch.stack(mm, axis = 0)
#                     models[b] = mm
#                     max_models = max(max_models, models[b].shape[0])
#                 for b in range(C.shape[0]):
#                     models[b] = torch.cat([models[b], m.new_zeros((max_models - models[b].shape[0], 3, 3))])
#                 models = torch.stack(models) # stack along batch dim
#                 # models = torch.tensor(models).to(dtype=torch.float32).cuda()
#                 data['models'] = models.cuda()
#                 errors, _, _ = pose_error_batch_torch(data['models'], data)
            
#             # residuals
#             start1.record()
#             compute_residuals(data)
#             end1.record()
#             torch.cuda.synchronize()
#             dtS += start1.elapsed_time(end1)/1000
#             R = data['residuals']
#             for W in methods:
#                 start0.record()
#                 if isinstance(W, MethodGT):  # GT
#                     best_models = GTEs # GT [B,3,3]
#                     best_s = best_models.new_zeros(best_models.shape[0]) # [B]
#                 else:
#                     if isinstance(W, Oracle):  # oracle errors
#                         # oracle has access to GT error function
#                         best_s, best_idx = (-errors).max(dim=-1)
#                     else:
#                         if o.kde: 
#                             scores = W.score_residuals(R, reduction=None)
#                             scores = (scores*data['kde_weights'].to(scores)).sum(dim=-1)
#                         else:
#                             scores = W.score_residuals(R)
#                         best_s, best_idx = scores.max(dim=1) # [B]
                        
#                     best_s = best_s.cpu()
#                     best_idx = best_idx.cpu()
#                     best_models = select_dim1(data['models'], best_idx) 
#                 # checker = Failure()
#                 # pre_score_count, selection_failure, degenerate = checker.check(data['correspondences'], data['errors'][:, :-1], data['num_pts'], best_idx)
#                 # print(pre_score_count)
#                 # import pdb; pdb.set_trace()  
#                 end0.record()
#                 torch.cuda.synchronize()
#                 dt2 += start0.elapsed_time(end0)/1000
#                 # so-far-the-best
#                 for b in range(C.shape[0]):
#                     if best_s[b] > res[(W,0)].best_s[b]:
#                         res[(W,0)].best_s[b] = best_s[b]
#                         res[(W,0)].best_M[b] = best_models[b]
#                 # polish selected so-far-the-best with polish methods
#                 for p in polishes:
#                     record = False
#                     if p == 0:
#                         best_models = res[(W,0)].best_M
#                         record = True
#                     elif (m_batch == 0 or (m_batch+1) % 5 == 0): # apply polish sparsely
#                         if W.name  == 'GaU' or (W.name  == 'Oracle' and p == 'GaU') or (W.name  == 'GT' and (m_batch == 0 or p == 'GaU')):
#                             # print(W.name, p, m_batch)
#                             start0.record()
#                             best_models, best_s = local_optimization(data, res[(W,0)].best_M, p)
#                             end0.record()
#                             torch.cuda.synchronize()
#                             dtp += start0.elapsed_time(end0)/1000
#                             # Evaluate LO optimized so-far-the-best
#                             # end0.record()
#                             record = True
                    
#                     if record:
#                         start0.record()
#                         # compute test error of the best found models
#                         best_e, best_r, best_t = pose_error_batch_torch(best_models.unsqueeze(1), data)  # [B]
#                         best_e = best_e.squeeze(-1).cpu().numpy()
#                         best_r = best_r.squeeze(-1).cpu().numpy()
#                         best_t = best_t.squeeze(-1).cpu().numpy()
#                         end0.record()
                    
#                         mp = (W,p)
#                         res[mp].running_s += [res[mp].best_s + 0]  # copy
#                         res[mp].running_e += [best_e]
#                         res[mp].running_r += [best_r]
#                         res[mp].running_t += [best_t]
#                         res[mp].running_M += [res[mp].best_M.cpu().numpy()]
#                         torch.cuda.synchronize()
#                         dt3 += start0.elapsed_time(end0)/1000
#                 # Polish
#                 # if polish != 0 and W.name in ['GaU','Oracle'] and (m_batch == 0 or (m_batch+1) % 5 == 0): # only occasionally reoptimize current best_M
#                 #     start0.record()
#                 #     best_models, best_s = local_optimization(data, res[W].best_M, polish)
#                 #     # res[W].best_M = best_models # avoid repeatedly re-optimizing
#                 #     end0.record()
#                 #     torch.cuda.synchronize()
#                 #     dtp += start0.elapsed_time(end0)/1000
#                 #     # Evaluate LO optimized so-far-the-best
#                 #     start0.record()
#                 #     best_e, best_r, best_t = pose_error_batch_torch(best_models.unsqueeze(1), data)  # [B]
#                 #     # end0.record()
#                 # else:
#                 #     # Evaluate selected so-far-the-best
#                 #     start0.record()
#                 #     best_e, best_r, best_t = pose_error_batch_torch(res[W].best_M.unsqueeze(1), data)  # [B]
#                 #     # end0.record()
#                 #
#                 # best_e = best_e.squeeze(-1).cpu().numpy()
#                 # best_r = best_r.squeeze(-1).cpu().numpy()
#                 # best_t = best_t.squeeze(-1).cpu().numpy()
                
#                 # if polish == 0 and False:
#                 #     # DEBUG
#                 #     best_e1, best_r1, best_t1 = pose_error_batch(res[W].best_M.unsqueeze(1).cpu().numpy(), data)  # [B]
#                 #     assert(np.abs(best_r1 - best_r).max() < 0.1)
#                 #     assert(np.abs(best_t1 - best_t).max() < 0.1)
#                 #     assert(np.abs(best_e1 - best_e).max() < 0.1)
#                 # end0.record()
                
                
#         print(f'Timing: Solver: {dt1:3.2f}', f'Residuals: {dtS:3.2f}', f'Oracle: {dtg:3.2f}', f'Scoring: {dt2:3.2f}', f'Ploish: {dtp:3.2f}', f'Eval: {dt3:3.2f}')
#         if idx % 10 == 0:
#             print(idx)
#         # concatenate runing results
#         for mp in itertools.product(methods, polishes):
#             running_s = npstack(res[mp].running_s).swapaxes(0,1)
#             running_e = npstack(res[mp].running_e).swapaxes(0,1)
#             running_r = npstack(res[mp].running_r).swapaxes(0,1)
#             running_t = npstack(res[mp].running_t).swapaxes(0,1)
#             running_M = npstack(res[mp].running_M).swapaxes(0,1)
#             eval_results[mp].running_s += [running_s]
#             eval_results[mp].running_e += [running_e]
#             eval_results[mp].running_r += [running_r]
#             eval_results[mp].running_t += [running_t]
#             eval_results[mp].running_M += [running_M]
#         # if idx > 0:
#         #     break
#     # concatenate over batches
#     for mp in itertools.product(methods, polishes):
#         for k in eval_results[mp].keys():
#             if len(eval_results[mp][k]) > 0:
#                 eval_results[mp][k] = np.concatenate(eval_results[mp][k], axis=0)
#     return eval_results, files



# %%  _________Construct / Load models______________
print('________Construct / Load models___________')
max_distance = o.max_distance; N_bins = o.N_bins # filter to select models to evaluate
methods = []

# Add GU
W = ScoreWeightsGU(N_bins=N_bins, max_distance=max_distance, pow=1)
W.name = 'GaU'
W.cuda()
W.gen_hyperparams(o.val_thresholds)
M = W.score_matrix()
W.register_buffer('M', M)
WGU = W
methods += [W]

# # Add MSAC
W = ScoreWeightsMSAC(N_bins=N_bins, max_distance=max_distance, pow=1)
W.name = 'MSAC'
W.cuda()
W.gen_hyperparams(o.val_thresholds)
M = W.score_matrix()
W.register_buffer('M', M)
methods += [W]
WMSAC = W

# # Add MAGSAC++
W = ScoreWeightsMAGSAC(N_bins=N_bins, max_distance=max_distance, dof =o.MAGSAC_dof)
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
# TODO: Add loaded (learned) models

# Add Oracle and GT
methods += [Oracle(N_bins=N_bins, max_distance=max_distance, pow=1), MethodGT(N_bins=N_bins, max_distance=max_distance, pow=1)]

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
        # dataset = ResidualData(dataset_info, val_src, padding=True) # padding the 
        # dataset = H_dataset(dataset_info, val_src, padding=True)
        loader = create_loader(val_src)
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
        loader = create_loader(val_src)
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

if o.var or o.largeval:
    print('Variance Test')
    vt_scenes = val_scenes + test_scenes
    res = dict()
    # load or compute T results
    for val_src in vt_scenes:
        val_src_name = val_src.replace('/','_')
        var_file = res_root + val_src + '/val_T.pkl'
        if os.path.exists(var_file) and not o.recompute: # recompute flag off
            print(f'loading {var_file}')
            eval_results = load_object(var_file)
        else:
            print(f'__Multithreshold on {val_src}___________')
            torch.manual_seed(1)
            loader = create_loader(val_src)
            pairs_boud = 20000/len(vt_scenes) # select 20000 from all validation and test scenes, stratified per scene
            # if val_src in val_scenes:
            #     pairs_boud = 10000/len(val_scenes) # select 10000 from all validation scenes
            # else:
            #     pairs_boud = 10000/len(test_scenes) # and 10000 from all test scenes
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
    def expected_series(nn, sigmas, adjust_thresholds=False, subsample_MAGSAC=False, bootstrap_samples=1000):
        """
        nn - list of training set sizes
        sigmas - list of smoothing sigmas (experimental, =[0] no smoothing)
        subsample_MAGSAC # make the number of verified temperatures equal for the adjusted range
        bootstrap_samples = 5000 # how many times to re-draw the validation subset
        """
        print(f"Expected test error vs training set size {'adjusted' if adjust_thresholds else ''}")
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
            tv_scenes = val_scenes + test_scenes
            # e_val = np.concatenate([res[s][name].best_e for s in val_scenes], axis=0) # [B, T] (i. e. [image pair, threshold]) all validation pairs
            # e_test = np.concatenate([res[s][name].best_e for s in test_scenes], axis=0) # all test pairs
            # sample bootstrap subsets
            ee = []
            for s in tqdm(range(bootstrap_samples), desc=f"Bootstrap for method {M.name}"): #bootstrap samples
                # index = np.random.choice(N, val_samples, replace=False) # choose a validation subset
                # Leave-one-scene-out cross-validation
                # pick 2 validation scenes from tv_scenes at random:
                xval = np.random.choice(tv_scenes,2)
                xtest = list(set(tv_scenes) - set(xval))
                e_val = np.concatenate([res[s][name].best_e for s in xval], axis=0)
                # all the remaning scenes are test
                e_test = np.concatenate([res[s][name].best_e for s in xtest], axis=0)
                N = e_val.shape[0]
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
                    idx = np.random.choice(N, n, replace=True) # choose a validation subset
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
                            if subsample_MAGSAC: # make the number of points tried approximatelly equal only evry 3rd is retained
                                V[5::3] = 1000
                                V[6::3] = 1000
                        else:
                            V[tt>o.max_distance/3] = 1000
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
    eres_s = expected_series(nn, sigmas, bootstrap_samples=5000)
# %% 
if o.var and tune_sigma: # Experimental: smooth validation plot before selecting best value
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
if o.var: # Statisitcal analyzis of mean and variance of test performance w.r.t. training set size, unajusted
    nn = 2**np.arange(1,13,1)
    bootstrap_samples=1000
    eres = expected_series(nn, [0], bootstrap_samples=bootstrap_samples)
# %%    
if o.var: # Statisitcal analyzis of mean and variance of test performance w.r.t. training set size, unajusted    
    eres_adjusted = expected_series(nn, [0], adjust_thresholds=True, bootstrap_samples=bootstrap_samples, subsample_MAGSAC=False)
    # eres_s2 = expected_series(nn, [0], adjust_thresholds=True, subsample_MAGSAC=True)
    # eres_smooth = expected_series(nn, [0]) # repeatability

# %% 
if o.var: # plotting E[e] versus training set size
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
            vs = eres_adjusted[M.name].expected            
            cci_s = eres_adjusted[M.name].std_ci
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
    plt.grid(axis='y', which='major', linestyle='-', alpha=0.3)
    plt.xscale('log', base=2)
    plt.xlabel('Validation set size')
    # plt.gca().set_xticklabels(nn)
    plt.ylabel(r'$\rm\mathbb{E}[e_R]$')
    # plt.title(f'Expected median test pose error vs. validation set size, {dataset_info.name}')
    outf = res_root + f'training_size_e_mean.pdf'
    savefig(outf)
    plt.show()
    plt.close(f)

# %% 
if o.var: # plotting std[e] versus training set size
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
            v_s = eres_adjusted[M.name].std
            cci_s = eres_adjusted[M.name].std_ci
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
    plt.xlabel('Validation set size')
    plt.grid(axis='y', which='major', linestyle='-', alpha=0.3)
    # plt.gca().set_xticklabels(nn)
    plt.ylabel(r'${\rm std}[e_R]$')
    # plt.ylim([0,2])
    plt.yscale('log')
    # plt.title(f'Std of test error with respect to random trining set, smoothed, {dataset_info.name}')
    plt.draw()
    outf = res_root + f'training_size_e_std.pdf'
    savefig(outf)
    plt.show()
    plt.close(f)

# %%
def err_vs_T(scenes = [], val_samples=None, bootstrap_samples = 100):
    """
    scenes -- list of scenes
    val_samples -- number of samples to use for validation plots vs T, default None -- use all samples of the given scenes
    """
    print('Measuring Large Set Validation Error vs Temperature')
    # MM = [M for M in methods if not(isinstance(M, MethodGT) or hasattr(M, 'locked'))]
    MM = [M for M in methods if M.name in res[scenes[0]].keys()]
    # MM = [methods[-2]]
    eres = {M.name:dotdict() for M in MM}
    for M in MM:
        name = M.name
        e_val = np.concatenate([res[s][name].best_e for s in scenes], axis=0) # [B, T] (i. e. [image pair, threshold]) all results for method M
        if val_samples is None:
            val_samples = e_val.shape[0]
        elif val_samples > e_val.shape[0]:
            raise RuntimeError("Not enough validation samples in the given scenes")

        # e_test = np.concatenate([res[s][name].best_e for s in test_scenes], axis=0)
        N = e_val.shape[0]
        VV = []
        Vm = []
        for s in tqdm(range(bootstrap_samples), desc=f"Bootstrap samples for {name}"): #bootstrap samples
            idx = np.random.choice(N, val_samples, replace=True) # choose a validation subset                
            if e_val.ndim ==2:
                V = np.median(e_val[idx],axis=0) # [T] we use median pose error as the criterion, here median over the selected validation subset
                Vmin = np.median(np.min(e_val[idx],axis=-1)) # median error of best temperature per image
            else:
                V = Vmin = np.median(e_val[idx],axis=0)
            VV += [V]
            Vm += [Vmin]
        # average and std of validation errors over bootstrap samples
        VV = np.vstack(VV) # [bootstrap_samples, T]
        Vm = np.vstack(Vm)
        EV = np.mean(VV, axis=0) # [T]
        stdV = np.std(VV, axis=0) # [T]
        if hasattr(M,'hyperparams'):
            t_best_idx = np.argmin(EV) # select best hyperparameter
            eres[name].best_t_idx = t_best_idx
            eres[name].best_t = M.hyperparams[t_best_idx]
        else:
            eres[name].best_t_idx = None
            eres[name].best_t = None
        eres[name].val_errors_mean = EV
        eres[name].val_errors_std = stdV
        eres[name].val_errors_min = np.mean(Vm)
        eres[name].val_errors_min_std = np.std(Vm)
    return eres
    
# %%
if o.largeval:
    # print("Computing validation erorrs vs T")
    eres_large = err_vs_T(scenes = val_scenes + test_scenes)
    # eres_large = err_vs_T(scenes = val_scenes, val_samples=2000)

#%% 
if o.largeval:
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
        bot = 10        
        for (i,M) in enumerate(methods):
            if M.name == 'GT':
                continue
            if 'gamma=30.0' in M.name:
                continue
            ax = ax1
            has_hyperparam = hasattr(M, 'set_hyperparam') and not (hasattr(M, 'locked') and M.locked) # chose hyperparam only if has_hyperparam, not locked
            # if not has_hyperparam:
                # continue
            r = eres_large[M.name]
            v_besti = r.best_t_idx
            stdV = r.val_errors_std
            v = r.val_errors_mean
            t_best = r.best_t
            style = '-'
            label=M.name
            if has_hyperparam:
                M.set_hyperparam(t_best)
                M.best_hyperparam = t_best
                M.hbest_i = v_besti
                x = M.hyperparams
                xb = M.best_hyperparam
                name = M.name + r' ($\tau{=}'+f'{M.best_hyperparam:2.2f}$)'
                name = name.replace('gamma=30.0', '').replace('gamma=10.0', '')
                label = name
                ax.plot(xb, v[v_besti], 'o', label=None, color=cc[i])
                ax.plot(x, v, style, label=label, color=cc[i])
                ax.fill_between(x, v - stdV, v + stdV, alpha=0.3, color=cc[i])
                ax.axhline(r.val_errors_min, color = cc[i], linestyle=":", label=None)
                ax.fill_between(x, r.val_errors_min - r.val_errors_min_std, r.val_errors_min + r.val_errors_min_std, alpha=0.3, color=cc[i])
            else: # no hyperparameter to tune: for example GT or Oracle
                ax.axhline(v, label=label, color = 'k', linestyle=style)
            bot = min(bot, v[v_besti])
                
        plt.legend()
        ax1.legend(loc=1)
        # bot = v[v_besti]*0.99 - 0.1
        # bot = max(0, bot*0.99 - 0.1)
        if False:
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
        ax1.set_xlabel('Hyperparameter $\\tau$ [px]')
        ax1.locator_params(axis='x', nbins=10)
        plt.gca().yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())            
        ax1.locator_params(axis='y', nbins=10)
        ax1.set_ylabel(f'{stat.__name__} R error')
        plt.draw()
        # Path('fig').mkdir(exist_ok=True)
        fig_path = results_path + f'/fig/'
        # force_path(fig_path)
        savefig(fig_path + f'validation.pdf')
        plt.show()
        plt.close(fig)

# %%
# %%
if o.largeval:
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
    plt.xlim(0,10)
    plt.xlabel('Residual [px]')
    plt.show()
    plt.draw()
    force_path(fig_path)
    savefig(fig_path + f'validation_kernels.pdf')

    force_path(val_file)
    save_object(val_file, methods)
    
# %%
