# %%
import os, sys
if  __name__ == "__main__":
    __name__ = 'score_learn.train'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
else:
    __run__ = False
import timeit
from argparse import ArgumentParser
from types import SimpleNamespace
from collections import namedtuple
from scipy.special import gamma, gammainc, gammaincc
import scipy
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch
import shlex



### !%load_ext autoreload
### !%autoreload 2
from .load_data import *
from .score_weights import *
from .drawing import *
#!%matplotlib inline

op = ArgumentParser()
op.set_defaults(batch_size=32)
op.add_argument("-l", type=str, default='None', help="ML, softmax_mask, softmax_subset, logreg")
op.add_argument("--N_bins", type=int, default=500)
op.add_argument("--max_distance", type=float, default=10.0, help="max distance for the histogram")
op.add_argument("--reg", type=float, default=0.0, help="max distance for the histogram")
op.add_argument("--max_batches", type=int, default=None, help ="None uses all batches of the dataset, can specify fewer batches for a fast check" )
op.add_argument("--epochs", type=int, default=None, help ="training epochs")
op.add_argument("--lr", type=float, default=0.1, help ="learning rate")
op.add_argument("--load", action="store_true", default=False, help="continue training from last saved model")
op.add_argument("-T", type=int, default=10, help ="multiple hyperparams (thresholds)")
op.add_argument("--sqrt", action="store_true", default=True, help="take square root of residuals if True")
#op.add_argument("-F", action="store_true", default=False, help="Fundamental matrix case, no extra scaling")

# RootSIFT
op.add_argument("--data", type=str, default='PhotoTourismRootSIFT', help="dataset")
# SP+SG
# op.add_argument("--data", type=str, default='PhotoTourismSPSG', help="dataset")

## ETH3D
# op.add_argument("--data", type=str, default="eth3d/", help="path to data") #
# op.add_argument("--modelpath", type=str, default='models/eth3d/', help="path for saving models")

# # scannet
# op.add_argument("--data", type=str, default="scannet/train", help="path to data") #
# op.add_argument("--modelpath", type=str, default='models/scannet/', help="path for saving models")
# #

# # KITTI setup
# op.set_defaults(batch_size=8)
# op.add_argument("--data", type=str, default="KITTI", help="dataset")

#
# python score_weights_script.py -l logreg --reg 0.0 --N_bins 200 --max_batches=2 --epochs 100 --lr 0.5 --load
# python score_weights_script.py -l softmax_mask --reg 0.0 --max_batches 2 --epochs 100 --N_bins 200 --load
#
#

args_str = ' '.join(sys.argv[1:])
ops, args = op.parse_known_args(shlex.split(args_str))
o = SimpleNamespace(**vars(ops))
print(o)
N_bins = o.N_bins
max_distance = o.max_distance
batch_size = o.batch_size
max_batches = o.max_batches
pow = 1 if o.sqrt else 2
o.sqrt = True

for dataset_info in datasets:
    if dataset_info.name == o.data:
        break
if dataset_info.name != o.data:
    print(f'cannot find dataset {o.data}')
    exit(1)

train_scene = dataset_info.train[0]
res_root = f'results/{dataset_info.name}/'
model_path = res_root + 'models/'
o.modelpath = model_path

force_path(o.modelpath)

residual_dataset = ResidualData(dataset_info, train_scene, padding=True)  # padding the
# residual_dataset = ResidualData(src, padding=True, sqrt=True, size=3000, F = o.F) # padding the residuals with infs
data_loader = torch.utils.data.DataLoader(residual_dataset,batch_size=batch_size,num_workers=0,shuffle=True)


# %%
W1 = ScoreWeightsMAGSAC(maximum_threshold=3, N_bins=N_bins, max_distance=max_distance)
W2 = ScoreWeightsTZ(N_bins=N_bins, max_distance=5, gamma=None, threshold_scale = 3.64, maximum_threshold=3)
plt.figure()
for W in [W1,W2]:
    w = W.score_weights_normalized()
    xx = (torch.arange(w.shape[0], device=w.device)/w.shape[0]*W.max_distance**pow)**(1/pow)
    plt.plot(xx.cpu(), w.cpu().detach())
plt.draw()

# %%
#
counts, hh_bins, n_inside, n_points = sufficient_statistic_GT(data_loader, N_bins=N_bins, max_distance=10, max_batches=max_batches, pow=pow)
# binc = (hh_bins[:-1] + hh_bins[1:])/2

# %%
# plt.plot(binc, torch.log(hh+1e-10), label='hist')
plt.figure()
hh_bins = hh_bins.cpu()
f = (counts / n_points).cpu()
plt.plot(hh_bins[:-1]**(1/pow), f, label='hist')
# f_last = f[50]
# p_in = 1 - f_last/f
# plt.plot(hh_bins[:-1], p_in, label='p(in|r)')
# plt.plot(-hh_bins[1:], counts / n_inside, label='hist')
# counts, hh_bins, n_inside, n_points = sufficient_statistic_GT(data_loader2, N_bins, max_distance=max_distance, max_batches=200)
# plt.plot(hh_bins[:-1], counts / n_inside, label='hist')
# plt.plot(-binc, hh, label='hist')
plt.yscale('log')
plt.legend()

# %%
# assert(False)
# %%
from . import local_optimization as LO
import kornia 
# %%
if False:
    counts = 0
    n_inside = 0
    n_points = 0
    for idx, data in enumerate(data_loader):
        # filter out only GT model correspondances
        R = data['gt_R']
        T = data['gt_t']
        M = R.clone().detach()
        for b in range(R.shape[0]):
            M[b] = LO.compose_essential_matrix(R[b], -T[b].flatten())
            R1, R2, t = kornia.geometry.epipolar.decompose_essential_matrix(M[b])
            M[b] = LO.compose_essential_matrix(R1[0], -t.flatten())
        M = M.unsqueeze(1)
        #
        # M = data['models'][:, -1:, ...]
        # M = data['models'][:, 1:2, ...]
        # tr = M.diagonal(dim1=-1, dim2=-2).sum(-1)
        # M = M / tr.unsqueeze(-1).unsqueeze(-1)
        data['models'] = M
        compute_residuals(data)
        R = data['residuals']
        R = R[:, -1:, :]  # select GT model only [B 1 N]
        H, n_points_H, bins = hist_values(R, N_bins, -max_distance, max_distance)
        cc = H * n_points_H
        # total counts from all images
        counts = counts + cc.flatten(start_dim=-2, end_dim=-1).sum(dim=0)

    hh_bins = bins.cpu().numpy()
    counts = counts.cpu().numpy()

    plt.plot(hh_bins[:-1], counts, label='hist')
    plt.plot(-hh_bins[:-1], counts, label='hist')
    plt.yscale('log')

# %%
# assert(False)

#%%
#assert (False)
# %%

def ML_loss(ss, W:ScoreWeightsMonotone):
    """
    errors [*, M] -- errors of M models to compare (lower the better)
    R [*, M, n] -- residuals
    W: ScoreWeightsMonotone or derived
    """
    #scores = W.score_hist(data.H, data.n_points)
    scores = W.score_ss(ss)
    loss = -scores
    return loss

import time

hash = dict()

def learn_ML(data_loader, W: ScoreWeightsMonotone, lr=0.1, epochs=500, plots=False, max_batches=None):
    # Test training
    optimizer = torch.optim.Adam(W.parameters(), lr=lr)
    W.to('cuda')
    LL = []
    #
    key = (W.N_bins, W.max_distance, W.pow)
    if key not in hash:
        start = time.time()
        counts, hh_bins, n_inside, n_points = sufficient_statistic_GT(data_loader, W.N_bins, max_distance=W.max_distance, max_batches=max_batches, pow=W.pow)
        n_out = (n_points-n_inside).to('cuda')
        end = time.time()
        print(f"Time of sufficient statistics:{end-start:3.2f}s")        
        ss = SimpleNamespace(counts=counts.to('cuda'), n_inside=n_inside, n_points=n_points.to('cuda'), n_out=n_out)
        hash[key] = ss
    else:
        ss = hash[key]
    # counts, hh_bins, n_inside, n_points = sufficient_statistic_GT(data_loader, W.N_bins, max_distance=W.max_distance, max_batches=max_batches, pow=W.pow)
    start = time.time()
    W.P_out = 1 - ss.n_inside / ss.n_points
    W.P_in = ss.n_inside / ss.n_points
    #
    uni_fraction = (ss.counts[-50:].mean()*W.N_bins)/ss.n_inside
    print(f"Uni fraction={uni_fraction.item()}")
    W.uni_fraction = uni_fraction
    # n_points = n_points.to('cuda') # [B]
    # H = counts.to('cuda') / n_points # for some reason the convention
    # H, n_points = hist_values(R, W.N_bins, W.max_distance, loop=True)
    # data = SimpleNamespace(H=H, n_points=n_points)
    for i in range(epochs):
            L = ML_loss(ss, W) / ss.n_points
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            LL.append(L.item())

    end = time.time()
    print(f'L: {L.item():3.5f}' )
    print(f"Time of ML fitting:{end-start:3.2f}s")
    # fig, ax = plt.subplots(1, 2, sharey=False, figsize=(5*4,3))
    # ax[0].plot(LL)
    # ax[0].set_title('Loss')
    # W.p_out = 1 - W.p_in
    
def learn_ML2(loader, W: ScoreWeightsMonotone, lr=0.1, epochs=500, plots=False, max_batches=None):
    # Test training
    optimizer = torch.optim.Adam(W.parameters(), lr=lr)
    W.to('cuda')
    LL = []
    for epoch in range(epochs):
        L = 0
        n = 0        
        for idx, data in enumerate(loader):
            E = data['models'][:,-1:].cuda() # GT model [B 1,3,3]
            K1 = data['K1'].to(E) # [B,3,3]
            K2 = data['K2'].to(E)  # [B,3,3]
            compute_residuals(data)
            R = data['residuals'][:,-1:,:].cuda().float() # residuals of GT model [B,1,N]
            estimate_distance = 50
            vol = estimate_volume(E, K1, K2, n_points=100_000, repeats=1, estimate_distance=estimate_distance) # [B]
            #
            
            # vol_bin = vol / estimate_distance * W.bin_size() # (x,y) volume per bin [B, M]
            # logV_bin = torch.log(vol_bin)  # [B 1]
            # vol_out = 1 - vol_bin*W.N_bins # (x,y) volum for r> max_distance
            # logV_out = torch.log(vol_out)  # [B 1]
            # #
            # ss = sufficient_statistic(R, W.N_bins, max_distance=W.max_distance)  # [B 1000 K]
            # #
            # n_in = ss.n_inside.squeeze(-1)  # [B 1]
            # n_pts = ss.n_points.squeeze(-1)  # [B 1]
            # n_out = n_pts - n_in
            # #
            # w = W.score_weights().float()
            # W1 = torch.cat([w.view([1,-1]) + logV_bin, W.log_p_out + logV_out], -1)
            # logZ = torch.logsumexp(W1, dim=-1, keepdim=True) # [B,1]
            # scores = ss.counts @ w + n_out*W.log_p_out  - logZ*n_pts # [B, 1]
            ss = sufficient_statistic(R, W.N_bins, max_distance=W.max_distance, pow= W.pow)  # [B 1 K]
            scores = W.score_vol(ss, vol, estimate_distance)
            #
            l = -scores.mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            L += l.item()
            n += R.shape[0]
            if max_batches is not None and idx > max_batches:
                break
        L/=n
        LL += [L]
        print(f'L: {L:3.5f}')
        #
        file_name = o.modelpath + W.name + '.pkl'
        torch.save(W, file_name)
        #
        f = plt.figure(num='compare_weights', figsize=(7, 3))
        ax = plt.gca()
        w = W.score_weights().float()
        p_in = (1 - torch.exp(w[-1]-w)).detach().cpu().numpy()
        wmin = w.min()
        s = w.max()-wmin
        w = ((w - wmin)/s).detach().cpu().numpy()
        x = torch.arange(W.N_bins)/W.N_bins*W.max_distance
        ax.plot(x, w, '-', zorder=12, clip_on=False, label='norm weights')
        ax.plot(x, p_in, '-', zorder=12, clip_on=False, label='prob. inlier')
        plt.tight_layout()
        plt.draw()
        plt.savefig(o.modelpath + W.name + '.pdf')
        plt.close(f)
    return LL

# %% --------------------------------------------------------------

def softmax_loss(data, W, tau_S = 1.0, mask=False):
    """
    errors [*, M] -- errors of M models to compare (lower the better)
    scores [*, M] -- scores of M models to compare (higer the better)
    n_points [*] -- number of poitns for the score histogram
    tau_S -- smoothnig parameter, higher is smoother
    """
    errors = data.E
    # wmax = W.score_weights()[0]
    scores = W.score_hist_normalized(data.H, data.n_points)
    #/data.n_points.float().mean()
    # scores = W.score(H, n_points) / wmax
    # scores = W.score(R) / wmax
    # scores = scores - scores.max(dim=-1,keepdim=True)[0].detach()
    # sample 500 at random
    M = errors.shape[-1]
    B = errors.shape[0]
    #   I = torch.arange(M, device = scores.device)
    #   I = torch.randperm(M, device = scores.device)
    #   I = I[0:(M+1)//2]
    #   I = torch.arange(M-1, device = scores.device) # exclude GT model
    errors = errors.clamp(max=30.0) # do not care for selecting amongst bad models
    scores = scores.unsqueeze(-2).expand(B,M,M)
    errors = errors.unsqueeze(-2).expand(B,M,M)
    if not mask:
        # random subset sampling
        subM = M//2
        unif = torch.ones_like(scores).view(-1,M)
        idx = unif.multinomial(subM, replacement=False).view([B,M,subM])
        scores = scores.gather(-1, idx).squeeze(-1)
        errors = errors.gather(-1, idx).squeeze(-1)
        loge = (errors.double() + 1e-10).log()
        scores = scores.double()
        loss = torch.logsumexp(scores/tau_S + loge, dim=-1) - torch.logsumexp(scores/tau_S, dim=-1)
        loss = loss.exp() - errors.min(dim=-1)[0] # excess loss
    else:
        # random mask sampling
        drop_prob = 1 - torch.exp(-errors/10) # with hight probability keep good models
        drop = (torch.ones_like(scores)*drop_prob).bernoulli()
        drop[:,:,-1] = 1 # drop GT model
        loge = (errors.double() + 1e-10).log()
        log_drop = (-30*drop).double()
        scores = scores.double()
        loss = torch.logsumexp(scores/tau_S + log_drop + loge, dim=-1) - torch.logsumexp(scores/tau_S + log_drop, dim=-1)
        loss = loss.exp() - (errors + drop*100).min(dim=-1)[0]
    # I = torch.randperm(M-1, device = scores.device)
    # I = I[0:M//2] # half of models at random
    # scores = scores[...,I]
    # errors = errors[...,I]

    # p = torch.softmax(scores.double()/tau_S, dim=-1)
    # errors = errors.double()
    # loss = ((errors * p).sum(dim=-1) - errors.min(dim=-1)[0]) # excess loss
    return loss.mean(dim=-1)

def softmax_loss_mask(data, W, tau_S = 1.0):
    return softmax_loss(data, W, tau_S = tau_S, mask=True)


def softmax_loss_subset(data, W, tau_S=1.0):
    return softmax_loss(data, W, tau_S=tau_S, mask=False)

def logreg_loss(data, W, tau_S=1.0):
    """
    errors [*, M] -- errors of M models to compare (lower the better)
    R [*, M, n] -- residuals
    tau_S -- smoothnig parameter, higher is smoother
    """
    errors = data.E
    w = W.score_weights().float()
    scores = data.H.to(w) @ w
    num_points = data.n_points.view([-1] + [1]*(scores.dim()-1))
    scores = scores * num_points / 10  # arbitrary scaling
    # scores = W.score_hist_normalized_bias(data.H, data.n_points) 
    # print(scores.shape)
    assert (errors.shape == scores.shape)
    EE1 = errors.unsqueeze(-1) # e_i
    # EE1 = torch.log(EE1 + 0.001) # rescale the errrors to give more focus to accurate models
    EE1 = EE1.clip(max=30) # rescale the errrors to give more focus to accurate models
    SS = -F.logsigmoid(scores.unsqueeze(-2) - scores.unsqueeze(-1)) # log(1+ e^(s_j-s_i))
    # SS = torch.sigmoid((scores.unsqueeze(-2) - scores[..., I].unsqueeze(-1))/tau_S)
    loss = (SS*EE1).mean(dim=-1) # sum_j
    return loss

def learn_epoch(loss_fn, loader, W: ScoreWeightsMonotone, tau_S=1.0, max_batches=None):
    W.to('cuda')
    L = 0
    n = 0
    for idx, data in enumerate(loader):
        R = data['residuals']
        R = R[:,:-1,:].cuda() # throw away GT model [B M N] (learnnig from a true sample during ransac)
        E = data['errors'][:, :-1].cuda()
        H, n_points, bins = hist_values(R, W.N_bins, 0, W.max_distance)
        data = SimpleNamespace(H=H, n_points=n_points, E=E)
        l = loss_fn(data, W, tau_S = tau_S).mean()
        reg = W.regularizer()
        l = l + o.reg*reg
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        L += l.item()
        n += R.shape[0]
        if max_batches is not None and idx > max_batches:
            break
    print(f'L: {L:3.5f}' )
    return L/n

# %%
def compare_weights(*WW, density=True, name = None, range = max_distance):
    f = plt.figure(num='compare_weights', figsize=(7, 3))
    ax = plt.gca()
    markers = 'oxsd^<v'
    s = None
    for (i,W) in enumerate(WW):
        if hasattr(W, 'o'):
            param = f' ($\\gamma={(W.gamma.item())*100:3.1f}$%)'
        else:
            param = f' ($\\tau={W.max_distance}$)'
        label = W.name + param
        if W.density is True:
            w = W.r_density().detach().cpu() # categorical over N_bins
            # w = torch.log_softmax(w, dim=-1)
            x = torch.arange(w.shape[0], device = w.device)/w.shape[0]*W.max_distance**pow
            # w = w + torch.log(2*x**0.5) # density change of variables
            # extend last value to max_distance
            x = torch.cat((x, x[-1:], torch.tensor([range**pow])), 0)
            # w_end = (-math.log(W.max_outlier_dist) + torch.log(1-W.get_gamma())).to(w).view([1]).detach()
            w_end = W.w_end().detach().to(w).view([1])
            w = torch.cat((w, w_end,w_end),0)
            if density:
                w = w.exp()
            else:
                if i==0:
                    # normalzie to 0,1
                    wmin = w.min()
                    s = w.max()-wmin
                    # wmin = 0
                    # s = 1
                w = (w -wmin)/s
            p = ax.plot(x**(1/pow), w, '-', color=cc[i], label=label, zorder = 12)
            log_p_in, _ = W.log_likelihood_inliers()
            p = ax.plot(x[:-2]**(1/pow), log_p_in.detach().cpu().exp(), '--', color = cc[i], label='Inliers', zorder=12) #, color=p[0].get_color())
        else:
            w = W.score_weights_normalized().detach().cpu() # categorical over 
            w = w-w.min()
            w = w/w.max()
            x = torch.arange(w.shape[0], device = w.device)/w.shape[0]*W.max_distance**pow
            p = ax.plot(x**(1/pow), w, '-', label=label, zorder=2)
            # p_in = W.p_inlier1().detach().cpu()
            # ax.plot(x, p_in, '--', label='p(in|r)', zorder=2, color=p[0].get_color())
        # p_in = W.p_inlier().clip(min=1e-3).detach().cpu()
        # ax.plot(x, p_in, '--', label='p(in|r)', color=p[0].get_color(), zorder=2)
        # w = (W.monotone_w() - W.logZ_inliers()).exp().detach().cpu()
        # w = w-w.min()
        # w = w/w.max()
        # ax.plot(x, w, ':', label=None, zorder=2, color=p[0].get_color())
    # add hist
    hh = (counts / n_inside).cpu()
    # hh = counts/counts.sum() * n_inside / n_points
    # hh = counts/n_points
    N_bins =hh.shape[-1]
    bin_size = max_distance**pow/N_bins
    w = np.log(hh+1e-10) - math.log(bin_size)
    x = torch.arange(N_bins, device = w.device)/N_bins*max_distance**pow
    # w = w + torch.log(2*x**0.5) # density change of variables
    if density:
        w = w.exp()
    else:
        if s is None:
            wmin = w.min()
            s = w.max()-wmin
        w = (w -wmin)/s
    ax.plot(x**(1/pow), w, '-k', label='Histogram', linewidth = 0.5, zorder = 11, clip_on=False)
    ax.legend()
    if density:
        ax.set_yscale('log')
        ax.set_ylabel('density')
        ax.set_ylim(bottom=1e-3)
    else:
        pass
        # ax.set_yscale('log')
        # ax.set_ylim(-0.3,1)
        # ax.set_ylabel('normalized scale')
    ax.set_xlim(left=0, right=range)
    ax.set_xlabel('residual [px]')
    if name is not None:
        plt.tight_layout()
        plt.draw()
        plt.savefig('fig/' + name +'.pdf')
        plt.savefig(model_path + 'density' + name +'.pdf')
    plt.show()
    plt.close(f)

# %%
try:
    compare_weights(*WW, density=True, range=10, name='ML_comparison')
except NameError:
    pass

# %% pretraining

# load or train ML

def W_name(W, method, o):
    # name = f'{method}_tau={W.max_distance}_bins={W.N_bins}'
    name = f'{method}_tau={W.max_distance}_bins={W.N_bins}'
    if method!='ML':
        name += f'_reg={o.reg}'
    if hasattr(W, 'M'):
        name += f'_mult'
    elif hasattr(W, 'gamma'):
        name += f'_gamma={(W.gamma.item())*100:3.1f}'
    return name

def load_W(fname):
    W = torch.load(fname)
    W.name = fname.replace('.pkl', '').replace(o.modelpath, '')
    return W


def load_or_train_ML(tau, N_bins=o.N_bins, gamma=None, ModelClass=ScoreWeightsMonotoneMix):
    # fname = f'{o.modelpath}ML_tau={tau}_bins={o.N_bins}_SP.pkl'
    # W = load_W(fname)
    # W = ScoreWeightsCat(N_bins=N_bins, max_distance=tau, max_outlier_dist = 100.0)
    # W = ScoreWeightsMonotoneSpherical(N_bins=N_bins, max_distance=tau, monotone=True, gamma=0.6, max_outlier_dist = 100.0)
    # W = ScoreWeightsUnconstrained(N_bins=N_bins, max_distance=tau, max_outlier_dist = 100.0)    
    # W = ScoreWeightsMonotone(N_bins=N_bins, max_distance=tau, max_outlier_dist = 100.0, pow=pow)
    W = ModelClass(N_bins=N_bins, max_distance=tau, monotone=True, gamma=gamma, max_outlier_dist = 100.0, pow=pow)

    Wfname = W_name(W, 'ML', o)
    W.name = Wfname
    file_name = o.modelpath + Wfname + '.pkl'
    if os.path.exists(file_name) and False:
        W = torch.load(file_name)
    else:
        print(f'Learning ML max_distance={tau}')
        learn_ML(data_loader, W, lr=0.1, epochs=500, max_batches=o.max_batches)  # ML
        W.o = o
        w = W.score_weights_normalized().detach().cpu().numpy()
        # np.savez(o.modelpath + Wfname, weight=w, tau=W.max_distance, max_distance=W.max_distance, gamma=W.get_gamma().item())
        # np.savez(o.modelpath + Wfname, weight=w, tau=W.max_distance, max_distance=W.max_distance)
        # print((W.uniform_in - W.logZ_density() + W.log_Vmax).exp())
        # print((W.monotone_w() - W.logZ_density() + W.log_dV).exp().sum())
        torch.save(W, file_name)
    return W 

# %%
WW = []
# %%
if False:
    W = ScoreWeightsTZ(N_bins=N_bins, max_distance=max_distance, gamma=None, max_outlier_dist=100.0, pow=pow)
    learn_ML(data_loader, W, lr=0.1, epochs=500, max_batches=o.max_batches)  # ML
    compare_weights(W, density=True, range=10, name='ML_comparison')
    Wfname = W_name(W, 'TZ', o)
    W.name = Wfname
    file_name = o.modelpath + Wfname + '.pkl'
    torch.save(W, file_name)

# %%
if o.l == 'ML':
    W = ScoreWeightsMonotoneMix(N_bins=N_bins, max_distance=max_distance, monotone=True, gamma=0.1, max_outlier_dist=100.0, pow=pow)
    W.gen_hyperparams(o.T)
    W.o = o
    M = []
    for t in W.hyperparams:
        W.__init__(N_bins=N_bins, max_distance=max_distance, monotone=True, gamma=0.1, max_outlier_dist=100.0, pow=pow)
        W.set_hyperparam(t)
        print(f'Learning with gamma={t}')
        learn_ML(data_loader, W, lr=0.1, epochs=500, max_batches=o.max_batches)  # ML
        M += [W.score_weights_normalized()]
        WW += [copy.deepcopy(W)]
    M = torch.stack(M).detach()
    W.register_buffer('M',M)
    #   
    Wfname = W_name(W, 'ML', o)
    W.name = Wfname
    file_name = o.modelpath + Wfname + '.pkl'
    torch.save(W, file_name)
    print('Trained mult-gamma ML models')
    # thresholds = [10.0, 3.0, 2.0, 1.5]
    # # thresholds = [5, 10, 20]
    # compare_weights(*WW, density=True, range=11, name='ML_comparison')
    # gammas = np.linspace(0.01, 0.99, 20, endpoint=True)
    # WW[0].gen_hyperparams(20)
    # M = WW[0].score_matrix()
    M = W.M
    plt.figure()
    xx = (torch.arange(M.shape[1], device=M.device)/M.shape[1]*W.max_distance**pow)**(1/pow)
    for i in range(M.shape[0]):
        plt.plot(xx.cpu(), M[i].cpu().detach())
    plt.draw()
    plt.savefig(o.modelpath + W.name + '-family.pdf')
    plt.show()

# %%
thresholds = [o.max_distance]*2
NN_bins = [o.N_bins]*2
# gammas = [0.1, 0.3, 0.5]
gammas = [0.1, 0.3]
WW = []
for (tau, N_bins, gamma) in zip(thresholds, NN_bins, gammas):
    W = load_or_train_ML(tau, N_bins=N_bins, gamma=gamma, ModelClass = ScoreWeightsMonotoneMix)
    W.name = 'Learned Mixture'
    WW += [W]
compare_weights(*WW, density=True, range=10, name='ML_comparison') 
print('Trained gamma ML models')

# %%
plt.figure()
W = WW[-1]
W.gen_hyperparams(o.T)
for t in W.hyperparams:
    W.set_hyperparam(t)
    w = W.score_weights_normalized()
    xx = (torch.arange(w.shape[0], device=w.device)/w.shape[0]*W.max_distance**pow)**(1/pow)
    plt.plot(xx.cpu(), w.cpu().detach())
plt.draw()
plt.savefig(o.modelpath + W.name + '-family.pdf')
plt.show()

# %%
# thresholds = np.linspace(0.1, 10, 20, endpoint=True)
W = ScoreWeightsMAGSAC(maximum_threshold=3, N_bins=N_bins, max_distance=max_distance)
W.gen_hyperparams(o.T)
M = W.score_matrix()
plt.figure()
for i in range(M.shape[0]):
    plt.plot(xx.cpu(), M[i].cpu().detach())
plt.draw()
plt.savefig(o.modelpath + 'magsac-family.pdf')
plt.show()
torch.save(W, f'{o.modelpath}magsac.pkl')

# %%
# W = ScoreWeightsTZ(N_bins=N_bins, max_distance=max_distance, gamma=None, max_outlier_dist=100.0, threshold_scale = 3.7)
# W.name = 'TZ'
# W.gen_hyperparams(o.T)
# M = W.score_matrix()
# plt.figure()
# for i in range(M.shape[0]):
#     plt.plot(xx.cpu(), M[i].cpu().detach())
# plt.draw()
# plt.savefig(o.modelpath + 'TZ-family.pdf')
# plt.show()
# torch.save(W, f'{o.modelpath}TZ.pkl')


# %%
# W = ScoreWeightsMSAC(tau=3, N_bins=N_bins, max_distance=max_distance)
# W.gen_hyperparams(o.T)
# M = W.score_matrix()
# plt.figure()
# for i in range(M.shape[0]):
#     plt.plot(xx.cpu(), M[i].cpu().detach())
# plt.draw()
# plt.savefig(o.modelpath + 'msac-family.pdf')
# plt.show()
# torch.save(W, f'{o.modelpath}msac.pkl')

# %%
W = ScoreWeightsRANSAC(tau=3, N_bins=N_bins, max_distance=max_distance)
W.gen_hyperparams(o.T)
M = W.score_matrix()
plt.figure()
for i in range(M.shape[0]):
    plt.plot(xx.cpu(), M[i].cpu().detach())
plt.draw()
plt.savefig(o.modelpath + 'ransac-family.pdf')
plt.show()
torch.save(W, f'{o.modelpath}ransac.pkl')

# WWM = []
# for tau in Mthresholds:
#     if tau <= max_distance:
#         w = W.score_weights_normalized().numpy()
#         np.savez(f'{o.modelpath}magsac_tau={W.maximum_threshold}', weight=w, tau=W.maximum_threshold, max_distance=W.max_distance)
#         torch.save(W, f'{o.modelpath}magsac_tau={W.maximum_threshold}.pkl')

# %%
#exit(0)
assert (False)
# %%    
if o.l == 'ML':
    thresholds = [10.0, 3.0, 2.0, 1.5]
    NN_bins = [o.N_bins] * 5
    gammas = [None] * 5
    # thresholds = [80.0]
    if o.l == 'ML':
        for (tau, N_bins, gamma) in zip(thresholds, NN_bins, gammas):
            W = load_or_train_ML(tau, N_bins=N_bins, gamma=gamma, ModelClass = ScoreWeightsMonotone)
            WW += [W]
            # compare_weights(*WW, density=True, range=10, name='ML_comparison')
    
print('Trained ML models')
# %%
compare_weights(*WW, density=True, range=11, name='ML_comparison')
# %%
assert(False)
# ----------- discriminative-------------
if o.l == 'softmax_mask':
    loss_fn = softmax_loss_mask
elif o.l == 'softmax_subset':
    loss_fn = softmax_loss_subset
elif o.l == 'logreg':
    loss_fn = logreg_loss
    
W = load_or_train_ML(max_distance)
W1 = copy.deepcopy(W)
W1.density = False
Wfname = W_name(W1,loss_fn.__name__,o)    
if o.load: # init with last saved model
    file_name = o.modelpath + Wfname + '.pkl'
    W1 = torch.load(file_name)
    print('Loaded ' + Wfname)
    Wfname += '+'
else: # init with ML model
    pass

print('Learning ' + Wfname)
optimizer = torch.optim.Adam(W1.parameters(), lr=o.lr)
# if o.max_batches is not None:
    # small_dataset, remainder_set = torch.torch.utils.data.random_split(residual_dataset, [o.batch_size*o.max_batches, len(residual_dataset) - batch_size*o.max_batches])
    # loader = torch.utils.data.DataLoader(small_dataset,batch_size=batch_size,num_workers=0,shuffle=True)
# else:
loader = data_loader
LL = []
for it in range(o.epochs):
    print(f'epoch {it}')
    tau_S = 0.1
    if it>=o.epochs//2:
        tau_S = 0.1
    L = learn_epoch(loss_fn, loader, W1, tau_S=tau_S, max_batches=o.max_batches)
    LL += [L]
    W1.o = o
    w = W1.score_weights_normalized().detach().cpu().numpy()
    np.savez(o.modelpath + Wfname, weight=w,
             tau=W1.max_distance, max_distance=W1.max_distance)
    torch.save(W1, o.modelpath + Wfname + '.pkl')
    compare_weights(W, W1, density=False, name=Wfname)

#assert(False)

WW += [W1]
# %%
# WW = []
# max_batches = 200
# # N_bins = 1000
# # max_distance=10.0
# W = ScoreWeightsMonotone(N_bins=N_bins, max_distance=max_distance, max_outlier_dist=100)
# learn_ML(data_loader, W, lr=0.1, epochs=500, max_batches = max_batches)  # ML
# w = W.score_weights_normalized().detach().cpu().numpy()
# np.savez(f'models/monotone_tau={W.max_distance}_bins={W.N_bins}_SP',weight=w,tau=W.max_distance,max_distance=W.max_distance)
# WW += [W]
# compare_weights(*WW, density=False)
# %%


# build the expected inliers w


# WW += [load_W('models/logreg_loss_tau=10.0_bins={o.N_bins}_SP.pkl')]

# %%
density = False
max_distance = 10.0
prange = max_distance
# N_bins = 2000
# thresholds = [10.0, 3.0, 2.0, 1.5]
thresholds = [3.0, 10.0]
WWM = []
for tau in thresholds:
    if tau<=max_distance:
        W = ScoreWeightsMAGSAC(maximum_threshold=tau, N_bins=N_bins, max_distance = prange)
        w = W.score_weights_normalized().numpy()
        np.savez(f'{o.modelpath}magsac_tau={W.maximum_threshold}', weight=w, tau=W.maximum_threshold, max_distance=W.max_distance)
        WWM += [W]

# %%
labels = 'Our Monotone'
#labels = 'Our Convex'
plt.figure(figsize=(7,2.5))
# plt.figure(figsize=(8,4))
ax = plt.gca()
markers = 'oxsd^<v'
# RANSAC
tau = 3.0
x = torch.linspace(0, prange, N_bins)
y = (x < tau).float()
ax.plot(x, y, '-.',label=f'RANSAC ($\\tau = {tau}$)', zorder=1, linewidth=2)
np.savez(f'{o.modelpath}ransac_tau={tau}', weight=y, tau=tau, max_distance=max_distance)
# MSAC
tau = 3.0
x = torch.linspace(0, prange, N_bins)
y = (1-x**2/3.0**2).clamp(min=0)
# ax.plot(x, y, zorder=-1, linewidth=2) # skip color
ax.plot(x, y, ':', label=f'MSAC ($\\tau = {tau}$)', zorder=1, linewidth=2)
np.savez(f'{o.modelpath}msac_tau={tau}', weight=y,
         tau=tau, max_distance=max_distance)
# 
for W in WWM:
    w = W.score_weights_normalized().detach().cpu() # categorical over N_bins
    x = (torch.arange(w.shape[0], device = w.device)+0.5)/w.shape[0]*W.max_distance
    ax.plot(x, w, '--', label=f'MAGSAC ($\\tau_\max = {W.maximum_threshold}$)', zorder=2, linewidth=2)
    
    

for (i,W) in enumerate(WW):
    w = W.score_weights_unnormalized().detach().cpu() # categorical over N_bins
    # w = torch.log_softmax(w, dim=-1)
    x = (torch.arange(w.shape[0], device = w.device)+0.5)/w.shape[0]*W.max_distance
    # w = w + torch.log(2*x**0.5) # density change of variables
    if hasattr(W,'o'):
        label = W.o.l + f' ($\\tau = {W.max_distance})$'
    else:
        label = f'Monotone ($\\tau = {W.max_distance})$'
    if W.density and i==0:
        if density:
            w = w.exp()        
        if i==0:
            wmin = w.min()
            s = w.max()-wmin
        w = (w -wmin)/s
        ax.plot(x, w, '-', label=label, zorder=2, linewidth=2)
    else:
        w = w - w.min()
        p = ax.plot(x, w/w[0], '-', label=label, zorder=2, linewidth=2)
        # ax.plot(x, w/w[1], '-', label=label, zorder=2, linewidth=2, color=p[0].get_color())
    if i==0:
        w = W.log_p().detach().cpu() # categorical over N_bins
        bin_size = max_distance/w.shape[0]
        p = torch.exp(w)
        # renormalize
        p = p/(p.sum()*bin_size)
        p_in = 1-p[-1]/p
    # w1 = p_in
        # ax.plot(x, p, '-', label='mix density', zorder=2, linewidth=2)
        ax.plot(x, p_in, '-', label='expected inliers', zorder=2, linewidth=2)

if False:
    # add hist
    N_bins = counts.shape[-1]
    bin_size = max_distance/N_bins
    w = np.log(counts/n_points+1e-10) - math.log(bin_size)
    x = (torch.arange(N_bins, device = w.device)+0.5)/N_bins*max_distance
    # w = w + torch.log(2*x**0.5) # density change of variables
    if density:
        w = w.exp()    
    w = (w -wmin)/s
    ax.plot(x, w, '-', color=(0.0,0.0,0.0), label=f'Histogram ($\\tau = {max_distance})$', linewidth = 0.3, zorder = 0)
##
ax.legend()
ax.set_ylim(-0.1,1.1)
ax.set_xlim(0,prange)
ax.set_xlabel('residual [px]')
ax.set_ylabel('normalized score')
##
# handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 1, 3, 4, 5, 2]
# plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
ax.set_ylim(-0.2,1.05)
#ax.set_yscale('log')
plt.tight_layout()
plt.draw()
plt.savefig(f'fig/score-comparison-{o.l}.pdf')
plt.show()
#
# %%
