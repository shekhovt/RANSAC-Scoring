# %%
# if  __name__ == "__main__":
#     __name__ = 'score_weights.score_weights_script'
#     __package__ = 'score_weights'
#     __run__ = True
# else:
#     __run__ = False
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
import os
import shlex
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#!%matplotlib inline

from score_learn import *
from score_learn.load_data import *



op = ArgumentParser()
op.set_defaults(batch_size=8)
op.add_argument("-l", type=str, default='ML', help="ML, softmax_mask, softmax_subset, logreg")
op.add_argument("--N_bins", type=int, default=200)
op.add_argument("--max_distance", type=float, default=10.0, help="max distance for the histogram")
op.add_argument("--reg", type=float, default=0.0, help="max distance for the histogram")
op.add_argument("--max_batches", type=int, default=None, help ="None uses all batches of the dataset, can specify fewer batches for a fast check" )
op.add_argument("--epochs", type=int, default=10, help ="training epochs")
op.add_argument("--lr", type=float, default=0.1, help ="learning rate")
op.add_argument("--load", action="store_true", default=False, help="continue training from last saved model")

# SP RootSIFT
# op.add_argument("--data", type=str, default="/tmp/RANSAC/st_peters_square_RootSIFT/", help="path to data")
# op.add_argument("--sqrt", action="store_true", default=True, help="take square root of residuals if True")
# op.add_argument("--modelpath", type=str, default='models/RS/', help="path for saving models")
# op.add_argument("-F", action="store_true", default=False, help="Fundamental matrix case, no extra scaling")

# # SP-SG
# op.add_argument("--data", type=str, default="/tmp/RANSAC/st_peters_square1/", help="path to data")
# op.add_argument("--sqrt", action="store_true", default=True, help="take square root of residuals if True")
# op.add_argument("--modelpath", type=str, default='models/', help="path for saving models")
# op.add_argument("-F", action="store_true", default=False, help="Fundamental matrix case, no extra scaling")

## KITTI
# # op.set_defaults(batch_size=8)
# op.add_argument("--data", type=str, default="/tmp/RANSAC/KITTI/train/", help="path to data")
# op.add_argument("--sqrt", action="store_true", default=False, help="take square root of residuals if True")
# op.add_argument("--modelpath", type=str, default='models/KITTI/', help="path for saving models")
# op.add_argument("-F", action="store_true", default=True, help="Fundamental matrix case, no extra scaling")

## ETH3D
op.add_argument("--data", type=str, default="/tmp/RANSAC/eth3d/", help="path to data") # is this SPSG
op.add_argument("--sqrt", action="store_true", default=True, help="take square root of residuals if True")
op.add_argument("--modelpath", type=str, default='models/eth3d/', help="path for saving models")


args_str = ' '.join(sys.argv[1:])
ops, args = op.parse_known_args(shlex.split(args_str))
o = SimpleNamespace(**vars(ops))
print(o)
N_bins = o.N_bins
max_distance = o.max_distance
batch_size = o.batch_size
max_batches = o.max_batches

density = False
max_distance = 10.0
prange = max_distance
# N_bins = 2000
# thresholds = [10.0, 3.0, 2.0, 1.5]


def W_name(W, method, o):
    name = f'{method}_tau={W.max_distance}_bins={W.N_bins}'
    if method != 'ML':
        name += f'_reg={o.reg}'
    if hasattr(W, 'alpha'):
        name += f'_alpha={(W.get_alpha().item())*100:3.1f}'
    return name


thresholds = []
WW = []
for tau in thresholds:
    W = ScoreWeightsMonotone(N_bins=N_bins, max_distance=tau)
    Wfname = W_name(W, 'ML', o)
    file_name = o.modelpath + Wfname + '.pkl'
    W = torch.load(file_name)
    WW += [W]

ll = []
# ll += ['models/logreg_loss_tau=10.0_bins=200_reg=0.0.pkl']
# ll += ['models/softmax_loss_mask_tau=10.0_bins=200_reg=0.0.pkl']

for file_name in ll:
    W = torch.load(file_name)
    W.o = copy.deepcopy(o)
    if 'logreg' in file_name:
        fname = 'logreg'
    elif 'mask' in file_name:
        fname = 'softmax_mask'
    else:
        fname = 'softmax_subset'
    W.o.l = fname
    WW += [W]
    # print(file_name)
    # print(W.o)

Mthresholds = [10.0, 3.0, 2.0, 1.5]
WWM = []
for tau in Mthresholds:
    if tau <= max_distance:
        W = ScoreWeightsMAGSAC(maximum_threshold=tau, N_bins=N_bins, max_distance=prange)
        w = W.score_weights_normalized().numpy()
        np.savez(f'{o.modelpath}magsac_tau={W.maximum_threshold}',weight=w, tau=W.maximum_threshold, max_distance=W.max_distance)
        torch.save(W, f'{o.modelpath}magsac_tau={W.maximum_threshold}.pkl')
        WWM += [W]

# %%
labels = 'Our Monotone'
# labels = 'Our Convex'
# plt.figure(figsize=(8, 3))
plt.figure(figsize=(7, 3))
# plt.figure(figsize=(8,4))
ax = plt.gca()
markers = 'oxsd^<v'
# RANSAC
for tau in Mthresholds:
    W = ScoreWeightsRANSAC(tau=tau, N_bins=N_bins, max_distance=max_distance)
    x = torch.linspace(0, max_distance, N_bins)
    y = W.w.detach().cpu()
    ax.plot(x, y, '-.', label=f'RANSAC ($\\tau = {W.tau}$)', zorder=101, linewidth=2,clip_on=False)
    np.savez(f'{o.modelpath}ransac_tau={tau}', weight=y, tau=tau, max_distance=max_distance)
    torch.save(W, f'{o.modelpath}ransac_tau={tau}.pkl')
# MSAC
for tau in Mthresholds:
    # tau = 3.0
    W = ScoreWeightsMSAC(tau=tau, N_bins=N_bins, max_distance=max_distance)
    x = torch.linspace(0, max_distance, N_bins)
    y = W.w.detach().cpu()
    # y = (1-x**2/tau**2).clamp(min=0)
    # ax.plot(x, y, zorder=-1, linewidth=2) # skip color
    ax.plot(x, y, ':', label=f'MSAC ($\\tau = {tau}$)', zorder=101, linewidth=2,clip_on=False)
    np.savez(f'{o.modelpath}msac_tau={tau}', weight=y, tau=tau, max_distance=max_distance)
    torch.save(W, f'{o.modelpath}msac_tau={tau}.pkl')
#
for W in WWM:
    w = W.score_weights_normalized().detach().cpu()  # categorical over N_bins
    x = (torch.arange(w.shape[0], device=w.device) +
         0.5)/w.shape[0]*W.max_distance
    ax.plot(
        x, w, '--', label=f'MAGSAC ($\\tau_\max = {W.maximum_threshold}$)', zorder=102, linewidth=2)


for (i, W) in enumerate(WW):
    w = W.score_weights_unnormalized().detach().cpu()  # categorical over N_bins
    # w = torch.log_softmax(w, dim=-1)
    x = (torch.arange(w.shape[0], device=w.device) +
         0.5)/w.shape[0]*W.max_distance
    # w = w + torch.log(2*x**0.5) # density change of variables
    if hasattr(W, 'o'):
        # print(W.o)
        label = W.o.l + f' ($\\tau = {W.max_distance})$'
    else:
        label = f'Monotone ($\\tau = {W.max_distance})$'
    if W.density and i == 0:
        if density:
            w = w.exp()
        if i == 0:
            wmin = w.min()
            s = w.max()-wmin
        w = (w - wmin)/s
        ax.plot(x, w, '-', label=label, zorder=102, linewidth=2, clip_on=False)
    else:
        w = w - w.min()
        p = ax.plot(x, w/w[0], '-', label=label, zorder=102, linewidth=2, clip_on=False)
        # ax.plot(x, w/w[1], '-', label=label, zorder=2, linewidth=2, color=p[0].get_color())
    if i == 0:
        # print(W.o)
        w = W.log_p().detach().cpu()  # categorical over N_bins
        bin_size = max_distance/w.shape[0]
        p = torch.exp(w)
        # renormalize
        # p = p/(p.sum()*bin_size)
        p_in = 1-p[-1]/p
    # w1 = p_in
        # ax.plot(x, p, '-', label='mix density', zorder=2, linewidth=2)
        ax.plot(x, p_in, '-', label='expected inliers', zorder=2, linewidth=2)

if False:
    # add hist
    N_bins = counts.shape[-1]
    bin_size = max_distance/N_bins
    w = np.log(counts/n_points+1e-10) - math.log(bin_size)
    x = (torch.arange(N_bins, device=w.device)+0.5)/N_bins*max_distance
    # w = w + torch.log(2*x**0.5) # density change of variables
    if density:
        w = w.exp()
    w = (w - wmin)/s
    ax.plot(x, w, '-', color=(0.0, 0.0, 0.0),
            label=f'Histogram ($\\tau = {max_distance})$', linewidth=0.3, zorder=0)
##
ax.legend()
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0, prange)
ax.set_xlabel('residual [px]')
# ax.set_ylabel('normalized score')
##
# handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 1, 3, 4, 5, 2]
# plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
ax.set_ylim(0, 1)
# ax.set_yscale('log')
plt.tight_layout()
plt.draw()
plt.savefig(f'fig/score-comparison-{o.l}.pdf')
plt.show()
#
# %%
