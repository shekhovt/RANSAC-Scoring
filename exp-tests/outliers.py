# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import scipy
from scipy.special import gamma, gammainc, gammaincc
import os

from score_learn.load_data import *
from score_learn import *

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#!%matplotlib inline

#%% 

# def Sampson(x: Tensor, y: Tensor, F: Tensor) -> Tensor:
#     """
#     x [n x 3]
#     y [n x 3]
#     F [3 x 3]
#     """
#     xF = x @ F # [n, 3]
#     Fy = y @ F.T # [n,3 ]
#     numerator = (xF * y).sum(dim=-1) # [n]
#     denom = ((xF[:, 0:2]**2 + Fy[:, 0:2]**2).sum(dim=-1))**0.5
#     return numerator/denom

F = torch.Tensor([[-0.00310695, -0.0025646, 2.96584],
[-0.028094, -0.00771621, 56.3813],
[13.1905, -29.2007, -9999.79]])
F = F.to('cuda')

def t(r):
    if not isinstance(r, Tensor):
        r = torch.tensor(r)
    return r

def hist_Sampson(res = [800,800], repeats=1000, n= 10_000_000, bins=20, max_distance=20):
    H = 0
    res = torch.tensor(res, device='cuda').view([1,-1])
    for i in range(repeats):
        x = torch.rand([2*n, 2], device='cuda')*res
        x = torch.cat([x, torch.ones(2*n, device='cuda').unsqueeze(-1)], dim=1)      
        y = x[:n,:]
        x = x[n:,:]
        r = Sampson(x,y, F)
        r = t(r)
        r = r.cpu().numpy()
        h,bins = np.histogram(r, bins,range = [-max_distance,max_distance], density=True)
        H = H + h
    H = H / repeats
    return H, bins,r

rscale = 1

def hist_Sampson_loader(n=10000_000, repeats=101, N_bins=20, max_distance=20):
    dataset_info = PhotoTourismRootSIFT
    src = 'st_peters_square'; sqrt = True; # SP+SG features
    residual_dataset = ResidualData(dataset_info, src, padding=True)
    loader = torch.utils.data.DataLoader(residual_dataset, batch_size=1, num_workers=0, shuffle=True)
    H_rep = 0
    H = 0
    torch.manual_seed(10) # (70,10) n
    vol = []    
    for idx,data in enumerate(loader):
        F = data['models'][0,-1].cuda()
        K1 = data['K1'][0].to(F)
        K2 = data['K2'][0].to(F)
        sz1 = 2*K1[:-1, -1] # [B 2]
        sz2 = 2*K2[:-1, -1] # [B 2]
        for rep in range(repeats):
            with torch.no_grad():
                # F = data['models'][0, rep].cuda()
                # torch.manual_seed(10)
                x = (torch.rand([2*n, 2], device='cuda')) # normalized coordinates in [-1,1]
                x = torch.cat([x, torch.ones(2*n, device='cuda').unsqueeze(-1)], dim=1)
                y = x[:n, :]
                x = x[n:, :]
                # unnormalized coordinates
                # x = x @ K1.T # (K1)x in the format [n,3]
                # y = y @ K2.T # (K2)y in the format [n,3]
                # rscale = (K1[0, 0] + K1[1, 1] + K2[0, 0] + K2[1, 1]).item()/4
                x[...,0] *= sz1[0].item()
                x[...,1] *= sz1[1].item()
                y[..., 0] *= sz2[0].item()
                y[..., 1] *= sz2[1].item()
                # r = Sampson(y, x, F).abs()
                r = Sampson(y, x, F)
                n_points = r.shape[0]
                v = (r.abs() < 10).sum().item()/n_points
                v = (1-v)/v
                vol += [v]
                r = r.cpu().numpy()
                h, bins = np.histogram(r, N_bins, range = [-max_distance,max_distance], density=False) # this renormalizes and throws away values out of range?
                h = h / n_points
                H = H + h
                H_rep += 1
        if idx>=0:
            break
    H /= H_rep
    vol = np.array(vol)
    print('mean', vol.mean())
    print('std', vol.std())
    print(vol.min())
    print(vol.max())
    return H, bins


# %%
H, bins  = hist_Sampson_loader()
# %%
# H, bins, r = hist_Sampson(res=[1200, 1200])
if True:
    plt.figure()
    ax = plt.gca()
    bin_centers = (bins[:-1]+bins[1:])/2
    ax.plot(bin_centers, H)
    # mask = np.abs(r)<400
    # std = r.std()*1.1
    # print(std)
    x = bin_centers
    # ax.plot(x, scipy.stats.norm.pdf(x, scale=std))

    # bin_centers1 = (bins1[:-1]+bins1[1:])/2
    # ax.plot(bin_centers1, H1)
    # ax.plot(x, scipy.stats.norm.pdf(x, scale=std * 1200/800))
    # n, bb, patches = ax.hist(r, 300, density=True)
    plt.grid()
    # plt.yscale('log')
    # plt.xlim(-400,400)
    plt.show()
    # # %%
    # from scipy import stats
    # kernel = stats.gaussian_kde(r)
    # rr = np.linspace(bins[0], bins[-1],200)
    # pp = kernel(rr)
    # plt.figure()
    # plt.plot(rr, np.log(pp))
    # plt.show()
# %%
