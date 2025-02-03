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
from collections import namedtuple
from types import SimpleNamespace
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from score_learn.load_data import *
#!%matplotlib inline


# %%
tau = 3

EE = []
II = []

for idx, data in enumerate(data_loader):
    residuals = data['residuals'] # [B, M, N]
    errors = data['errors'] # [B, M]
    num_pts = data['num_pts']
    scale = K_scaling(data['K1'], data['K2'])
    R = residuals.abs() * scale.view([-1, 1, 1])
    # RANSAC score
    score = (R < tau).sum(dim=-1).float() # [B M]
    
    EE += [errors.flatten()]
    II += [score.flatten()]
    if idx>30:
        break
    print(idx)
#%%

EE = torch.cat(EE)
II = torch.cat(II)
    
# %%
r = [[0, 10], [0, 300]]
hh, xe, ye = np.histogram2d(EE.numpy(),II.numpy(), bins=[40,50], range=r)

# %%
plt.figure(figsize=(6,6))
h1 = hh/hh.max(axis=1,keepdims=True)
plt.imshow(h1,interpolation='nearest', origin='lower') #, extent=np.array(r).flatten())
plt.draw()
plt.ylabel('Error')
plt.xlabel('score')
plt.show()
# %%
