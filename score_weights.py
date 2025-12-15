import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import scipy
from scipy.special import gamma, gammainc, gammaincc
from scipy.stats import chi2
from collections import namedtuple
from types import SimpleNamespace
import os
import cv2

from .functional import *
from .tools import *
from sklearn.neighbors import KernelDensity


def hist_values(V, N_bins, min_V, max_V, interpolate = False, weights = None):
    """
    X [* M, N] padded with inf
    """
    X = (V - min_V)/(max_V-min_V)*N_bins
    X[torch.logical_or(X<0, X>N_bins) ] = N_bins # will go to the waste bin for values out of range
    Xi = X.floor()
    Xi = Xi.to(torch.long)
    H = V.new_zeros(list(V.shape[:-1]) + [N_bins + 2], dtype= torch.float32) # extra waste bin for points out of range + extra waste bin for index Xi+1
    if weights is None:
        weights = torch.tensor(1, dtype=torch.float32, device=H.device).expand(V.shape)
    else:
        weights = weights.expand(V.shape)
    H.scatter_add_(-1, Xi, weights)
    H = H[...,:-2] # chop off last 2 bins
    
    if False:
        X = (V - min_V)/(max_V-min_V)*N_bins
        X[torch.logical_or(X<0, X>N_bins) ] = N_bins # will go to the waste bin for values out of range
        Xi = X.floor()
        Xi = Xi.to(torch.long)
        H = V.new_zeros(list(V.shape[:-1]) + [N_bins + 2]) # extra waste bin for points out of range + extra waste bin for index Xi+1
        II = []
        for d in range(V.dim()-1):
            s = [1]*V.dim()
            s[d] = V.shape[d]
            I = torch.arange(V.shape[d], device=V.device).view(s).expand(V.shape)
            II += [I]
        II += [Xi]
        if not interpolate:
            H.index_put_(indices=II, values=torch.tensor(1).to(V), accumulate=True) # wants to allocate a lot of memory...
        else:
            u = X - Xi
            H.index_put_(indices=II, values = u, accumulate=True)
            II[-1] = Xi+1 # upper bin
            H.index_put_(indices=II, values = 1-u, accumulate=True)
        H = H[...,:-2] # chop off last 2 bins
    bin_edges = torch.arange(N_bins+1, device=V.device, dtype=V.dtype)/N_bins*(max_V - min_V) + min_V
    # check correctness with other implementations
    check = False
    if check:
        H1 = V.new_zeros(list(V.shape[:-1]) + [N_bins])        
        with torch.no_grad():
            for bin in range(N_bins):
                H1[..., bin] += torch.logical_and(V >= bin_edges[bin], V < bin_edges[bin+1]).sum(dim=-1)
        assert(((H-H1).abs()<1e-6).all())
        shape = [1]*V.dim() + [-1]
        bin_left = bin_edges[:-1].view(shape)
        bin_right = bin_edges[1:].view(shape)
        # H = torch.histogram(V, N_bins, range=(0, max_V)) -- does not work, maybe works in newer torch?
        H1 = torch.logical_and(V.unsqueeze(-1) >= bin_left, V.unsqueeze(-1) < bin_right).sum(dim=-2) # sum over residuals [*, N_bins]
        assert(((H-H1).abs()<1e-6).all())
    #
    num_points = (torch.logical_not(torch.isinf(V).any(dim=-2, keepdim=True))).sum(dim=-1, keepdim=True)
    assert H.dim() == num_points.dim()
    H = H / num_points
    return H, num_points, bin_edges


class PosiParam(nn.Module):
  """
  Keeps an internal unconstrained optimizable parameter _param to represent a positive value
  Use functions set_value and get_value
  """
  def __init__(self, value:Tensor):
    super().__init__()
    self._param = Parameter(value.double())
    # scale = 1/torch.arange(1,value.shape[0]+1, dtype = value.dtype, device = value.device) # preconditioning for per-coefficient learning rate: tail coefficients learn slower
    scale = torch.arange(1,value.shape[0]+1, dtype = value.dtype, device = value.device)
    scale = scale.flip(0)
    # scale[:] = 1
    self.register_buffer('scale', scale)
    self.set_value(value)

  def get_value(self) -> Tensor:
    # access the positive values represented
    # transform through softplus, Apply MD gradient technique when training
    v = F.softplus(self._param).detach() + self._param - self._param.detach()
    return v*self.scale

  def set_value(self, value:Tensor):
    # set positive values
    assert(value>=0).all()
    self._param.data = soft_minus((value.double() + 1e-6)/self.scale)

  def __repr__(self):
    # print the positive values represented
    return self.get_value().__repr__()

  def postfix_cumsum(self, dim=-1):
    # convert to monotone non-decreasing weights
    c = self.get_value()
    return (c.sum(dim=dim) - c.cumsum(dim=dim) + c) #/c.shape[0] # 


class ScoreWeightsUnconstrained(nn.Module):
    def __init__(self, N_bins: int, max_distance: float = 10.0, max_outlier_dist=100.0, tau=0.0, pow=1):
        super().__init__()
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.max_outlier_dist = max_outlier_dist
        self.pow = pow
        self.tau = tau
        self.w = Parameter(self.init_w())
        self.score_out = Parameter(torch.tensor(-5.0))

    def score_weights(self) -> Tensor:
        return self.w
    
    def score_weights_normalized(self) -> Tensor:
        w = self.w
        w = w- w.min()
        w = w / w.max()
        return w  

    def bin_size(self):
        return self.max_distance/self.N_bins
    
    def init_w(self):
        return torch.ones(self.N_bins)*0.1
    
    def gen_hyperparams(self, T):
        self.hyperparams = np.linspace(0.1, self.max_distance, T, endpoint=True)
        
    def set_hyperparam(self, t):
        self.tau = t
        self.w.data = self.init_w()
    
    def score_matrix(self):
        M = []
        for t in self.hyperparams:
            self.set_hyperparam(t)
            M += [self.score_weights_normalized()]
        M = torch.stack(M)
        return M
           
class ScoreWeightsRANSAC(ScoreWeightsUnconstrained):
    def init_w(self):
        x = torch.linspace(0, self.max_distance, self.N_bins)
        y = (x < self.tau).float()
        return y
    
    def score_residuals(self, rr: Tensor, reduction='sum') -> Tensor:
        """
        rr [*,N] - residuals of N data points
        output:
        scores [*]
        """
        score =  (rr < self.tau).float()
        if reduction == 'sum':
            score = score.sum(dim=-1)
        return score
    
# _________________________________________________________________________
#                      ScoreWeightsCat
# Monotone categorical distribution over bins (not a density), ignoring outside bin
##_________________________________________________________________________


class ScoreWeightsCat(nn.Module):
    def __init__(self, N_bins: int, max_distance: float = 10.0, max_outlier_dist=None):
        super().__init__()
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.max_outlier_dist = max_outlier_dist
        c = torch.ones(N_bins, dtype=torch.float)/N_bins
        self.c = PosiParam(c)
        self.score_out = Parameter(torch.tensor(1/N_bins))
        self.scale = None
        self.density=False

    def score_weights(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        W = self.c.postfix_cumsum()  # monotone decreasing weights
        W = W - self.score_out
        # if self.scale is None:
        #     self.scale = W[0].item()
        # else:
        #     W = W/self.scale
        return W
    
    def score_weights_normalized(self):
        w = self.score_weights()
        # w = torch.logsoftmax(w) - math.log(self.bin_size()) # as log_density over uniform bins
        w = w - w.min()
        w = w / w.max()
        return w
    
    def bin_size(self):
        return self.max_distance/self.N_bins

    def logZ(self):
        W = self.score_weights()
        W1 = torch.cat([W, W.new_zeros(1)], -1)
        logZ = W1.logsumexp(dim=-1)
        return logZ

    def log_p(self):
        return (self.score_weights() - self.logZ()).float()

    def score_ss(self, ss) -> Tensor:
        w = self.score_weights()
        logZ = self.logZ()
        w_last = 0
        n_out = ss.n_points - ss.n_inside
        S = ss.counts.to(w) @ (w - logZ) - n_out * logZ # H contains counts/num_points
        return S.float()

# _________________________________________________________________________
#                      ScoreWeightsMonotone
# Piecewise constant denistiy of observations p(x|theta) \propto exp(rho(r(x,theta))) -- volume corrected
##_________________________________________________________________________

class ScoreWeightsMonotone(nn.Module):
    def __init__(self, N_bins:int, max_distance:float = 10.0, max_outlier_dist = 100.0, pow = 1, **kwargs):
        super().__init__()
        self.N_bins = N_bins
        self.pow = pow
        self.max_distance = max_distance
        self.max_outlier_dist = max_outlier_dist
        c = torch.ones(N_bins, dtype=torch.float)/N_bins 
        self.c = PosiParam(c)
        self.score_out = Parameter(torch.tensor(-5.0))
        self.scale = None
        self.density = True

    def monotone_w(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        W = self.c.postfix_cumsum() # monotone decreasing weights
        return W
    
    def bin_size(self):
        return self.max_distance**self.pow/self.N_bins

    def logZ(self):
        W = self.monotone_w()
        # log partition function for ML
        N_bins = W.shape[0]
        bin_size = self.bin_size()
        last_bin_size = self.max_outlier_dist**self.pow
        W1 = torch.cat([W + math.log(bin_size), W[-1:] + math.log(last_bin_size)], -1)
        logZ =  W1.logsumexp(dim=-1)
        # DEBUG: ignore out of range data
        logZ = W.logsumexp(dim=-1) + math.log(bin_size) 
        return logZ
    
    def log_p(self):
        return (self.monotone_w() - self.logZ()).float()
       
    def score_ss(self , ss) -> Tensor:
        w = self.log_p()
        S = ss.counts.to(w) @ w + ss.n_out * w[-1]
        # DEBUG: ignore out of range data
        S = ss.counts.to(w) @ w
        return S.float()
    
    def p_in(self):
        w = self.score_weights().float()
        return (1 - torch.exp(w[-1]-w))
    
    def expected_inliers(self, ss):
        p = self.p_in()
        S = ss.counts @ p
        return S
    
    def score_weights_normalized(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        W = self.monotone_w()
        W = W - W[-1]
        W = W / W[0]
        return W.float()
    
    def get_alpha(self):
        U = 1/self.max_outlier_dist**self.pow
        p = self.log_p().exp()
        alpha = 1-p[-1]/U
        return alpha
    
    def w_end(self):
        w = self.log_p()
        # w_end = (-math.log(self.max_outlier_dist**2) + torch.log(1-self.get_alpha())).item()
        return w[-1]
    
    def r_density(self):
        return self.log_p()


# _________________________________________________________________________
#                      ScoreWeightsMonotoneVol
# Piecewise constant denistiy of observations p(x|theta) \propto exp(rho(r(x,theta))) -- volume corrected
##_________________________________________________________________________

class ScoreWeightsMonotoneVol(nn.Module):
    def __init__(self, N_bins:int, max_distance:float = 10.0, max_outlier_dist = 100.0):
        super().__init__()
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.max_outlier_dist = max_outlier_dist
        c = torch.ones(N_bins, dtype=torch.float)/N_bins 
        self.c = PosiParam(c)
        self.score_out = Parameter(torch.tensor(-5.0))
        self.scale = None
        self.density = True

    def score_weights(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        W = self.c.postfix_cumsum() # monotone decreasing weights
        W = W - W[-1]
        if self.scale is None:
            self.scale = W[0].item()
        else:
            W = W/self.scale
        return W
    
    def regularizer(self) -> Tensor:
        D = self.c.get_value()
        D = D / self.scale
        return (D**2).mean()

    def bin_size(self):
        return self.max_distance/self.N_bins

    def logZ(self):
        W = self.score_weights()
        # log partition function for ML
        N_bins = W.shape[0]
        bin_size = self.max_distance/N_bins
        last_bin_size = self.max_outlier_dist
        # W1 = torch.cat([W + math.log(bin_size), W[-1:] + math.log(last_bin_size)], -1)
        # logZ =  W1.logsumexp(dim=-1)
        W1 = torch.cat([W, self.score_out], -1)
        # logZ = torch.logsumexp(W + math.log(bin_size), dim=-1) # DEBUG: this normalizes the distribution to the data in range only
        logZ = W1.logsumexp(dim=-1)
        assert(False)
        return logZ
    
    def log_p(self):
        return (self.score_weights() - self.logZ()).float()
    
    def prob(self):
        return (self.score_weights() - self.logZ()).exp().float()

    def score(self , R:Tensor) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        H, num_points = hist_values(R, self.N_bins, 0, self.max_distance, loop=False)
        return self.score_hist(H, num_points)
        W = self.log_p()
        S = H.to(W) @ W
        num_points = num_points.view([-1] + [1]*(S.dim()-1))
        S = S * num_points + W[-1] *(1 - H.sum(dim=-1))*num_points # all points not in hist count same as the last bin
        return S.float()
    
    def score_hist(self , H:Tensor, num_points:Tensor) -> Tensor:
        w = self.log_p()
        S = H.to(W) @ w + ss.n_out * self.log
        num_points = num_points.view([-1] + [1]*(S.dim()-1))
        # S = S * num_points # DEBUG: do not account for points ouside of range, using only in-range counts in H
        # S = S * num_points + W[-1] *(1 - H.sum(dim=-1))*num_points # all points not in hist count same as the last bin
        S = S * num_points # DEBUG: do not account for points ouside of range, using only in-range counts in H
        return S.float()
    
    def score_ss(self , ss) -> Tensor:
        assert(False)
        w = self.log_p()
        S = ss.counts.to(w) @ w 
        # num_points = num_points.view([-1] + [1]*(S.dim()-1))
        # S = S * num_points # DEBUG: do not account for points ouside of range, using only in-range counts in H
        # S = S * num_points + W[-1] *(1 - H.sum(dim=-1))*num_points # all points not in hist count same as the last bin
        # S = S * num_points # DEBUG: do not account for points ouside of range, using only in-range counts in H
        return S.float()
    
    def score_vol(self, ss, vol, estimate_distance):
        """
        ss.counts [B M N]
        vol [B M]
        """
        W = self
        vol_bin = vol / estimate_distance * W.bin_size() # (x,y) volume per bin [B, M]
        logV_bin = torch.log(vol_bin)  # [B M]
        vol_out = 1 - vol_bin*W.N_bins # (x,y) volum for r> max_distance
        logV_out = torch.log(vol_out)  # [B M]
        #
        n_in = ss.n_inside.squeeze(-1)  # [B M]
        n_pts = ss.n_points.squeeze(-1)  # [B 1]
        n_out = n_pts - n_in # [B M]
        #
        w = W.score_weights().float()
        # w = W.c.postfix_cumsum().float()
        W1 = torch.cat([w.view([1, 1, -1]) + logV_bin.unsqueeze(-1), W.score_out + logV_out.unsqueeze(-1)], -1) # [B M N+1]
        logZ = torch.logsumexp(W1, dim=-1, keepdim=False) # [B, M]
        scores = ss.counts @ w + n_out*W.score_out  - logZ*n_pts # [B, M]
        return scores # [B, M]

    def p_in(self):
        w = self.score_weights().float()
        return (1 - torch.exp(w[-1]-w))
    
    def expected_inliers(self, ss):
        p = self.p_in()
        S = ss.counts @ p
        return S
    
    def score_hist_normalized(self , H:Tensor, num_points:Tensor) -> Tensor:
        W = self.score_weights_normalized()
        S = H.to(W) @ W
        num_points = num_points.view([-1] + [1]*(S.dim()-1))
        S = S * num_points + W[-1] *(1 - H.sum(dim=-1))*num_points # all points not in hist count same as the last bin
        return S.float()
    
    def score_hist_normalized_bias(self , H:Tensor, num_points:Tensor) -> Tensor:
        W = self.score_weights()
        W = W - W[-1] # this is already part of score_weights()
        S = H.to(W) @ W
        num_points = num_points.view([-1] + [1]*(S.dim()-1))
        S = S * num_points
        return S.float()

    def score_weights_unnormalized(self) -> Tensor:
        return self.log_p()

    def score_weights_normalized(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        W = self.score_weights()
        W = W - W[-1]
        W = W / W[0]
        return W.float()


def log_sum_exp2(x: Tensor, y: Tensor) -> Tensor:
    """
    compute log(exp(x) + exp(y))
    """
    m = torch.max(x, y).detach()
    return m + torch.log(torch.exp(x - m) + torch.exp(y - m))

def log_sub_exp2(x: Tensor, y: Tensor) -> Tensor:
    """
    compute log(exp(x) - exp(y))
    """
    m = torch.max(x, y).detach()
    return m + torch.log(torch.exp(x - m) - torch.exp(y - m))

def log_ball_volume(r,d:int):
    # r [*] / float
    # d - dimensions
    return math.log(math.pi)*d/2  - math.log(gamma(d/2+1)) + d*torch.log(to_tensor(r+1e-10))

# _________________________________________________________________________
#                      ScoreWeightsMonotoneSpherical
# Density of observed coordinates p(x|theta) \propto exp(rho(r(x,theta)))
# spherical model of inliers + uniform outliers
##_________________________________________________________________________

class ScoreWeightsMonotoneSpherical(nn.Module):
    def __init__(self, N_bins:int, max_distance:float = 10.0, d = 4, P_out=0, monotone=False, max_outlier_dist = 100.0, alpha = None, **kwargs):
        """
        Monotone Spherical density in range [0, max_distance] represented as piece-wise model with N_bins on the radius + A discrete component for all mass with r>max_distance
        The discrete mass component will estimate just the fraction of points out of range
        d -- dimensionality of the observed features space
        p(x|x') = 1/Z e^{w_{k}} for ||x-x'|| in bin k
        P(||x-x'||>r) = P_out
        """
        super().__init__()
        self.d = d
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.max_outlier_dist = max_outlier_dist
        c = torch.ones(N_bins, dtype=torch.float64)/N_bins
        self.monotone = monotone
        if self.monotone:
            self.c = PosiParam(c)  # positive incrementsto form monotone density component
        else:
            self.c = Parameter(torch.zeros(N_bins, dtype=torch.float64))
        if alpha is None:
            self.alpha = Parameter(torch.tensor(0.0)) # logit of inliers fraction
        else:
            self.alpha = alpha
        # self.c1 = Parameter(c)
        self.c1 = PosiParam(c)
        # self.uniform_in = Parameter(torch.tensor(-5.0)) # discrete mass component
        # self.c_bias = Parameter(torch.tensor(0.0)) # bias for the monotone component
        # self.score_out = Parameter(torch.tensor(-5.0)) # discrete mass component
        self.density = True
        self.P_out = P_out
        # precompute Volumes
        ri = torch.arange(self.N_bins+1,dtype=torch.float)*self.bin_size() # radii: [0, bin_size, 2*bin_size,...]
        log_dV = log_sub_exp2(log_ball_volume(ri[1:],d), log_ball_volume(ri[:-1],d)) # log( V(r_{k+1}) - V(r_{k}) ) -- volume where the density is proportional to w_k
        self.register_buffer('log_dV', log_dV)
        self.log_Vmax = log_ball_volume(ri[-1],d).item() # volume of the whole density domain
        # precompute Areas
        log_dA = torch.zeros(N_bins, N_bins) # r,k
        for x_bin in range(N_bins):
            for k in range(N_bins):
                a = 0
                for bine in range(1):
                    r = (x_bin + bine) *self.bin_size()
                    ry1 = k * self.bin_size()
                    ry2 = (k + 1) * self.bin_size()
                    # if k<r: # not posible for true $x'$ to be on the manifold at distance k*bin_size becasue $x$ is further away
                        # log_dA_rk = math.log(1e-20)
                    # else: # $x'$ will be on the manifold between circles of radius R1 and R2
                    R1 = math.sqrt(max(ry1**2 - r**2,0)) # can be zero
                    R2 = math.sqrt(max(ry2**2 - r**2,0))
                    a += log_sub_exp2(log_ball_volume(R2,d-1), log_ball_volume(R1,d-1))
                log_dA[x_bin, k] = a
        self.register_buffer('log_dA', log_dA)
        
    def bin_size(self):
        return self.max_distance/self.N_bins

    def monotone_w(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        # w = self.c.postfix_cumsum() + self.c_bias  # monotone decreasing density scores
        if self.monotone:
            w = self.c.postfix_cumsum()  # monotone decreasing density scores
        else:
            w = self.c
        return w
    
    def logZ_inliers(self):
        """ 
        # 1 = Int_x p(x) dx + P_out =  1/Z(sum_k dV_k e^{w_k} + e^{s_out})
        # Z = sum_k e^{w_k + logdV_k} + e^{s_out)
        1 = Int_x p(x) dx = 1/Z(sum_k dV_k e^{w_k})
        """
        w = self.monotone_w()
        # score_uniform_in = math.log(self.P_out)
        # W1 = torch.cat([w + self.log_dV, self.score_out.view([1]) ], -1)
        # W1 = torch.cat([w + self.log_dV, self.uniform_in.view([1]) + self.log_Vmax], -1)
        # uniform_score = torch.tensor(math.log(self.P_out)).to(w)
        # W1 = torch.cat([w + self.log_dV, uniform_score.view([1])], -1) # uniform chance is same as outside
        # logZ = W1.logsumexp(dim=-1)
        w = w + 0
        # w[-1] = w[-1] + math.log(self.P_out/(1-self.P_out)) + log_ball_volume(self.max_distance,d=self.d)
        logZ = torch.logsumexp(w + self.log_dV, dim=-1)
        return logZ
    
    def log_likelihood_inliers(self) -> Tensor:
        """
        Log-likelihood of x at distance r from M(theta) for the density part r<max_distance
        p(x|theta) = \int_{x' \in M(theta)}p(x|x') = \sum_k \int_{x' \in M(theta) and ||x-x'|| in bin_{k}} e^{w_k}/Z = \sum_k A_{rk} e^{w_k}/Z, where r = dist(x, M(theta))
        returns log_p [N_bins] -- density of p(x|theta) in terms of r = d(x,theta), relative to the uniform density on the manifold
        """
        w = self.monotone_w().view(1,-1) # [1,N_bins]
        logZ = self.logZ_inliers()
        # uniform_score = self.uniform_in self.log_Vmax
        # uniform_score = torch.tensor(math.log(self.P_out)).to(w)
        # W1 = torch.cat([w + self.log_dA, uniform_score.view([1,1]).expand([self.N_bins, 1])], -1) # for each r mixture over x' bins k and the uniform component 
        # log_Aw = torch.logsumexp(W1,dim=-1)
        log_Aw = torch.logsumexp(w + self.log_dA, dim=-1) # log sum_k (dA_rk e^{w_k})
        log_p = log_Aw - logZ # [N_bins]
        return log_p, logZ
    
    # def log_likelihood_outliers(self) -> Tensor:
    #     w = self.c1.postfix_cumsum() # flat density over r with monotonicity constraint
    #     # w = self.c1
    #     logZ = torch.logsumexp(w + math.log(self.bin_size()), dim=-1)
    #     log_p = w - logZ
    #     return log_p
    
    def get_alpha(self):
        if isinstance(self.alpha, torch.Tensor):
            return self.alpha.sigmoid()
        else:
            return torch.tensor(self.alpha).to(self.log_dV)
    
    def log_likelihood_full(self) -> Tensor:
        log_p_in, logZ = self.log_likelihood_inliers()  # [N_bins]
        # mixture of inlier density and outlier uniform density
        # P_in = 1-P_out
        # alpha = torch.sigmoid(self.alpha)
        # alpha = torch.tensor(0.6).to(log_p_in)
        alpha = self.get_alpha()
        # alpha = self.uni_fraction
        # alpha = torch.tensor(0.98).to(log_p_in)
        # alpha = 1 - self.P_out
        # log_V_out_max = log_ball_volume(self.max_outlier_dist, self.d)
        # log_A_out_max = log_ball_volume(self.max_outlier_dist, self.d-1)
        # log_uniform_density = log_A_out_max - log_V_out_max # A/V
        # log_p_out = -torch.tensor(self.max_distance*2).log().to(log_p_in) # A_max is a common factor
        # log_p_out = log_p_in[0] -  math.log(self.max_outlier_dist)
        log_p_out = - math.log(self.max_outlier_dist)
        integrate_density_x = torch.logsumexp(log_p_in, dim=-1) + math.log(self.bin_size()) # integrate log_p_in in x, should be equal to 0.5
        # log_p_out = - log_V
        # log_p_out = -torch.tensor(10).log().to(log_p_in) # A_max is a common factor
        # learn a flat distribution of r
        # log_p_out = self.log_likelihood_outliers()
        ll = log_sum_exp2(log_p_in + torch.log(alpha), log_p_out + torch.log(1-alpha) ) # density 
        # ll = log_p
        return ll
    
    def r_density(self):
        f = self.log_likelihood_full()
        return f + math.log(2) # both sides adound manifol  -> density doubled
    
    def p_inlier1(self):
        log_p_in, logZ = self.log_likelihood_inliers()  # [N_bins]
        alpha = self.get_alpha()
        log_p_out = -torch.tensor(self.max_distance*2).log().to(log_p_in) # A_max is a common factor
        p_in = log_p_in.exp()*alpha / (log_p_in.exp()*alpha + log_p_out.exp()*(1-alpha))
        return p_in.float()
      
    def score_ss(self , ss) -> Tensor:
        ll = self.log_likelihood_full()
        # log_p, logZ = self.log_likelihood_density()  # [N_bins]
        # log_P_out = self.score_out - logZ # scalar
        P_out = self.P_out
        P_in = 1-P_out
        # log_p = log_p + math.log(1-P_out) # p(1-P_out)
        S = ss.counts.to(ll) @ ll + ss.n_out * math.log(P_out) # second part is const anyhow, the factor in the first part is const for all data -- does not matter
        return S.float()
    
    # def P_out(self):
    #     return (self.score_out - self.logZ()).exp()
    
    def p_inlier(self):
        # log_p, logZ = self.log_likelihood_density()
        w = self.log_likelihood_full()
        return (1 - torch.exp(w[-1] - w)).float()
    
    def expected_inliers(self, ss):
        p = self.p_inlier()
        S = ss.counts @ p
        return S
    
    def score_weights_normalized(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        log_p = self.log_likelihood_full()
        # log_P_out = self.score_out - logZ
        P_out = self.P_out
        w = log_p + math.log(1-P_out)  # p(1-P_out)
        w = w - math.log(P_out) # shift such that score for r>max_distance is zero
        w = w / w.abs().max() # normalize scale
        return w.float()
    
    
# _________________________________________________________________________
#                      ScoreWeightsMonotoneMix
# Piecewise constant density of inlier residuals + uniform density of outlier residuals
##_________________________________________________________________________

class ScoreWeightsMonotoneMix(nn.Module):
    def __init__(self, N_bins:int, max_distance:float = 10.0, P_out=0, max_outlier_dist = 100.0, gamma = None, monotone=True, pow = 1, **kwargs):
        super().__init__()
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.max_outlier_dist = max_outlier_dist
        self.pow = pow
        c = torch.ones(N_bins, dtype=torch.float64)/N_bins
        self.monotone = monotone
        if self.monotone:
            self.c = PosiParam(c)  # positive incrementsto form monotone density component
        else:
            self.c = Parameter(torch.zeros(N_bins, dtype=torch.float64))
        if gamma is None:
            self.logit_gamma = Parameter(torch.tensor(0.0)) # logit of inliers fraction
        else:
            self.logit_gamma = torch.logit(to_tensor(gamma))
        self.density = True
        self.P_out = P_out
        
    def bin_size(self):
        return self.max_distance**self.pow/self.N_bins

    def monotone_w(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        # w = self.c.postfix_cumsum() + self.c_bias  # monotone decreasing density scores
        if self.monotone:
            w = self.c.postfix_cumsum()  # monotone decreasing density scores
        else:
            w = self.c
        return w
    
    def logZ_inliers(self):
        w = self.monotone_w()
        logZ = torch.logsumexp(w, dim=-1) + math.log(self.bin_size())
        return logZ
    
    def log_likelihood_inliers(self) -> Tensor:
        w = self.monotone_w()
        logZ = self.logZ_inliers()
        log_p = w - logZ # [N_bins]
        return log_p, logZ
    
    @property
    def gamma(self):
        return torch.sigmoid(self.logit_gamma)
    
    @gamma.setter
    def gamma(self, a):
        self.logit_gamma = torch.logit(to_tensor(a))
    
    def log_p_out(self):
        return -math.log(self.max_outlier_dist**self.pow)
    
    def ll_out(self):
        return self.log_p_out() + torch.log(1-self.gamma)   
    
    def log_likelihood_full(self) -> Tensor:
        log_p_in, logZ = self.log_likelihood_inliers()  # [N_bins]
        gamma = self.gamma.to(log_p_in)
        integrate_density_x = torch.logsumexp(log_p_in, dim=-1) + math.log(self.bin_size()) # integrate log_p_in in x, should be equal to 0.5
        ll = log_sum_exp2(log_p_in + torch.log(gamma), self.ll_out()) # density 
        return ll
      
      
    def score_ss(self , ss) -> Tensor:
        ll = self.log_likelihood_full()
        ll_out = self.ll_out() # density of outliers
        S = ss.counts.to(ll) @ ll + ss.n_out * ll_out # second part is const anyhow, the factor in the first part is const for all data -- does not matter
        return S.float()
    
    def p_inlier(self):
        # log_p, logZ = self.log_likelihood_density()
        w = self.log_likelihood_full()
        return (1 - torch.exp(w[-1] - w)).float()
    
    def expected_inliers(self, ss):
        p = self.p_inlier()
        S = ss.counts @ p
        return S
    
    def r_density(self):
        return self.log_likelihood_full()
    
    def score_weights_normalized(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        w = self.log_likelihood_full()
        ll_out = self.ll_out().item()
        w = w - ll_out
        w = w / w.abs().max() # normalize scale
        return w.float().detach()
    
    def gen_hyperparams(self, T):
        # self.hyperparams = torch.sigmoid(torch.linspace(-7, 3, T))
        self.hyperparams = np.linspace(0.1, self.max_distance, T, endpoint=True)
        
    def gamma_from_tau(self, tau):
        log_p_in, logZ = self.log_likelihood_inliers()  # [N_bins]
        bins = np.linspace(0, self.max_distance, self.N_bins+1, endpoint=True)
        lp = np.interp(tau, bins[:-1], log_p_in.detach().cpu().numpy())
        mu = np.exp(lp)
        a = 1/self.max_outlier_dist
        gamma = a/(a + mu)
        return gamma
    
    def set_hyperparam(self, t):
        # self.gamma = t
        # keep gamma constant
        # set bundelled noise scale and tau
        # self.scale = t
        self.tau = t
        self.gamma = self.gamma_from_tau(t)
        
    def score_residuals(self, rr: Tensor) -> Tensor:
        """
        rr [*,N] - residuals of N data points
        output:
        scores [*]
        """
        ss = sufficient_statistic(rr,self.N_bins,self.max_distance)
        score = ss.counts @ self.val_w
        return score
    
    def score_matrix(self):
        M = []
        for t in self.hyperparams:
            self.set_hyperparam(t)
            M += [self.score_weights_normalized()]
        M = torch.stack(M)
        return M
    
    def w_end(self):
        w_end = self.log_p_out() + torch.log(1-self.gamma)
        return w_end
    

# _________________________________________________________________________
#                      ScoreWeightsTZ
# Piecewise constant density of inlier residuals + uniform density of outlier residuals
##_________________________________________________________________________

class ScoreWeightsTZ(nn.Module):
    def __init__(self, N_bins:int, max_distance:float = 10.0, P_out=0, max_outlier_dist = 100.0, alpha = None, pow = 1, threshold_scale= 3.7, maximum_threshold=None,**kwargs):
        super().__init__()
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.max_outlier_dist = max_outlier_dist
        self.pow = pow
        self.scale = Parameter(torch.tensor(1.0))
        self.scale_out = Parameter(torch.tensor(max_distance).float())
        self.scale_out.requires_grad = False
        self.threshold_scale = threshold_scale
        # logit of inliers fraction
        if alpha is None:
            self.logit_alpha = Parameter(torch.tensor(0.0)) # logit of inliers fraction
        else:
            self.logit_alpha = torch.logit(to_tensor(alpha))
        self.density = True
        self.P_out = P_out
        if maximum_threshold is not None:
            self.set_hyperparam(t = maximum_threshold)
        
    def bin_size(self):
        return self.max_distance**self.pow/self.N_bins

    def log_likelihood_inliers(self) -> Tensor:
        rr = torch.linspace(0, self.max_distance, self.N_bins).to(self.scale)
        distr = torch.distributions.Normal(loc=torch.zeros_like(self.scale),scale=self.scale)
        log_p = distr.log_prob(rr)
        log_Z = torch.log(distr.cdf(rr[-1]) - distr.cdf(rr[0])) # mass in the modelled range
        log_p = log_p - log_Z
        log_P_tail = torch.log((1- distr.cdf(rr[-1])).clamp(min=1e-30)) + math.log(2)
        return log_p, log_P_tail
    
    def ll_out(self) -> Tensor:
        rr = torch.linspace(0, self.max_distance, self.N_bins).to(self.scale_out)
        # distr = torch.distributions.Normal(loc=torch.zeros_like(self.scale_out),scale=self.scale_out)
        # log_p = distr.log_prob(rr)
        log_p = torch.log(1/self.scale_out).expand(rr.shape)
        # log_P_tail = torch.log((1- distr.cdf(rr[-1])).clamp(min=1e-30))
        log_P_tail = 0 # do not model
        return log_p, log_P_tail
    
    @property
    def alpha(self):
        return torch.sigmoid(self.logit_alpha)
    
    @alpha.setter
    def alpha(self, a):
        self.logit_alpha.data = torch.logit(to_tensor(a))
    
    # def log_p_out(self):
        # return -math.log(self.max_outlier_dist**self.pow)
    
    # def ll_out(self):
        # return self.log_p_out() + torch.log(1-self.alpha)
    
    def log_likelihood_full(self) -> Tensor:
        log_p_in, log_P_in_tail = self.log_likelihood_inliers()  # [N_bins]
        log_p_out, log_P_out_tail = self.ll_out()  # [N_bins]
        alpha = self.alpha.to(log_p_in) 
        ll = log_sum_exp2(log_p_in + torch.log(alpha), log_p_out + torch.log(1-alpha)) # density 
        ll_tail = log_sum_exp2(log_P_in_tail + torch.log(alpha), log_P_out_tail + torch.log(1-alpha)) # density 
        return ll, ll_tail # [N_bins]
      
    def score_ss(self , ss) -> Tensor:
        ll, ll_tail = self.log_likelihood_full()
        # S = ss.counts.to(ll) @ ll + ss.n_out * ll_tail
        S = ss.counts.to(ll) @ ll
        return S.float()
    
    def p_inlier(self):
        w, tail = self.log_likelihood_full()
        return (1 - torch.exp(w[-1] - w)).float()
    
    def expected_inliers(self, ss):
        p = self.p_inlier()
        S = ss.counts @ p
        return S
    
    def r_density(self):
        return self.log_likelihood_full()[0]
    
    def score_weights_normalized(self) -> Tensor:
        """ Weights for multiplying with histogram of residuals """
        w = self.log_likelihood_full()[0]
        w = w - w[-1]
        w = w / w.abs().max() # normalize scale
        return w.float().detach()
    
    def gen_hyperparams(self, T):
        # self.hyperparams = torch.sigmoid(torch.linspace(-7, 3, T))
        self.hyperparams = np.linspace(0.1, self.max_distance, T, endpoint=True)
        
    def set_hyperparam(self, t):
        # self.alpha = t
        t = to_tensor(t).to(self.scale)
        self.scale.data = t/self.threshold_scale # 
        distr = torch.distributions.Normal(loc=torch.zeros_like(self.scale),scale=self.scale)
        p_in = distr.log_prob(self.scale).exp()*2 # Normal density (renormalized to [0,inftly]) at t
        p_out = 1/self.scale_out # uniform density
         # set alpha such that t is the threshold of the profile likelihood
        alpha = p_out/(p_in + p_out)
        assert(self.alpha > 0)
        assert(self.alpha < 1)
        self.alpha = alpha
    
    def score_matrix(self):
        M = []
        for t in self.hyperparams:
            self.set_hyperparam(t)
            M += [self.score_weights_normalized()]
        M = torch.stack(M)
        return M
    
    def w_end(self):
        w_end = self.log_likelihood_full()[0][-1]
        return w_end
    

# # __________________Torr-Zisserman mixture of Gaussian and Unioform parameterized adaptively with a threshold_________________
# class ScoreWeightsTZC(nn.Module): 
#     def __init__(self, threshold = 1.0, threshold_scale= 3.7, **kwargs):
#         super().__init__()
#         self.scale = Parameter(torch.tensor(1.0))
#         self.threshold_scale = threshold_scale
#         self.logit_alpha = Parameter(torch.tensor(0.0)) # logit of inliers fraction
#         self.scale_out = 10 # no effect
#         self.in_distr = None
#         self.density = True
#         self.set_hyperparam(self, threshold)
        
#     @property
#     def alpha(self):
#         return torch.sigmoid(self.logit_alpha)
    
#     @alpha.setter
#     def alpha(self, a):
#         self.logit_alpha.data = torch.logit(to_tensor(a))
        
#     def set_hyperparam(self, t):
#         t = to_tensor(t).to(self.scale)
#         self.scale.data = t/self.threshold_scale #
#         self.in_distr = torch.distributions.Normal(loc=torch.zeros_like(self.scale),scale=self.scale)
#         # set alpha and outlier probability adaptively from the threshold
#         p_in = self.in_distr.log_prob(self.scale).exp()*2 # Normal density (renormalized to [0,inftly]) at t
#         p_out = 1/self.scale_out # uniform density -- the actual value is irrelevant
#          # set alpha such that t is the threshold of the profile likelihood
#         alpha = p_out/(p_in + p_out)
#         self.alpha = alpha
#         self.lim_ll = (torch.log(1-alpha) + self.ll_out(torch.ones_like(self.scale)*torch.inf)).detach()
    
#     def ll_in(self, rr) -> Tensor:
#         log_p = self.in_distr.log_prob(rr)
#         return log_p
    
#     def ll_out(self, rr:Tensor) -> Tensor:
#         log_p = torch.log(1/self.scale_out).expand(rr.shape)
#         return log_p
        
#     def ll_full(self, rr:Tensor) -> Tensor:
#         log_p_in = self.ll_in(rr)
#         log_p_out = self.ll_out(rr)
#         alpha = self.alpha.to(log_p_in) 
#         ll = log_sum_exp2(log_p_in + torch.log(alpha), log_p_out + torch.log(1-alpha))
#         return ll
        
#     def normalized_score(self, rr:Tensor) -> Tensor:
#         ll = self.ll_full(rr)
#         score = (ll - self.lim_ll)/(self.ll_full(rr.new_zeros([])) - self.lim_ll)
#         return score
    
#     def score_residuals(self, rr:Tensor) -> Tensor:
#         """
#         rr [*,N] - residuals of N data points
#         output:
#         scores [*]
#         """
#         # use unnormalized score
#         scores = self.ll_full(rr)
#         return scores    
    

# __________________GU continuous model parameterized adaptively with a threshold_________________
class ScoreWeightsGU(nn.Module): 
    def __init__(self, N_bins: int, max_distance=10.0, threshold=1.0, k=1, pow=1, **kwargs):
        super().__init__()
        self.tau = to_tensor(threshold)
        self.scale = threshold / k
        self.k = k
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.pow = pow
        self.score_norm = None
        self.density = False
        self.set_hyperparam(threshold)
        
    def set_hyperparam(self, t):
        self.tau = to_tensor(t)
        self.scale = t / self.k
        self.score_norm = F.softplus(self.tau**2/(2*self.scale**2))

    def score_residuals(self, rr:Tensor, reduction='sum') -> Tensor:
        """
        rr [*,N] - residuals of N data points
        output:
        scores [*]
        """
        score = F.softplus((self.tau**2 - rr**2)/(2*self.scale**2)) / self.score_norm
        if reduction == 'sum':
            score = score.sum(dim=-1)
        return score
    
    # def score_R(self, RR:Tensor, reduction='sum') -> Tensor:
    #     """
    #     rr [*,N] - squared residuals of N data points
    #     output:
    #     scores [*]
    #     """
    #     score = F.softplus((self.tau**2 - RR)/(2*self.scale**2)) / self.score_norm
    #     if reduction == 'sum':
    #         score = score.sum(dim=-1)
    #     return score

    def IRLS_weight(self, rr:Tensor) -> Tensor:
        """
        rr [*,N] - squared residuals of N data points
        output:
        weight [*, N] -- IRLS weight per point (up to scale, normalized by total number of points)
        """
        weight = F.sigmoid((self.tau**2 - rr**2)/(2*self.scale**2))
        # total_weight = weight.detach().sum(dim=-1, keepdim=True)
        # weight = weight / total_weight
        return weight

    def score_weights_normalized(self):
        rr = torch.linspace(0, self.max_distance, self.N_bins)
        w = self.score_residuals(rr, reduction=None)
        return w

    def score_ss(self, ss) -> Tensor:
        ll = self.score_weights_normalized()
        S = ss.counts.to(ll) @ ll
        return S.float()
    
    def gen_hyperparams(self, T):
        self.hyperparams = np.linspace(0.1, self.max_distance, T, endpoint=True)
        
    def score_matrix(self):
        M = []
        for t in self.hyperparams:
            self.set_hyperparam(t)
            M += [self.score_weights_normalized()]
        M = torch.stack(M)
        return M
# __________________GaU continuous model parameterized adaptively with a threshold_________________

class ScoreWeightsGaU(nn.Module): 
    def __init__(self, N_bins: int, max_distance=10.0, threshold=1.0, k=1, pow=1, **kwargs):
        super().__init__()
        self.tau = to_tensor(threshold)
        # self.scale = threshold / k
        self.scasle = None
        self.k = k
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.pow = pow
        self.density = False
        self.gamma = torch.tensor(0.5)
        self.alpha = torch.tensor(1/max_distance)
        self.scale = torch.tensor(threshold)
        
    def log_p_inliers(self, rr):
        log_p = -rr**2/(2*self.scale**2) + math.log(2/math.pi)/2 - math.log(self.scale)
        return log_p
    
    def log_p_mix(self, rr):
        log_p_in = self.log_p_inliers(rr)
        log_p_out = torch.log(self.alpha)
        ll = log_sum_exp2(log_p_in + torch.log(self.gamma), log_p_out + torch.log(1-self.gamma)) # density 
        return ll
        
    def score_residuals(self, rr:Tensor, reduction='sum') -> Tensor:
        """
        rr [*,N] - residuals of N data points
        output:
        scores [*]
        """
        score = self.log_p_mix(rr)
        if reduction == 'sum':
            score = score.sum(dim=-1)
        return score

    def score_weights(self):
        rr = torch.linspace(0, self.max_distance, self.N_bins)
        w = self.score_residuals(rr, reduction=None)
        return w

    # def score_ss(self, ss) -> Tensor:
    #     ll = self.score_weights_normalized()
    #     S = ss.counts.to(ll) @ ll
    #     return S.float()
    
    def gen_hyperparams(self, T):
        self.hyperparams = torch.tensor(np.linspace(0.1, self.max_distance, T, endpoint=True))
        # self.gammas = torch.tensor([0.001])
        self.gammas = torch.tensor([0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3])
        # self.ks = torch.tensor([0.001])
        
    def set_hyperparam(self, s):
        self.scale = to_tensor(s)
        self.alpha = 1/(3*torch.max(s,torch.tensor(3)))
        # self.gamma = self.gamma_from_tau(self.scale)
        
    def gamma_from_tau(self, tau):
        mu = self.log_p_inliers(tau).exp()
        gamma = self.alpha/(self.alpha + mu)
        return gamma
        
    def score_matrix(self):
        M = []
        for s in self.hyperparams:
            self.set_hyperparam(s)
            m = []
            for gamma in self.gammas:
                # self.gamma = self.gamma_from_tau(s*k)
                self.gamma = gamma
                w = self.score_weights()
                m += [w]
            m = torch.stack(m)
            M += [m]
        M = torch.stack(M)
        return M

# __________________MSAC with IRLS weights_________________
class ScoreWeightsMSAC(ScoreWeightsUnconstrained):
    def init_w(self):
        x = torch.linspace(0, self.max_distance, self.N_bins)
        y = (1-x**2/self.tau**2).clamp(min=0)
        return y

    def score_residuals(self, rr:Tensor, reduction='sum') -> Tensor:
        """
        rr [*,N] - residuals of N data points
        output:
        scores [*]
        """
        score = (1 - (rr/self.tau)**2).clip(min=0)
        if reduction == 'sum':
            score = score.sum(dim=-1)
        return score
    
    def IRLS_weight(self, rr:Tensor) -> Tensor:
        """
        rr [*,N] - squared residuals of N data points
        output:
        weight [*, N] -- IRLS weight per point
        """
        weight = (rr.abs() <= self.tau).float()
        return weight

    def score_weights_normalized(self):
        rr = torch.linspace(0, self.max_distance, self.N_bins)
        w = self.score_residuals(rr, reduction=None)
        return w


# -----------------------------------------------------------------------------
#                           ScoreWeightsConvex
# -----------------------------------------------------------------------------

class ScoreWeightsConvex(ScoreWeightsMonotone):
    def __init__(self, N_bins: int, max_squared_distance:float = 10.0**2, max_outlier_squared_dist:float = 100.0**2):
        super().__init__(N_bins, max_squared_distance,max_outlier_squared_dist)
        self.scale = None


    def monotone_weights(self) -> Tensor:
        return self.c.postfix_cumsum() # monotone decreasing weights

    def IRLS_weights(self) -> Tensor:
        """ Weights for IRLS as a function of squared residuals """
        return self.monotone_weights()

    def score_weights(self) -> Tensor:
        """ Weights for multiplying with histogram of squared residuals """
        nu = self.monotone_weights()
        rho = nu.cumsum(dim=-1) # integral like rho function
        W = - rho
        W = W - W[-1] # <W, h> requires the last bin to have score 0, otherwise incorrectly counting points not in the histogram
        # constant scaling
        if self.scale is None:
            self.scale = W[0].item()
        W = W / self.scale
        return W
    

# -----------------------------------------------------------------------------
#                           Oracle -- artifficial class for evaluation
# -----------------------------------------------------------------------------

class Oracle:
    def __init__(self, N_bins, max_distance, pow):
        self.N_bins = N_bins
        self.max_distance  = max_distance
        self.pow = pow
        self.name = 'Oracle'
        self.locked = True

class MethodGT:
    def __init__(self, N_bins, max_distance, pow):
        self.N_bins = N_bins
        self.max_distance  = max_distance
        self.pow = pow
        self.name = 'GT'
        self.locked = True


# class GCMAGSAC:
#     def __init__(self, N_bins, max_distance, pow):
#         self.N_bins = N_bins
#         self.max_distance  = max_distance
#         self.pow = pow
#         self.name = 'GC-MAGSAC++'
#         self.locked = True


# -----------------------------------------------------------------------------
#                           ScoreWeightsMAGSAC
# -----------------------------------------------------------------------------

class ScoreWeightsMAGSAC(nn.Module):
    def __init__(self, maximum_threshold: float = 10, N_bins: int = 200, max_distance: float = 10**2, dof: int = 4):
        super().__init__()
        self.maximum_threshold = maximum_threshold
        self.N_bins = N_bins
        self.max_distance = max_distance
        self.dof = dof
    def _WS(self):
        maximum_threshold = self.maximum_threshold
        # The degrees of freedom of the data from which the model is estimated.
        # E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
        degrees_of_freedom = self.dof
        # A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
        alpha = 0.99
        if degrees_of_freedom == 4:
            k = 3.64
        elif degrees_of_freedom == 2:
            k = 3.03
        else:
            k = np.sqrt(chi2.ppf(alpha, degrees_of_freedom))
        C = 1 / (2**(degrees_of_freedom / 2) * gamma(degrees_of_freedom / 2)*alpha)
        # Convert the maximum threshold to a sigma value
        maximum_sigma = maximum_threshold / k

        RR, step = np.linspace(0, self.max_distance, self.N_bins, endpoint=False, retstep=True)
        RR = RR + step/2
        residuals = RR
        #print(residuals)
        W = torch.zeros(self.N_bins)
        S = torch.zeros(self.N_bins)
        
        def gammainc_upper(a, x):
            v = gammaincc(a, x)
            Z = gamma(a)
            return v*Z
        
        def gammainc_lower(a,x):
            v = gammainc(a, x)
            Z = gamma(a)
            return v*Z
        
        for i,residual in enumerate(residuals):
            R_sigma = residual**2 / (2 * maximum_sigma**2)
            a = (degrees_of_freedom - 1) / 2            
            delta_gamma = (gammainc_upper(a, R_sigma) - gammainc_upper(a, k**2 / 2))
            # Calculate the IRLS weight implied by the current point
            weight = (1 / maximum_sigma) * C * 2**a *delta_gamma
            
            # Calculate the score implied by the current point
            a1 = (degrees_of_freedom + 1) / 2
            score = (1 / maximum_sigma) * C * 2**a1 * (maximum_sigma**2/2 * gammainc_lower(a1,R_sigma) + residual**2/4*delta_gamma)
            W[i] = weight
            S[i] = score
            
        W[residuals > maximum_threshold + step/2] = 0
        S[residuals > maximum_threshold + step/2] = S[residuals <= maximum_threshold + step/2][-1]
        S = -S + S[-1]
        return W,S
    
    
    def IRLS_weights(self):
        W,S = self._WS()
        return W
    
    def score_weights(self):
        W,S = self._WS()
        return S
    
    def gen_hyperparams(self, T):
        self.hyperparams = np.linspace(0.1, self.max_distance, T, endpoint=True)
        
    def set_hyperparam(self, t):
        self.maximum_threshold = t
    
    def score_matrix(self):
        M = []
        for t in self.hyperparams:
            self.set_hyperparam(t)
            M += [self.score_weights_normalized()]
        M = torch.stack(M)
        return M
    
    def score_weights_normalized(self) -> Tensor:
        """ Weights for multiplying with histogram of squared residuals """
        W = self.score_weights()
        W = W - W[-1]
        W = W / W.max()
        return W.float()
    
    def score_residuals(self, rr: Tensor) -> Tensor:
        """
        rr [*,N] - residuals of N data points
        output:
        scores [*]
        """
        ss = sufficient_statistic(rr,self.N_bins,self.max_distance)
        score = ss.counts @ self.val_w
        return score
    

# todo: upgrade to support ML, monotone / convex variants
class ScoreWeightsContinuous(nn.Module):       
    def __init__(self, N_basis: int, max_squared_distance: float = 10.0**2, max_outlier_squared_dist: float = 100.0**2, N_bins=200):
        def basisf(x: Tensor) -> Tensor:
            return torch.sigmoid(-x)
        
        super().__init__()
        self.max_squared_distance = max_squared_distance
        self.N_basis = N_basis
        self.N_bins = N_bins # used when quering the histogram form, can change dynamically
        self.max_outlier_squared_dist = max_outlier_squared_dist
        # create basis locations, assuming the argument of the function is squared residual
        bins = torch.arange(N_basis+1, dtype=torch.float64)**2 / (N_basis**2) * max_squared_distance  # fixed, but can be learnable
        mu = (bins[0:-1] + bins[1:])/2  # bin centers
        self.register_buffer('mu', mu) # basis locations
        bin_sizes = (bins[1:] - bins[0:-1])
        # 
        c = torch.ones(N_basis, dtype=torch.float64)/N_basis
        self.amp = PosiParam(c)
        scale = bin_sizes  # learned hinge smoothness
        self.register_buffer('scale', scale)
        self.basisf = basisf

    def _score(self, R: Tensor) -> Tensor:
        # we will use sigmoid basis functions for monotone model
        shape = [1]*R.dim() + [-1]
        # all combinations d_i - mu_j #
        dmu = (R.unsqueeze(-1) - self.mu.view(shape))
        x = dmu / self.scale.view(shape)
        score = self.basisf(x) * self.amp.get_value().view(shape)
        S = score.sum(dim=-1) 
        # S[-1] is just the score of the last point in the batch
        return S
    
    def score_weights(self, N_bins: int) -> Tensor:
        R = (torch.arange(N_bins, dtype=torch.float, device=self.mu.device) + 0.5) / N_bins * self.max_squared_distance  # uniform over [0 max_squared_d]
        return self._score(R)
    
    def logZ(self):
        N_bins = 10000
        S = self.score_weights(N_bins)
        bin_size = self.max_squared_distance/N_bins
        last_bin_size = self.max_outlier_squared_dist
        S1 = torch.cat([S + math.log(bin_size), S[-1:] + math.log(last_bin_size)], -1)
        assert S1.dtype == torch.float64
        return S1.logsumexp(dim=-1)
    
    def log_p(self, R:Tensor) -> Tensor:
        scores = self._score(R)
        logZ  = self.logZ() # scalar
        return (scores - logZ).float()

    def prob(self, R:Tensor) -> Tensor:
        return self.log_p(R).exp().float()

    def score(self, R: Tensor) -> Tensor:
        mask = torch.logical_not(torch.isinf(R)) # inf represents padded values, not to be counted
        S = (self.log_p(R)*mask).sum(dim=-1)
        return S.float()

    def score_weights_unnormalized(self, N_bins: int = None) -> Tensor:
        if N_bins is None:
            N_bins = self.N_bins
        W = self.score_weights(N_bins)
        W = W - self.logZ()
        return W.float()

    def score_weights_normalized(self, N_bins: int = None) -> Tensor:
        if N_bins is None:
            N_bins = self.N_bins
        """ Weights for multiplying with histogram of squared residuals """
        W = self.score_weights(N_bins)
        W = W - W[-1]
        W = W / W[0]
        return W.float()


##_____________________________________________________________________
class ScoreWeightsContinuousConvex(nn.Module):
  def __init__(self, N_basis: int, max_squared_distance:float, N_bins:int, normalize = False):
    super().__init__()
    self.max_squared_distance = max_squared_distance
    N = N_basis
    self.N_basis = N_basis
    self.N_bins = N_bins
    c = torch.ones(N, dtype=torch.float)*0.00001
    c[N//4] = 1
    self.slopes = PosiParam(c)
    bins = torch.arange(N+1, dtype=torch.float)**2 / (N**2) * max_squared_distance # fixed, but can be learnable
    mu = (bins[0:-1] + bins[1:])/2 # bin centers
    self.register_buffer('mu', mu)
    bin_sizes = (bins[1:] - bins[0:-1])
    betas = 1/bin_sizes # learned hinge smoothness
    self.register_buffer('betas', betas)
    self.normalize = normalize
    if not normalize:
      self.norm = 1.0
      W = self.score_weights()
      self.norm = W[0].item()
    


    def _score(self, d: Tensor) -> Tensor:
        """ score given squared residuals d, unnormalized """
        shape = [1]*d.dim() + [-1]
        dmu = -(d.unsqueeze(-1) - self.mu.view(shape)) # all combinations d_i - mu_j #
        f = soft_plus(dmu, self.betas.view(shape))*self.slopes.get_value().view(shape)
        score = f.sum(dim=-1)
        return score

    def score(self, d: Tensor) -> Tensor:
        """ score given squared residuals d """
        s = self._score(d)
        Z = self._score(d.new_zeros([1]))
        if self.normalize:
            s = s/s[0]
        else:
            s = s/self.norm
        return s

def weights_hist_squared_residuals(self, N_bins:int = None) -> Tensor:
    if N_bins is None:
        N_bins = self.N_bins
    d = (torch.arange(N_bins, dtype=torch.float, device=self.mu.device) + 0.5) / N_bins * self.max_squared_distance # uniform over [0 max_squared_d]
    return self.score(d)

    def weights_hist_residuals(self, N_bins:int = None) -> Tensor:
        r = (torch.arange(N_bins, dtype=torch.float, device=self.mu.device) + 0.5)/ N_bins * (self.max_squared_distance**0.5)
        return self.score(r**2)

    def score_weights(self):
        return self.weights_hist_squared_residuals(self.N_bins)


def sufficient_statistic(residuals, N_bins, max_distance, pow = 1, weights=None, accum_out = True):
    """
    residuals [B M N] -- padded with inf
    """
    H, n_points_H, bins = hist_values(residuals ** pow, N_bins, 0, max_distance **pow, weights=weights)
    counts = H * n_points_H
    n_inside = (residuals < max_distance).sum(dim=-1, keepdim=True)# [B M 1]
    n_points = (residuals < torch.inf)[:,:1,:].sum(dim=-1, keepdim=True) #  must have same number of points for all models  [B 1 1]
    n_out = (n_points-n_inside).squeeze(-1) # [B M]
    if accum_out: # move all out-of range residuals to the last bin
        counts[:,:, -1] += n_out
        n_inside[:,:,0] += n_out
        n_out[:] = 0
    return SimpleNamespace(counts=counts, bins=bins, n_inside=n_inside, n_points=n_points, n_out=n_out)

# def K_scaling(K1, K2):
    # return (K1[:,0,0]+ K1[:,1,1] + K2[:,0,0] + K2[:,1,1])/4

# def residuals_preproc(data):
#     residuals = data['residuals']
#     R = residuals.abs()
#     # if scaling:
#         # scale = K_scaling(data['K1'], data['K2'])
#         # R = R * scale.view([-1, 1, 1])
#     return R


def sufficient_statistic_GT(loader, N_bins, max_distance, max_batches=None, pow = 1):
    """
    """
    counts = 0
    n_inside = 0
    n_points = 0
    for idx, data in enumerate(loader):
        data['models'] = data['models'][:,-1:,...] # filter out only GT model correspondances        
        compute_residuals(data)
        R = data['residuals']
        R = R[:,-1:,:] # select GT model only [B 1 N]
        ss = sufficient_statistic(R, N_bins, max_distance, pow=pow, accum_out = False)
        # accumulate ss
        n_inside += ss.n_inside.sum() # total points inside
        n_points += ss.n_points.sum() # total points
        counts = counts + ss.counts.flatten(start_dim=-2, end_dim=-1).sum(dim=0) # total counts from all images
        if max_batches is not None and idx >= max_batches:
            break
    bins = ss.bins
    return counts, bins, n_inside, n_points


def normalized_points(data):
    correspondences = data['correspondences']
    C = to_tensor(correspondences).cuda()  # [B, N, 4]
    x = torch.cat([C[..., :2], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
    y = torch.cat([C[..., 2:], C.new_ones(list(C.shape[:-1]) + [1])], dim=-1)
    return x,y


def unnormalized_points(data):
    x,y = normalized_points(data)
    K1 = data['K1'].float() # [B, 3, 3] -- paired with x
    K2 = data['K2'].float() # [B, 3, 3] --AR paired with y        
    # converting to fundamental matrix and recomputing residuals anew
    # K1I = K1.inverse().cuda()
    K1 = K1.cuda()
    # K2I = K2.inverse().cuda()
    K2 = K2.cuda()
    if not data['is_F'][0]:
        x = torch.einsum('bij, bnj -> bni', K1, x) # (K1)x in the format [b,n,3]
        x = x/x[:,:,-1:]
        y = torch.einsum('bij, bnj -> bni', K2, y) # (K2)y in the format [b,n,3]            
        y = y/y[:,:,-1:]
    correspondences_px = torch.cat([x[...,:-1],y[...,:-1]],dim=-1)
    data['correspondences_px'] = correspondences_px
    return x,y

def compute_kde_weights(data, bw):
    # CC = data['correspondences_px'].cpu().numpy()
    CC = data['correspondences'].cpu().numpy()
    nn = data['num_pts']
    weights = np.zeros((CC.shape[0:2])) # [B N]
    for b in range(CC.shape[0]):
        n = nn[b].item()
        C = CC[b][:n]
        kde = KernelDensity(kernel='tophat', bandwidth=bw).fit(C)
        p = np.exp(kde.score_samples(C))
        w = 1/p
        w = w / w.sum() * n
        weights[b, :n] = w
    data['kde_weights'] = torch.from_numpy(weights).float() # [B N]


class Failure:
    def __init__(self) -> None:
        pass

    def is_collinear(self, points, threshold=1e-6):
        """
        Check if a set of points are collinear.
        Args:
            points (np.array): Nx2 array of 2D points.
            threshold (float): Threshold to determine collinearity.
        Returns:
            bool: True if points are collinear, False otherwise.
        """
        if len(points) < 3:
            return True
        p1, p2, p3 = points[:3]
        matrix = np.array([
            [p1[0], p1[1], 1],
            [p2[0], p2[1], 1],
            [p3[0], p3[1], 1]
        ])
        return np.abs(np.linalg.det(matrix)) < threshold

    def check_degenerate_case(self, points1, points2):
        """
        Check for degenerate cases in an image pair for fundamental matrix estimation.
        Args:
            points1 (np.array): Nx2 array of 2D points from the first image.
            points2 (np.array): Nx2 array of 2D points from the second image.
        Returns:
            bool: True if a degenerate case is detected, False otherwise.
        """
        # Check if points are collinear in either image
        if self.is_collinear(points1) or self.is_collinear(points2):
            return True

        # Check fundamental matrix estimation
        points1 = np.int32(points1)
        points2 = np.int32(points2)
        F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

        if F is None:
            return True  # Degenerate case detected, unable to compute a valid F

        # Count the number of inliers
        inliers = mask.ravel().tolist().count(1)
        total_points = points1.shape[0]

        # Check if the number of inliers is below a certain threshold
        inliers_threshold = 8  # Minimum number of inliers required
        if inliers < inliers_threshold:
            return True

        return False

    def check(self, points, errors, num_points, selected_indices):
        # valid model is a model with pose error below 10 degree
        # scoring failure: cannot find a good model in the pool
        # degenerate case
        # pre-scoring failure : no valid model in the pool
        # import pdb; pdb.set_trace()
        # if not any(error < 10 for error in errors):
        #     print("Pre-scoring failure detected.")
        pre_score_count = ((errors < 10 ).sum(-1) < 1).sum()
        degenerate = 0
        for i, pts in enumerate(points.cpu().numpy()):
            if self.check_degenerate_case(pts[:num_points[i], :2], pts[:num_points[i], 2:]):
                degenerate += 1
        selection_failure = (selected_indices !=  torch.argmin(errors, dim=-1)).sum()
        return pre_score_count, selection_failure, degenerate
        
            
