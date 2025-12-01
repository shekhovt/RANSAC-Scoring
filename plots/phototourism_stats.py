#%%
# load histograms from pkl files and plot them
# were saved with pickle.dump({'hist': hist, 'bin_edges': bin_edges}, f)
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib import ticker
from matplotlib import gridspec
from matplotlib import rcParams
from matplotlib import rc
rc('text', usetex=False)
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['figure.figsize'] = (8, 6)
def load_histogram(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    hist = data['hist']
    bin_edges = data['bin_edges']
    return hist, bin_edges

# %%

hist,bin_edges = load_histogram('residuals_histogram.pkl')
r = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Midpoints of bins
m = hist

# %%
# plot it
plt.figure(figsize=(8, 6))
plt.plot(r, m, label='Histogram Density', color='blue', linewidth=2)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Histogram of Residuals')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.draw()


# %%
hist, bin_edges = load_histogram('sampson_error_histogram.pkl')
r = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Midpoints of bins
m = hist


# %%
import numpy as np
from scipy.optimize import curve_fit

def zero_mean_gaussian(x, sigma, A):
    """1D zero-mean Gaussian with scale sigma and amplitude A"""
    return A * np.exp(-x**2 / (2 * sigma**2))

def fit_zero_mean_gaussian(x, y):
    """
    Fit a zero-mean Gaussian density A * exp(-x^2 / (2*sigma^2)) to data.
    
    Parameters:
        x (array-like): input x values
        y (array-like): corresponding y values (density estimates)
    
    Returns:
        sigma (float): estimated standard deviation
        A (float): estimated amplitude
        y_fit (np.array): fitted values at x
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Initial guesses: sigma from weighted variance, A from max
    sigma0 = np.sqrt(np.sum(x**2 * y) / np.sum(y))
    A0 = np.max(y)
    
    popt, _ = curve_fit(zero_mean_gaussian, x, y, p0=[sigma0, A0])
    sigma_est, A_est = popt
    
    return sigma_est, A_est

def fit_gaussian_density(r, m, threshold=0.5):
  mask = r < threshold
#   sigma, A = fit_zero_mean_gaussian(r[mask], m[mask])
  v = np.sum(r[mask]**2*m[mask]) / np.sum(m[mask])
  sigma = v**0.5
  A = 1
  q = zero_mean_gaussian(r, sigma, A)
  q = q / q.sum()
#   q = q * (r[1] - r[0])
  return q


# %%
# Combine the two histograms into subplots
fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

# First histogram
hist1, bin_edges1 = load_histogram('residuals_histogram.pkl')
r1 = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
m1 = hist1 / np.sum(hist1)
bin = (r[1] - r[0])
# axs[0].plot(r1, m1*bin, label='Histogram Density', color='k', linewidth=2)
# plot the histogram as a bar plot
axs[1].bar(r1, m1/bin, width=bin, label='Histogram Density', color='k', alpha=0.3)
# fit m1 with chi(4) distribution
# Fit chi(4) distribution
df = 4
from scipy.stats import chi
# Estimate scale parameter by matching the mean
mean_r1 = np.sum(r1 * m1) / np.sum(m1)
scale_est = mean_r1 / chi.mean(df)
chi_pdf = chi.pdf(r1, df, scale=scale_est)
# Normalize to match histogram area
chi_pdf = chi_pdf / chi_pdf.sum() * m1.sum()
# axs[1].plot(r1, chi_pdf/bin, label=r'$\chi(4)$ fit', color='red', linestyle='--', linewidth=2)
axs[1].set_xlabel(r'Residual to GT norm $\|x - \bar x\|$, px')
axs[1].set_ylabel('Density')
# axs[0].set_title('Histogram of Residuals')
axs[1].grid(True)
axs[1].legend()

# Second histogram
hist2, bin_edges2 = load_histogram('sampson_error_histogram.pkl')
r2 = 0.5 * (bin_edges2[1:] + bin_edges2[:-1])
m2 = hist2 / np.sum(hist2)
# axs[1].plot(r2, m2*bin, label='Histogram Density', color='k', linewidth=2)
axs[0].bar(r2, m2/bin, width=bin, label='Histogram Density', color='k', alpha=0.3)
# axs[1].set_title('Histogram of Sampson Error')
# fit gaussian to the histogram data of sampson errors 
fg = fit_gaussian_density(r2, m2, threshold=0.93)
# plot it
axs[0].plot(r2, fg/bin, label='Fitted Gaussian', color='red', linestyle='--', linewidth=2)
axs[0].set_xlabel('Sampson error, px')
axs[0].grid(True)
axs[0].legend()

plt.tight_layout()
plt.savefig('phototourism-inliers.pdf', bbox_inches='tight', pad_inches=0.0)
plt.show()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .spherical_density import *
from ..tools import *
from ..functional import *


# need a monotonic parameterization of the profile function h(r)
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
    # convert to monotone non-increasing weights
    c = self.get_value()
    return (c.sum(dim=dim) - c.cumsum(dim=dim) + c) #/c.shape[0] # 


class ScoreWeightsMonotoneSpherical(nn.Module):
    def __init__(self, N_bins:int, max_distance:float = 10.0, d = 4):
        super().__init__()
        self.d = d
        self.N_bins = N_bins
        self.max_distance = max_distance
        c = torch.ones(N_bins, dtype=torch.float64)/N_bins
        self.c = PosiParam(c)  # positive incrementsto form monotone density component
        
    def bin_size(self):
        return self.max_distance/self.N_bins
  
    def profile(self) -> Tensor:
        return self.c.postfix_cumsum()  # monotone decreasing density scores

  
    def logZ_inliers(self):
        """ 
        # 1 = Int_x p(x) dx  =  1/Z(sum_k dV_k h(r_k)
        # Z = sum_k e^{w_k + log dV_k}
        1 = Int_x p(x) dx = 1/Z(sum_k dV_k e^{w_k})
        """
        w = self.profile()
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



def ML_estimate_spherical(r, m):
    """
    Estimate the parameters of a spherical distribution using Maximum Likelihood Estimation (MLE) from a histogram data m.
    
    Returns:
        tuple: Estimated parameters (mean, covariance matrix)
    """


    # Placeholder for MLE estimation logic
    # This function should implement the MLE estimation for spherical distributions
    pass