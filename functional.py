import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Union, Callable
import numbers


def to_tensor(x: Union[Tensor, numbers.Number]) -> Union[Tensor, None]:
    if x is None or torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


def divup(x, y):
    return (x + y - 1) // y

def soft_minus(x: Tensor, beta=1):
    """ inverse of soft_plus """
    a = x * beta
    # prevent exp overflow:
    m = a.clamp(min=0).detach()
    return (torch.log(torch.exp(a - m) - torch.exp(-m)) + m) / beta


def soft_plus(x: Tensor, beta: Tensor):
    """ inverse of soft_plus """
    a = x * beta
    # prevent exp overflow:
    m = a.clamp(min=0).detach()
    return (torch.log(torch.exp(a - m) + torch.exp(-m)) + m) / beta


def batch_histogram(data_tensor, num_classes=-1):
    """
    Computes histograms, even if in batches (as opposed to torch.histc and torch.histogram).
    Arguments:
        data_tensor: a D1 x ... x D_n torch.LongTensor
        num_classes (optional): the number of classes present in data.
                                If not provided, tensor.max() + 1 is used (an error is thrown if tensor is empty).
    Returns:
        A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
        containing histograms of the last dimension D_n of tensor,
        that is, result[d_1,...,d_{n-1}, c] = number of times c appears in tensor[d_1,...,d_{n-1}].
    """
    maxd = data_tensor.max()
    nc = (maxd+1) if num_classes <= 0 else num_classes
    hist = torch.zeros(
        (*data_tensor.shape[:-1], nc), dtype=data_tensor.dtype, device=data_tensor.device)
    ones = torch.tensor(1, dtype=hist.dtype,
                        device=hist.device).expand(data_tensor.shape)
    hist.scatter_add_(-1, ((data_tensor * nc) // (maxd+1)).long(), ones)
    return hist


def unsqueeze_expand(input:Tensor, dim:int, size_dim:int=1):
    """
    unsqueezes a dimension and expands this dimension to size size_dim
    """
    if dim < 0:
        dim = dim + input.dim() + 1
    s = list(input.shape)
    return input.unsqueeze(dim).expand(s[:dim] + [size_dim] + s[dim:])

def select_dim1(source, best_idx):
    B = best_idx.shape[0]
    best_e = source[torch.arange(B).view([-1] + [1]*(best_idx.dim()-1)).expand(best_idx.shape), best_idx]  # [B T]
    return best_e  # [B T] / [B]

def sample_subsets(k:int, log_p: Tensor, n_samples: int) -> Tensor:
    """ 
    ### Draw n_samples of k subsets without replacement from Categorical distribution
    log_p [N] -- log of Categorical probabilities of drawing individual points
    rerurn: 
    ii [n_samples, k] -- indices of sampled subsets
    
    Sampling scheme:
        top-k(log(p) + G), G~Gumbel(0,1), F_G(x) = exp(-exp(-x)) 
    =top-k( log(p) -log(-log(U)) ) # G~-log(-log(U))
    =top-k( p/-log(U) ) # exp is monotone
    =top-k( log(U)/p ) # inverse negative is monotone
    =top-k( log(U^{1/p}) )
    =top-k( U^{1/p} )
    """
    p = log_p.exp()
    U = p.new_empty(n_samples, p.shape[0]).uniform_()
    keys = U**(1/p)
    (_, ii) = torch.topk(keys, k=k, dim=-1, largest=True)  # [n_samples, k]
    return ii


def sample_visible_subsets(k:int, log_p: Tensor, n_samples: int, x:Tensor, y:Tensor) -> Tensor:
    """ 
    ### Draw n_samples of k subsets without replacement from Categorical distribution
    log_p [N] -- log of Categorical probabilities of drawing individual points
    x [N, 2] -- points in image 1
    y [N, 2] -- points in image 2
    Check that sampled subsets preserve orientations of all triplets in x,y
    rerurn: 
    ii [n_samples, k] -- indices of sampled subsets
    """
    log_p = log_p.cuda()
    x = x.cuda()
    y = y.cuda()
    II = torch.zeros((0, k), dtype=torch.long, device=log_p.device)
    while len(II) < n_samples:
        ii = sample_subsets(k, log_p, n_samples)  # [n_samples, k]
        x1 = x[ii]  # [n_samples, k, 2]
        y1 = y[ii]  # [n_samples, k, 2]
        vec_x = x1[:, 1:] - x1[:, :1]  # [n_samples, k-1, 2]
        vec_y = y1[:, 1:] - y1[:, :1]  # [n_samples, k-1, 2]
        cross_x = vec_x[:, :-1, 0]*vec_x[:, 1:, 1] - vec_x[:, :-1, 1]*vec_x[:, 1:, 0]  # [n_samples, k-2]
        cross_y = vec_y[:, :-1, 0]*vec_y[:, 1:, 1] - vec_y[:, :-1, 1]*vec_y[:, 1:, 0]  # [n_samples, k-2]
        valid = ((cross_x * cross_y) > 0).all(dim=1)  # [n_samples]
        # check for points that are too close to each other in both images
        dist_x = (x1[:, None, :, :] - x1[:, :, None, :]).norm(dim=-1)  # [n_samples, k, k]
        dist_y = (y1[:, None, :, :] - y1[:, :, None, :]).norm(dim=-1)  # [n_samples, k, k]
        valid = valid & ((dist_x + torch.eye(k, device=log_p.device)[None, :, :]*1e6).min(dim=1).values > 1e-6).all()
        valid = valid & ((dist_y + torch.eye(k, device=log_p.device)[None, :, :]*1e6).min(dim=1).values > 1e-6).all()
        # check also for collinearity (i.e., zero area in either image)
        valid = valid & (cross_x.abs().min(dim=1).values > 1e-6) & (cross_y.abs().min(dim=1).values > 1e-6)
        II = torch.cat([II, ii[valid]], dim=0)
        
    return II[:n_samples].cpu()  # [n_samples, k]