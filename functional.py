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