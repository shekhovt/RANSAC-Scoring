# %% 

import os, sys
if  __name__ == "__main__":
    __name__ = 'score_learn.local_optimization'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    __run__ = False

import torch
import torch.func
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Union, Callable
import kornia

from .functional import *

# %% 
def SampsonELO(x, y, KX, KY, E) -> Tensor:
    """
    x,y [*, N, 3] -- normalized point coordinates
    E [*, N, 3, 3] -- separate copy of model per point pair (need for Gauss-Newton)
    KX [*, 3, 3] -- intrinsics matrix for x
    KY [*, 3, 3]
    return:
    residuals [*, N], in pixels
    """
    yE = torch.einsum('...i, ...ij -> ...j', y, E)  # [*, N, 3]
    Ex = torch.einsum('...j, ...ij -> ...i', x, E)  # [*, N, 3]
    numerator = torch.einsum('...i, ...i', yE, x) + 1e-5  # [*, N]
    # exctrac focal length from the diagonal
    fx = torch.diagonal(KX, dim1=-2, dim2=-1)[..., 0:2].unsqueeze(-2)  # [*, 1, 2]
    fy = torch.diagonal(KY, dim1=-2, dim2=-1)[..., 0:2].unsqueeze(-2)  # [*, 1, 2]
    denom = (((yE[..., 0:2]/fx)**2 + (Ex[..., 0:2]/fy)** 2).sum(dim=-1) + 1e-20)**0.5  # [*, N]
    r = numerator/denom  # [*, N]
    mask = (x[...,-1] == 0)
    r[mask] = 1e3
    # r = torch.nan_to_num(r, nan=1e3)
    return r # signed residual # needed for optimization


def compose_essential_matrix(R, t):
    Tx = kornia.geometry.epipolar.cross_product_matrix(t)
    return Tx @ R


class E_parameterization:
    def __init__(self, E: Tensor):
        """
        E [*, 3, 3] -- essential matrices
        """
        super().__init__()
        # decompose E into R,t, hopefully works in batches
        self.E0 = E
        R1, R2, t = kornia.geometry.epipolar.decompose_essential_matrix(E)
        self.bs = list(E.shape)[:-2]  # batch shape
        if self.bs == []:
            R1 = R1.squeeze(0)
        # 6 parameters per essential matrix
        self.param = E.new_zeros(self.bs + [7], dtype=torch.float64)
        self.param.requires_grad = True
        self.t = F.normalize(t.squeeze(-1), dim=-1)  # set translation
        
        self.R0 = R1.to(self.param)  # any is good
        # rotatino is modelled incrementally to R
        # set quaternion to [1,0,0,0] -- identity rotation
        self.q.data[..., 0] = 1
        E1 = self.forward()  # up to scale, posibly also negative
        pass

    @property
    def t(self):
        # return torch.tensor((0,0,1),dtype = torch.float64) # DEBUG
        return self.param[..., 0:3]

    @t.setter
    def t(self, v):
        self.param.data[..., 0:3] = v

    @property
    def q(self):
        return self.param[..., 3:]

    @q.setter
    def q(self, v):
        self.param.data[..., 3:] = v

    @property
    def R(self):
        q = F.normalize(self.q, dim=-1) # quaternion vector must be normalized
        dR = kornia.geometry.conversions.quaternion_to_rotation_matrix(q)
        # batched matrix multiplication
        R = torch.einsum('...ij, ...jk -> ...ik', self.R0, dR)
        return R
    
    @R.setter
    def R(self, R):
        self.R0.data = R
        self.q.data.fill_(0)
        self.q.data[..., 0] = 1

    def forward(self):
        # t = F.normalize(self.t, dim=-1) # translation vector is up to global scale
        t =  self.t # do not normalize, intentinally
        R = self.R
        E = compose_essential_matrix(R, t)
        return E
    
    def step(self, deltap):
        self.param.data += deltap
        # debug: freeze dR
        # self.q.data.fill_(0)
        # self.q.data[..., 0] = 1
        self.q = F.normalize(self.q, dim=-1)
        self.t = F.normalize(self.t, dim=-1)
        # self.t = torch.tensor((0,0,1),dtype = torch.float64) # DEBUG

    # @torch.compile(dynamic = True, backend = "inductor", mode="reduce-overhead")
    def Jacobian(self):
        def func(param):
            p = self.param
            self.param = param
            F = self.forward()
            self.param = p
            F = F.flatten(start_dim=-2)
            if p.dim()>1:
                return F.sum(dim=tuple(range(p.dim()-1)))  # for each batch dimension, independent inputs
            else:
                return F
        J_f = torch.func.jacrev(func)
        J = J_f(self.param)
        return J


# def SampsonBM(x: Tensor, y: Tensor, F: Tensor) -> Tensor:
#     """
#     x [B x n x 3]
#     y [B x n x 3]
#     F [B x M x 3 x 3]
#     """
#     xF = torch.einsum('bni, bmij -> bmnj', x, F)  # [M, n, 3]
#     Fy = torch.einsum('bnj, bmij -> bmni', y, F)  # [M, n, 3]
#     numerator = torch.einsum('bni, bnj, bmij -> bmn', x, y, F)  # [M, n]
#     denom = ((xF[..., 0:2]**2 + Fy[..., 0:2]**2).sum(dim=-1))**0.5
#     return numerator/denom  # [B, M, n]


# def compute_residuals_ref(x, y, KX, KY, E):
#     K1 = KX
#     K2 = KY
#     K1I = K1.inverse()
#     K2I = K2.inverse()
#     # convert E to F and unnormalize points
#     F = torch.einsum('bij, bmik, bkl -> bmjl', K2I,
#                      E, K1I)  # K2^{-T} E K1^{-1}
#     # (K1)x in the format [b,n,3]
#     x = torch.einsum('bij, bnj -> bni', K1, x)
#     # (K2)y in the format [b,n,3]
#     y = torch.einsum('bij, bnj -> bni', K2, y)
#     r = SampsonBM(y, x, F).abs()  # *scale.view([-1,1,1]) #[B M N]
#     r = torch.nan_to_num(r, nan=float('inf'))
#     return r.abs()


def local_optimization(x: Tensor, y: Tensor, KX, KY, model: nn.Module, score_f:Callable, weight_f: Callable, iterations: int = 20, damping_mult=1):
    """
    x,y [*, N, 3] -- normalized point pairs
    KX, KY -- intrinsics
    model --- nn.module with forward defining essential matrx of shape [*, 3, 3]
    score(residuals) -- scoring function, differentiable in residuals
    return:
    optimized models E [*, 3, 3], optimized score [*]
    """
    # x = torch.nan_to_num(x, posinf=1e3)
    # y = torch.nan_to_num(y, posinf=-1e3)
    mask = torch.isinf(x).any(dim=-1, keepdim=True).expand(x.shape)
    x[mask] = 0
    y[mask] = 0
    s0 = None
    N = x.shape[-2]  # number of points (padded in a batch)
    damping_mult = x.new_ones(x.shape[0])*damping_mult
    method = 'GN'
    for it in range(iterations+1):
        # forward compose essential matrix
        E = model.forward()
        # check
        if False:
            rr1 = compute_residuals_ref(
                x, y, KX, KY, model.E0.view([-1, 1, 3, 3])).squeeze(1)
            s1 = score(rr1)  # [*, N]
            print("s1", s1.sum(dim=-1)[0:10].cpu().detach().numpy())

            [U, S, V] = torch.svd(model.E0)
            S = S / S.max(dim=-1, keepdim=True)[0]
            S[S > 0.5] = 1
            S[S < 0.5] = 0
            # recompose
            for b in range(S.shape[0]):
                E[b] = torch.mm(torch.mm(U[b], torch.diag(S[b])), V[b].t())

            rr2 = compute_residuals_ref(
                x, y, KX, KY, E.view([-1, 1, 3, 3])).squeeze(1)
            s2 = score(rr2)  # [*, N]
            print("s2", s2.sum(dim=-1)[0:10].cpu().detach().numpy())

        # separate copy per point to compute separate gradient
        EE = unsqueeze_expand(E, -3, N)  # [*, N, 3, 3]
        rr = SampsonELO(x, y, KX, KY, EE) # signed residuals
        s = score_f(rr).sum(dim=-1)  # [*, N]
        if s0 is not None:
            # check improvement
            mask_backtrack = s0 > s
            model.param.data[mask_backtrack] = p0[mask_backtrack]
            damping_mult[mask_backtrack] *= 3 # more damping
            if mask_backtrack.any().item():
                # print('backtracking:', mask_backtrack.sum().item())
                # recompute sampson errors
                E = model.forward()
                EE = unsqueeze_expand(E, -3, N)  # [*, N, 3, 3]
                rr = SampsonELO(x, y, KX, KY, EE)  # signed residuals
                s = score_f(rr).sum(dim=-1)  # [*, N]
        else:
            # print(s.cpu().detach().numpy())
            s00 = s.detach().clone()
        if it == iterations:
            # print((s-s00).cpu().detach().numpy())
            assert((s >= s00).all())
            return E.detach(), s.detach()
        s0 = s.detach().clone()
        p0 = model.param.clone()
        # debug, print score, expect increasing
        if method == 'GD':
            model.zero_grad()
            (s.sum()/N).backward()  # full grad down to parameters
            for p in model.parameters():
                # maximizing score
                p.data += lr*p.grad
        elif method == 'GN':
            """
            GN rquires a single parameter vector, otherwise we cannot do JJ' and solve
            Our quality is
            Q(x) = \sum_i f_i(x)^2
            Linearizing f(x + dx) = f(x) + J dx
            J = df_i/ d x_j
            Q(x + dx) = \sum_i (f_i + J_i dx)^2 = sim_i f_i^2 + 2*f_i*J_i dx + (J_i dx)^2
            = 2*f'J dx + dx' J'J dx
            maximize:
            0 = 2*f'J + 2* dx'J'J
            0 = J'f + J'Jdx
            dx = -(J'J)^{-1} J'f
            """
            with torch.no_grad():
                weight = weight_f(rr)
            rr = rr /1000 # just to fix the scale
            J1 = torch.autograd.grad(rr.sum(), EE, retain_graph=True)[0]  # [*, N, 3, 3]
            # f = (1-s + 1e-5) ** 0.5 / N   # [*, N] square root for G-N
            # J1a = torch.autograd.grad(f.sum(), EE, retain_graph=True)[0]  # [*, N, 3, 3]
            # gradient of all residuals in E
            # J1 = torch.autograd.grad(rr.sum(), EE, retain_graph=True)[0]  # [*, N, 3, 3]
            assert (not J1.isnan().any())
            J1 = J1.flatten(start_dim=-2)  # [*, N, 9]
            # Jacobian of E in model parameters
            J2 = model.Jacobian()  # [9, *, 7]
            assert (not J2.isnan().any())
            with torch.no_grad():
                # apply it on J1
                J = torch.einsum('i...j, ...ni -> ...nj', J2, J1)  # [*, N, d], d = 7 -- marameters dimension
                # compose JJ'
                G = torch.einsum('...ni, ...nj,...n -> ...ij', J, J, weight)  # [*, d, d]
                # damping (we have a non-minimal parameterization)
                Gdiag = torch.diagonal(G, dim1=-1, dim2=-2) ## a view of the diagonal
                m = damping_mult.unsqueeze(-1)
                Gdiag[:] = Gdiag[:]*(1+1e-4*m) + 1e-3*m
                # compute J f
                b = torch.einsum('...ni, ...n, ...n -> ...i', J, rr, weight)  # [*, d]
                # solve for next parameter
                p_delta = torch.linalg.solve(G, b)
            model.param.data -= p_delta
    E = model.forward()


class LocalOptimization_GGN:
    """
    Generalized Gauss Newton
    """
    def __init__(self, N:int, model:nn.Module, loss_f:Callable, damping_mult=1e-5, max_iterations=50):
        """
        N -- number of points
        loss_f(EE) --- loss to miminize, see example below
        model --- nn.module with forward defining essential matrx of shape [3, 3]
        """
        self.loss_f = loss_f
        self.damping_mult = damping_mult
        self.max_iterations = max_iterations
        self.model = model
        self.N = N
        self.loss = None # current best loss
        self.ss = None # loss per point
        self.ff = None # residual per point
        self.ww = None # weight per point

    def __iter__(self):
        self.iteration = 0
        return self

    def __next__(self):
        """
        return:
        current model E [3, 3], current loss
        """
        if self.iteration >= self.max_iterations:
            raise StopIteration
        N = self.N
        
        if self.loss is None:
            E = self.model.forward() # [3,3]
            self.loss, EE, ff, ww  = self.loss_f(E) # [N, 3, 3], [N], [N]
        else:
            EE = self.EE
            ff = self.ff
            ww = self.ww
        J2 = self.model.Jacobian()  # [9, *, 7]
        
        J1 = torch.autograd.grad(ff.sum(), EE, retain_graph=True)[0]  # [*, N, 3, 3] -- Gradient of all residuals in E
        J1 = J1.flatten(start_dim=-2)  # [*, N, 9]
        #
        self.model.param.grad = None
        self.loss.backward()
        # Jacobian of E in model parameters
        # compute sum_i w_i J1_i J1_i.T
        G = torch.einsum('...ni, ...nj,...n -> ...ij', J1, J1, 1/ww)*2  # [*, 9, 9]
        # compute J2 G J2.T: sum_jk J2_{i...j} G_{i...k} J2_{k...l}
        G = torch.einsum('i...j, ...ik, k...l -> ...jl', J2, G, J2)  # [*, d, d], d = 7 -- marameters dimension
        # gradient:
        g = self.model.param.grad # [*, d]
        p0 = self.model.param.clone() # [d]
        for it in range(20):
            # try with current damping, until get an improvmeent
            m = self.damping_mult*N
            GD = G.clone()
            # damping (we have a non-minimal parameterization)
            GDdiag = torch.diagonal(GD, dim1=-1, dim2=-2) ## a view of the diagonal
            GDdiag[:] = GDdiag[:]*(1+1e-4*m) + 1e-3*m
            # GDdiag[:] = GDdiag[:] + 1e-3*m
            # solve for critcal point of quadratic approx
            p_delta = -torch.linalg.solve(GD, g)
            # step, to that critical point
            self.model.param.data = p0 + p_delta.squeeze(0)
            # check improvement
            new_E = self.model.forward() # current model
            new_loss, new_EE, new_ff, new_ww  = self.loss_f(new_E)
            if new_loss >= self.loss: # did not improve over current s, backtrack:
                self.model.param.data = p0
                self.damping_mult *= 3 # more damping
                print('<', end='')
            else: # successfully improved
                print('>', end='')
                self.damping_mult /= 1.5 # less damping
                # s has improved, remember as current best
                self.loss = new_loss # keep it differentiable
                self.EE = new_EE
                self.ff = new_ff
                self.ww = new_ww
                self.iteration += 1
                return new_E, new_loss # current model and total loss
        raise StopIteration #converged



class LocalOptimization_GGN_Batch:
    """
    Generalized Gauss Newton
    description: https://snip.mathpix.com/shekhovtsov/notes/ggn2-b7885cd3-c709-4964-9e67-60abdd32a3ba
    See also for GGN: 
    Martens (2017) New Insights and Perspectives on the Natural Gradient Method
    Schraudolph (2002) Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent
    """
    def __init__(self, model:nn.Module, loss_f:Callable, damping_mult=1e-5, max_iterations=50):
        """
        N -- number of points
        loss_f(EE) --- loss to miminize, see example below
        model --- nn.module with forward defining essential matrx of shape [3, 3]
        """
        self.loss_f = loss_f
        self.max_iterations = max_iterations
        self.model = model
        #
        with torch.no_grad():
            E = self.model.forward() # [...,3,3]
            self.loss, EE, ff, ww  = self.loss_f(E)# [...], [..., N, 3, 3], [..., N], [..., N]
            self.N = ff.shape[-1]
            B = self.loss.shape #
            self.damping_mult = self.loss.new_ones(B)*damping_mult

    def __iter__(self):
        # torch._functorch.config.donated_buffer=False 
        self.iteration = 0
        return self

    # @torch.compile(dynamic = True, backend = "inductor", mode="reduce-overhead")
    def __next__(self):
        """
        return:
        current model E [3, 3], current loss
        """
        if self.iteration >= self.max_iterations:
            raise StopIteration
        N = self.N

        E = self.model.forward() # [...,3,3]
        self.loss, EE, ff, ww  = self.loss_f(E)# [...], [..., N, 3, 3], [..., N], [..., N]

        J2 = self.model.Jacobian()  # [9, *, 7]
        
        J1 = torch.autograd.grad(ff.sum(), EE, retain_graph=True)[0]  # [*, N, 3, 3] -- Gradient of all residuals in E
        J1 = J1.flatten(start_dim=-2)  # [*, N, 9]
        #
        self.model.param.grad = None
        self.loss.sum().backward()
        # Jacobian of E in model parameters
        # compute sum_i w_i J1_i J1_i.T
        G = torch.einsum('...ni, ...nj,...n -> ...ij', J1, J1, 1/ww.detach())  # [*, 9, 9]
        # compute J2 G J2.T: sum_jk J2_{i...j} G_{i...k} J2_{k...l}
        G = torch.einsum('i...j, ...ik, k...l -> ...jl', J2, G, J2) * 2 # [*, d, d], d = 7 -- parameters dimension
        # gradient:
        g = self.model.param.grad # [*, d]
        p0 = self.model.param.clone() # [d]
        # try with current damping, until get an improvmeent
        m = self.damping_mult.unsqueeze(-1)*N
        # damping (we have a non-minimal parameterization)
        Gdiag = torch.diagonal(G, dim1=-1, dim2=-2) ## a view of the diagonal
        Gdiag[:] = Gdiag[:]*(1+1e-4*m) + 1e-3*m
        p_delta = -torch.linalg.solve(G, g)
        # step, to that critical point
        # self.model.param.data = p0 + p_delta.squeeze(0)
        self.model.step(p_delta.squeeze(0))
        # check improvement
        with torch.no_grad():
            new_E = self.model.forward() # current model
            new_loss, new_EE, new_ff, new_ww  = self.loss_f(new_E)

            self.loss = self.loss.detach()
            # models to backtrack
            mask_backtrack = new_loss >= self.loss # [B]
            self.model.param.data[mask_backtrack] = p0[mask_backtrack]
            self.damping_mult[mask_backtrack] *= 3 # more damping
            # models to accept
            mask_accept = torch.logical_not(mask_backtrack)
            self.damping_mult[mask_accept] /= 1.5 # more damping
            self.loss[mask_accept] = new_loss[mask_accept]
            E[mask_accept] = new_E[mask_accept]

        self.iteration += 1
        return E, self.loss
# %%
def rotation_matrix_y(angle_rad):
  """
  Creates a rotation matrix around the y-axis in PyTorch.

  Args:
    angle_rad: The rotation angle in radians.

  Returns:
    A 3x3 rotation matrix as a PyTorch tensor.
  """
  cos_theta = torch.cos(angle_rad)
  sin_theta = torch.sin(angle_rad)

  rotation_matrix = torch.tensor([
      [cos_theta, 0, sin_theta],
      [0, 1, 0],
      [-sin_theta, 0, cos_theta]
  ])

  return rotation_matrix


# %%
if __run__:
    torch.manual_seed(1)
    # Example usage:
    angle_degrees = 10
    angle_radians = torch.deg2rad(torch.tensor(angle_degrees))
    R = rotation_matrix_y(angle_radians).to(torch.float64)
    t = torch.ones(3, dtype = torch.float64)
    E = compose_essential_matrix(R, t)
    # E = torch.rand(3,3, dtype= torch.float64)
    R, R2, t = kornia.geometry.epipolar.decompose_essential_matrix(E)
    t = t.squeeze(-1)
    R = R.squeeze(0)
    R2 = R2.squeeze(0)
    E = compose_essential_matrix(R, -t)
    print(E.trace())
    # E1 = compose_essential_matrix(R2, t)
    # print(E / E1)
    # N = 20
    a_values = torch.linspace(-1, 1, 100)
    b_values = torch.linspace(-1, 1, 100)
    a, b = torch.meshgrid(a_values, b_values)
    X = torch.stack([a,b, torch.ones(a.shape, dtype= torch.float64)], dim=-1)

    # X = torch.cat([torch.randn(N,2, dtype= torch.float64), torch.ones(N,1, dtype= torch.float64)], dim=-1) # [N 3]
    # Y = (X) @ R.T - t
    # test pure rotation
    Y = (X) @ R.T # if we use R here, then R2 is not the other GT solution
    # Y = Y / Y[:, 2:]
    # X and Y are noise-free forrespondences
    # check they satisfy epipolar constraint:
    f = torch.einsum('...i,ij,...j->...',Y, E, X)
    assert((f.abs() < 1e-6).all())

    Xn = (X + torch.rand(X.shape, dtype=X.dtype)*0.000).view([-1, 3])
    Yn = (Y + torch.rand(X.shape, dtype=Y.dtype)*0.000).view([-1, 3])
    N = Xn.shape[0]

    model = E_parameterization(E)

    def test_loss(E:Tensor):
        EE = unsqueeze_expand(E, -3, N)  # [...,N, 3, 3]
        # Sampson Error
        yE = torch.einsum('...i, ...ij -> ...j', Yn, EE)  # [*, N, 3]
        Ex = torch.einsum('...j, ...ij -> ...i', Xn, EE)  # [*, N, 3]
        numerator = torch.einsum('...i, ...i', yE, Xn) # [*, N]
        denom = (((yE[..., 0:2])**2 + (Ex[..., 0:2])** 2).sum(dim=-1))**0.5  # [*, N]
        if False: # option 1: denominator differentiated
            ff = numerator/denom  # [*, N]
            ww =  torch.ones(ff.shape).to(ff)
        else: # option 2: denominator not differentiated
            ff = numerator
            ww = (denom**2).detach()
        # ff = torch.einsum('ni,nj,...nij->...n', Yn, Xn, EE) # algebraic error
        c = torch.zeros(ff.shape).to(ff)
        losses = ff**2 / ww + c
        return losses.sum(dim=-1), EE, ff, ww
    
    def loss():
        E = model.forward()
        ll , _, _, _ = test_loss(E)
        return ll.sum(dim=-1)

    print('Loss at GT:', loss().item())
    # single model
    batched = False
    torch.manual_seed(4)
    E_start  = E + torch.rand(3,3, dtype= torch.float64)*0.2
    U,S,V = torch.linalg.svd(E_start)
    S[:] = torch.tensor((1,1,0))
    E_start = U @ torch.diag(S) @ V
    model = E_parameterization(E_start)
    # batch of models
    # model = E_parameterization(E.unsqueeze(0) + torch.rand([5, 3, 3], dtype= torch.float64)*0.2)
    print('Loss at perturbed init:', loss().item())

    print('GGN')
    from .metrics import R_error
    GGN = LocalOptimization_GGN_Batch(model, test_loss, damping_mult=1e-5, max_iterations=10)
    for it, (E, l) in enumerate(GGN):
        Rm = model.R
        # rot_err = torch.min(R_error(Rm, R), R_error(Rm, R2)).item()
        rot_err = R_error(Rm, R)
        print(it, f'loss = {l.item()}, rot err = {rot_err:4.3f} deg')
    # model.R = R
    # print('Loss at GT R:', loss().item())
    # relative rotation:
    rR = R @ Rm.T
    v = TR.from_matrix(rR.detach()).as_rotvec()
    print("relative rot = ", v / np.linalg.norm(v))
    # 
    Y1 = X @ Rm.T
    t = model.t
    print(t)
    Tx = kornia.geometry.epipolar.cross_product_matrix(t)
    mixprod = torch.einsum('...i,...j,ij->...', Y, Y1, Tx) # algebraic error
    print(mixprod.abs().max())

    from scipy.spatial.transform import Rotation as TR
    import numpy as np
    vR = TR.from_matrix(R.detach()).as_rotvec()
    print("GT rotvect 1 = ", vR / np.linalg.norm(vR))
    # vR = TR.from_matrix(R2.detach()).as_rotvec()
    # print("GT rotvect 2 = ", vR / np.linalg.norm(vR))
    v = TR.from_matrix(Rm.detach()).as_rotvec()
    print("found rotvect = ", v / np.linalg.norm(v))


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    w = torch.linalg.cross(Y, Y1).detach().numpy()
    ax.plot_surface(w[:,:,0], w[:,:,1], w[:,:,2], cmap='viridis', edgecolor='none')
    ax.set_title('Manifold of the Cross Product of u and v')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    
    # print('================')
    # print('SGD test')
    # opt = torch.optim.Adam([model.param], lr=1e-3, betas = (0.0,0.99))
    # opt = torch.optim.SGD([model.param], lr=1e-4, momentum=0.0)
    # for it in range(2000):
    #     opt.zero_grad()
    #     l = loss()
    #     l.backward()
    #     opt.step()
    #     if it %100 == 0:
    #         print(l.item())

    # Rm = model.R
    # rot_err = torch.min(R_error(Rm, R), R_error(Rm, R2)).item()
    # print(it, f'loss = {l.item()}, rot err = {rot_err:4.3f} deg')

    # print('GGN')
    # # GGN = LocalOptimization_GGN(N,model, test_loss, damping_mult=1e-3)
    # GGN = LocalOptimization_GGN_Batch(model, test_loss, damping_mult=1e-5, max_iterations=10)
    # for it, (E, l) in enumerate(GGN):
    #     Rm = model.R
    #     rot_err = torch.min(R_error(Rm, R), R_error(Rm, R2)).item()
    #     print(it, f'loss = {l.item()}, rot err = {rot_err:4.3f} deg')
# %%
