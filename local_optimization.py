import os, sys
if  __name__ == "__main__":
    __name__ = 'score_learn.local_optimization'
    __package__ = 'score_learn'
    __run__ = True
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    sys.path.append(os.path.dirname(dname))
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
        # 6 parameters per essential matrix
        self.param = E.new_zeros(self.bs + [7], dtype=torch.float64)
        self.param.requires_grad = True
        self.t = t.squeeze(-1)  # set translation
        self.R = R1.to(self.param)  # any is good
        # rotatino is modelled incrementally to R
        # set quaternion to [1,0,0,0] -- identity rotation
        self.q.data[..., 0] = 1
        E1 = self.forward()  # up to scale, posibly also negative
        pass

    @property
    def t(self):
        return self.param[..., 0:3]

    @t.setter
    def t(self, v):
        self.param.data[..., 0:3] = v

    @property
    def q(self):
        return self.param[..., 3:]

    # @q.setter
    # def q(self, v):
        # self.param.data[..., 3:] = v

    def forward(self):
        # compose back essential matrix with parametric translation and rotation
        # v = self.v
        # s = list(self.v.shape)
        # s[-1] = 1
        # v = torch.cat((v.new_ones(s), v), dim=-1)
        # q = F.normalize(v, dim=-1)
        t = F.normalize(self.t, dim=-1) # translation vector is up to global scale
        q = F.normalize(self.q, dim=-1) # quaternion vector must be normalized
        dR = kornia.geometry.conversions.quaternion_to_rotation_matrix(q)
        # batched matrix multiplication
        R = torch.einsum('...ij, ...jk -> ...ik', self.R, dR)
        E = compose_essential_matrix(R, t)
        return E

    # def Jacobian(s§elf):
    #     params = dict(self.named_parameters())

    #     def func(pp):
    #         F = torch.func.functional_call(self, pp, tuple())
    #         F = F.flatten(start_dim=-2)
    #         return F.sum(dim=0)  # for each batch dimension, independent inputs
    #     J_f = torch.func.jacrev(func)
    #     J = J_f(params)["param"]
    #     return J
    def Jacobian(self):
        def func(param):
            p = self.param
            self.param = param
            F = self.forward()
            self.param = p
            F = F.flatten(start_dim=-2)
            return F.sum(dim=0)  # for each batch dimension, independent inputs
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


class LocalOptimization_LMA:
    """
    Optimize sum of squared residuals, non-linearly dependent on the geometric model, non-linearly parameterized
    """
    def __init__(self, N:int, model:nn.Module, residual_f:Callable, score_f:Callable=None, weight_f: Callable = None, damping_mult=1):
        """
        N -- number of points
        residual_f(EE) --- should return [N] residuals, one per data point, provided essential matrix E, expanded to N copies as [N, 3, 3]
        model --- nn.module with forward defining essential matrx of shape [3, 3]
        score_f and weight_f will be needed in case of optimizing the robust version of the loss
            score_f --- computes the robust score
            weight_f --- computes probabilities of points being inliers
        """
        def default_score(rr):
            return rr**2
        
        def default_weight(rr):
            return torch.ones_like(rr)

        self.residual_f = residual_f
        self.score_f = score_f if score_f is not None else default_score
        self.weight_f = weight_f  if weight_f is not None else default_weight
        self.damping_mult = damping_mult
        self.model = model
        self.N = N
        self.s = None # current best score
        self.rr = None
        self.p = None # model params copy

    def compute_residuals_and_scores(self, E):
        # separate copy per point to compute separate gradient
        EE = unsqueeze_expand(E, -3, self.N)  # [*, N, 3, 3]
        rr = self.residual_f(EE)
        s = self.score_f(rr).sum(dim=-1)  # [*]
        return EE, rr, s 

    def iteration(self):
        # evaluate score
        """
        return:
        current model E [3, 3], current score
        """
        N = self.N
        
        if self.s is None:
            E = self.model.forward()
            EE, rr, self.s = self.compute_residuals_and_scores(E)
            self.p = self.model.param.clone()
        else:
            EE = self.EE
            rr = self.rr
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
            weight = self.weight_f(rr)
        rr = rr /1000 # just to fix the scale for regularization
        J1 = torch.autograd.grad(rr.sum(), EE, retain_graph=True)[0]  # [*, N, 3, 3] -- Gradient of all residuals in E
        assert (not J1.isnan().any())
        J1 = J1.flatten(start_dim=-2)  # [*, N, 9]
        # Jacobian of E in model parameters
        J2 = self.model.Jacobian()  # [9, *, 7]
        assert (not J2.isnan().any())
        with torch.no_grad():
            # apply it on J1
            J = torch.einsum('i...j, ...ni -> ...nj', J2, J1)  # [*, N, d], d = 7 -- marameters dimension
            # compose JJ'
            G = torch.einsum('...ni, ...nj,...n -> ...ij', J, J, weight)  # [*, d, d]
            # compute J f
            b = torch.einsum('...ni, ...n, ...n -> ...i', J, rr, weight)  # [*, d]            
        while True:
            # try with current damping, until get an improvmeent
            m = self.damping_mult                
            GD = G.clone()
            # damping (we have a non-minimal parameterization)
            GDdiag = torch.diagonal(GD, dim1=-1, dim2=-2) ## a view of the diagonal
            GDdiag[:] = GDdiag[:]*(1+1e-4*m) + 1e-3*m    
            # solve for next parameter
            p_delta = torch.linalg.solve(G, b)
            # step, maximization
            self.model.param.data -= p_delta
            # check improvement
            new_E = self.model.forward() # current model
            new_EE, new_rr, new_s = self.compute_residuals_and_scores(new_E)
            if new_s < self.s0: # did not improve over s0, backtrack:
                self.model.param.data = self.p
                self.damping_mult *= 3 # more damping
            else: # successfully improved
                self.damping_mult *= 1.5 # less damping                
                # s has improved, remember as current best
                self.s = new_s.detach().clone()
                self.p = self.model.param.clone()
                self.EE = new_EE
                self.rr = new_rr
                return new_E, new_s


class LocalOptimization_GGN:
    """
    Generalized Gauss Newton
    """
    def __init__(self, N:int, model:nn.Module, loss_f:Callable, damping_mult=1e-3, max_iterations=50):
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
            # GDdiag[:] = GDdiag[:]*(1+1e-4*m) + 1e-3*m
            GDdiag[:] = GDdiag[:] + 1e-3*m
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
                self.damping_mult *= 1.5 # less damping                
                # s has improved, remember as current best
                self.loss = new_loss # keep it differentiable
                self.EE = new_EE
                self.ff = new_ff
                self.ww = new_ww
                self.iteration += 1
                return new_E, new_loss # current model and total loss
        raise StopIteration #converged


if __run__:
    torch.manual_seed(0)
    E = torch.rand(3,3, dtype= torch.float64)
    R, _, t = kornia.geometry.epipolar.decompose_essential_matrix(E)
    t = t.squeeze(-1)
    R = R.squeeze(0)
    E = compose_essential_matrix(R, t)
    N = 100
    X = torch.cat([torch.rand(N,2, dtype= torch.float64), torch.ones(N,1, dtype= torch.float64)], dim=-1) # [N 3]
    Y = (X) @ R.T - t
    # X and Y are noise-free forrespondences
    # check they satisfy epipolar constraint:
    f = torch.einsum('ni,ij,nj->n',Y, E, X)
    assert((f.abs() < 1e-6).all())

    model = E_parameterization(E)

    def test_loss(E):
         # algebraic error
        EE = unsqueeze_expand(E, -3, N)  # [N, 3, 3]        
        ff = torch.einsum('ni,nj,...nij->...n', Y,X, EE)
        ww =  torch.ones(N).to(ff)
        c = torch.zeros(N).to(ff)
        losses = ff**2 / ww + c
        return losses.sum(dim=-1), EE, ff, ww
    
    def loss():
        E = model.forward()
        ll , _, _, _ = test_loss(E)
        return ll.sum()

    print('Loss at GT:', loss().item())
    model = E_parameterization(E + torch.rand(3,3, dtype= torch.float64)*0.2)
    print('Loss perturbed:', loss().item())

    print('GGN')
    GGN = LocalOptimization_GGN(N,model, test_loss, damping_mult=10)
    for it, (E, l) in enumerate(GGN):
        print(it, l.item())

    print('================')
    print('SGD (200 it)')
    # opt = torch.optim.Adam([model.param], lr=1e-3, betas = (0.0,0.99))
    opt = torch.optim.SGD([model.param], lr=1e-5, momentum=0.0)
    for it in range(200):
        opt.zero_grad()
        l = loss()
        l.backward()
        opt.step()
    print(l.item())

    print('GGN')
    GGN = LocalOptimization_GGN(N,model, test_loss, damping_mult=1000)
    for it, (E, l) in enumerate(GGN):
        print(it, l.item())