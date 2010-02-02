# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Finite difference operators.

This file defines a series of finite difference operators used in grid mode.
"""

from __future__ import division
from math import pi

import numpy as np

from gpaw import debug, extra_parameters
from gpaw.utilities import contiguous, is_contiguous, fac
import _gpaw

class _FDOperator:
    def __init__(self, coef_p, offset_pc, gd, dtype=float,
                 allocate=True):
        """FDOperator(coefs, offsets, gd, dtype) -> FDOperator object.
        """

        # Is this a central finite-difference type of stencil?
        cfd = True
        for offset_c in offset_pc:
            if sum([offset != 0 for offset in offset_c]) >= 2:
                cfd = False

        maxoffset_c = [max([offset_c[c] for offset_c in offset_pc])
                       for c in range(3)]

        mp = maxoffset_c[0]
        if maxoffset_c[1] != mp or maxoffset_c[2] != mp:
##            print 'Warning: this should be optimized XXXX', maxoffsets, mp
            mp = max(maxoffset_c)
        n_c = gd.n_c
        M_c = n_c + 2 * mp
        stride_c = np.array([M_c[1] * M_c[2], M_c[2], 1])
        offset_p = np.dot(offset_pc, stride_c)
        coef_p = contiguous(coef_p, float)
        neighbor_cd = gd.neighbor_cd
        assert np.rank(coef_p) == 1
        assert coef_p.shape == offset_p.shape
        assert dtype in [float, complex]
        self.dtype = dtype
        self.shape = tuple(n_c)
        
        if gd.comm.size > 1:
            comm = gd.comm.get_c_object()
        else:
            comm = None

        self.operator = None
        self.args = [coef_p, offset_p, n_c, mp,
                     neighbor_cd, dtype == float,
                     comm, cfd]
        self.mp = mp # padding
        self.gd = gd
        self.npoints = len(coef_p)

        self.allocated = False
        if allocate:
            self.allocate()

    def allocate(self):
        assert not self.is_allocated()
        self.operator = _gpaw.Operator(*self.args)
        self.args = None
        self.allocated = True

    def is_allocated(self):
        return self.allocated

    def apply(self, in_xg, out_xg, phase_cd=None):
        self.operator.apply(in_xg, out_xg, phase_cd)

    def apply2(self, in_xg, out_xg, phase_cd=None):
        self.operator.apply2(in_xg, out_xg, phase_cd=None)

    def relax(self, relax_method, f_g, s_g, n, w=None):
        self.operator.relax(relax_method, f_g, s_g, n, w)

    def get_diagonal_element(self):
        return self.operator.get_diagonal_element()

    def get_async_sizes(self):
        return self.operator.get_async_sizes()

    def estimate_memory(self, mem):
        bufsize_c = np.array(self.gd.n_c) + 2 * self.mp
        itemsize = mem.itemsize[self.dtype]
        mem.setsize(np.prod(bufsize_c) * itemsize)


class FDOperatorWrapper:
    def __init__(self, operator):
        self.operator = operator
        self.shape = operator.shape
        self.dtype = operator.dtype
        self.npoints = operator.npoints

    def is_allocated(self):
        return self.operator.is_allocated()

    def apply(self, in_xg, out_xg, phase_cd=None):
        assert in_xg.shape == out_xg.shape
        assert in_xg.shape[-3:] == self.shape
        assert is_contiguous(in_xg, self.dtype)
        assert is_contiguous(out_xg, self.dtype)
        assert (self.dtype == float or
                (phase_cd.dtype == complex and
                 phase_cd.shape == (3, 2)))
        self.operator.apply(in_xg, out_xg, phase_cd)

    def apply2(self, in_xg, out_xg, phase_cd=None):
        assert in_xg.shape == out_xg.shape
        assert in_xg.shape[-3:] == self.shape
        assert is_contiguous(in_xg, self.dtype)
        assert is_contiguous(out_xg, self.dtype)
        assert (self.dtype == float or
                (phase_cd.dtype == complex and
                 phase_cd.shape == (3, 2)))
        self.operator.apply2(in_xg, out_xg, phase_cd)

    def relax(self, relax_method, f_g, s_g, n, w=None):
        assert f_g.shape == self.shape
        assert s_g.shape == self.shape
        assert is_contiguous(f_g, float)
        assert is_contiguous(s_g, float)
        assert self.dtype == float
        self.operator.relax(relax_method, f_g, s_g, n, w)
        
    def get_diagonal_element(self):
        return self.operator.get_diagonal_element()

    def get_async_sizes(self):
        return self.operator.get_async_sizes()

    def estimate_memory(self, mem):
        self.operator.estimate_memory(mem)

    def allocate(self):
        self.operator.allocate()


if debug:
    def FDOperator(coef_p, offset_pc, gd, dtype=float, allocate=True):
        return FDOperatorWrapper(_FDOperator(coef_p, offset_pc, gd, dtype,
                                             allocate))
else:
    FDOperator = _FDOperator


def Gradient(gd, v, scale=1.0, n=1, dtype=float, allocate=True):
    h = (gd.h_cv**2).sum(1)**0.5
    d = gd.xxxiucell_cv[:,v]
    A=np.zeros((2*n+1,2*n+1))
    for i,io in enumerate(range(-n,n+1)):
        for j in range(2*n+1):
            A[i,j]=io**j/float(fac[j])
    A[n,0]=1.
    coefs=np.linalg.inv(A)[1]
    coefs=np.delete(coefs,len(coefs)//2)
    offs=np.delete(np.arange(-n,n+1),n)
    coef_p = []
    offset_pc = []
    for i in range(3):
        if abs(d[i])>1e-11:
            coef_p.extend(list(coefs * d[i] / h[i] * scale))
            offset = np.zeros((2*n, 3), int)
            offset[:,i]=offs
            offset_pc.extend(offset)

    if(n > len(fac)):
        raise RuntimeError('extend fac')

    return FDOperator(coef_p, offset_pc, gd, dtype, allocate)

# Expansion coefficients for finite difference Laplacian.  The numbers are
# from J. R. Chelikowsky et al., Phys. Rev. B 50, 11355 (1994):
laplace = [[0],
           [-2, 1],
           [-5/2, 4/3, -1/12],
           [-49/18, 3/2, -3/20, 1/90],
           [-205/72, 8/5, -1/5, 8/315, -1/560],
           [-5269/1800, 5/3, -5/21, 5/126, -5/1008, 1/3150],
           [-5369/1800, 12/7, -15/56, 10/189, -1/112, 2/1925, -1/16632]]

# Cross term coefficients
# given in (1,1),(1,2),...,(1,n),(2,2),(2,3),...,(2,n),...,(n,n) order
# Calculated with Mathematica as:
# FD[n_, m_, o_] :=
# Simplify[NDSolve`FiniteDifferenceDerivative[
#         Derivative[n, m], {ha Range[-2*o, 2*o], hb Range[-2*o, 2*o]},
#             Table[f[x, y], {x, -2*o, 2*o}, {y, -2*o, 2*o}],
#             DifferenceOrder -> 2*o]][[2*o + 1, 2*o + 1]]
# e.g Nth order coeffiecients are given by FD[1, 1, N]
cross = [[0],
         [1/4],
         [4/9, -1/18, 1/144],
         [9/16, -9/80, 1/80, 9/400, -1/400, 1/3600],
         [16/25, -4/25, 16/525, -1/350, 1/25, -4/525, 1/1400, 16/11025, -1/7350, 1/78400],
         [25/36, -25/126, 25/504, -25/3024, 1/1512, 25/441, -25/1764, 25/10584, -1/5292, 25/7056, -25/42336, 1/21168, 25/254016, -1/127008, 1/1587600],
         [36/49, -45/196, 10/147, -3/196, 6/2695, -1/6468, 225/3136, -25/1176, 15/3136, -3/4312, 5/103488, 25/3969, -5/3528, 1/4851, -5/349272, 1/3136, -1/21560, 1/310464, 1/148225, -1/2134440, 1/30735936]]

# Check numbers:
if debug:
    for coefs in laplace:
        assert abs(coefs[0] + 2 * sum(coefs[1:])) < 1e-11


def Laplace(gd, scale=1.0, n=1, dtype=float, allocate=True):
    """Central finite diference Laplacian.

    Uses (max) 12*n**2 + 6*n neighbors."""

    if n == 9:
        return FTLaplace(gd, scale, dtype)
    n = int(n)
    h = (gd.h_cv**2).sum(1)**0.5
    h2 = h**2
    iucell_cv = gd.xxxiucell_cv

    d2 = (iucell_cv**2).sum(1)

    offsets = [(0, 0, 0)]
    coefs = [scale * np.sum(d2 * np.divide(laplace[n][0],h2))]

    for d in range(1, n + 1):
        offsets.extend([(-d, 0, 0), (d, 0, 0),
                        (0, -d, 0), (0, d, 0),
                        (0, 0, -d), (0, 0, d)])
        c = scale * d2 * np.divide(laplace[n][d], h2)

        coefs.extend([c[0], c[0],
                      c[1], c[1],
                      c[2], c[2]])

    #cross-partial derivatives
    ci=0

    for d1 in range(n):
        for d2 in range(d1,n):

            offset=[[( d1+1, d2+1, 0   ),(-d1-1 , d2+1 , 0   ),( d1+1 ,-d2-1, 0   ),(-d1-1,-d2-1,0    )],
                    [( 0   , d1+1, d2+1),( 0    ,-d1-1 , d2+1),( 0    , d1+1,-d2-1),( 0   ,-d1-1,-d2-1)],
                    [( d2+1, 0   , d1+1),( d2+1 , 0    ,-d1-1),(-d2-1 , 0   , d1+1),(-d2-1,0    ,-d1-1)]]

            for i in range(3):
                c = scale * 2. * cross[n][ci] * np.dot(iucell_cv[i],iucell_cv[(i+1)%3]) / (h[i]*h[(i+1)%3])

                if abs(c)>1E-11: #extend stencil only to points of non zero coefficient
                    offsets.extend(offset[i])
                    coefs.extend([c,-c,-c,c])

                    if (d2>d1):  #extend stencil to symmetric points (ex. [1,2,3] <-> [2,1,3])
                        ind=[0,1,2]; ind[i]=(i+1)%3; ind[(i+1)%3]=i
                        offsets.extend([tuple(np.take(offset[i][i2],ind)) for i2 in range(4)])
                        coefs.extend([c,-c,-c,c])

            ci+=1

    return FDOperator(coefs, offsets, gd, dtype, allocate)

from numpy.fft import fftn, ifftn

class FTLaplace:
    def __init__(self, gd, scale, dtype):
        assert gd.comm.size == 1 and gd.pbc_c.all()

        N_c1 = gd.N_c[:, np.newaxis]
        i_cq = np.indices(gd.N_c).reshape((3, -1))
        i_cq += N_c1 // 2
        i_cq %= N_c1
        i_cq -= N_c1 // 2
        B_vc = 2.0 * pi * gd.icell_cv.T
        k_vq = np.dot(B_vc, i_cq)
        k_vq *= k_vq
        self.k2_Q = k_vq.sum(axis=0).reshape(gd.N_c)
        self.k2_Q *= -scale
        self.d = 6.0 / gd.h_cv[0, 0]**2
        
    def apply(self, in_xg, out_xg, phase_cd=None):
        if in_xg.ndim > 3:
            for in_g, out_g in zip(in_xg, out_xg):
                out_g[:] = ifftn(fftn(in_g) * self.k2_Q).real
        else:
            out_xg[:] = ifftn(fftn(in_xg) * self.k2_Q).real

    def get_diagonal_element(self):
        return self.d

    def allocate(self):
        pass

    def estimate_memory(self, mem):
        mem.subnode('FTLaplace estimate not implemented', 0)


def LaplaceA(gd, scale, dtype=float, allocate=True):
    assert gd.orthogonal
    c = np.divide(-1/12, gd.h_cv.diagonal()**2) * scale  # Why divide? XXX
    c0 = c[1] + c[2]
    c1 = c[0] + c[2]
    c2 = c[1] + c[0]
    a = -16.0 * np.sum(c)
    b = 10.0 * c + 0.125 * a
    return FDOperator([a,
                       b[0], b[0],
                       b[1], b[1],
                       b[2], b[2],
                       c0, c0, c0, c0,
                       c1, c1, c1, c1,
                       c2, c2, c2, c2], 
                      [(0, 0, 0),
                       (-1, 0, 0), (1, 0, 0),
                       (0, -1, 0), (0, 1, 0),
                       (0, 0, -1), (0, 0, 1),
                       (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                       (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
                       (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0)],
                      gd, dtype, allocate=allocate)

def LaplaceB(gd, dtype=float, allocate=True):
    a = 0.5
    b = 1.0 / 12.0
    return FDOperator([a,
                       b, b, b, b, b, b],
                      [(0, 0, 0),
                       (-1, 0, 0), (1, 0, 0),
                       (0, -1, 0), (0, 1, 0),
                       (0, 0, -1), (0, 0, 1)],
                      gd, dtype, allocate=allocate)

def GUCLaplace(gd, scale=1.0, n=1, dtype=float, allocate=True):
    """Laplacian for general nororthorhombic grid.

    gd: GridDescriptor
        Descriptor for grid.
    scale: float
        Scaling factor.  Use scale=-0.5 for a kinetic energy operator.
    n: int
        Range of stencil.  Stencil has O(h^(2n)) error.
    dtype: float or complex
        Datatype to work on.
    allocate: bool
        Allocate work arrays.
    """

    if n == 9:
        return FTLaplace(gd, scale, dtype)
    
    # Order the 13 neighbor grid points:
    M_ic = np.indices((3, 3, 3)).reshape((3, -3)).T[-13:] - 1
    h2_i = (np.dot(M_ic, gd.h_cv)**2).sum(1)
    i_d = h2_i.argsort()

    
    m_mv = np.array([(2, 0, 0), (0, 2, 0), (0, 0, 2),
                     (0, 1, 1), (1, 0, 1), (1, 1, 0)])
    # Try 3, 4, 5 and 6 directions:
    for D in range(3, 7):
        h_dv = np.dot(M_ic[i_d[:D]], gd.h_cv)
        A_md = (h_dv**m_mv[:, np.newaxis, :]).prod(2)
        a_d, residual = np.linalg.lstsq(A_md, [1, 1, 1, 0, 0, 0])[:2]
        if residual.sum() < 1e-14:
            # D directions was OK
            break

    a_d *= scale
    offsets = [(0,0,0)]
    coefs = [laplace[n][0] * a_d.sum()]
    for d in range(D):
        M_c = M_ic[i_d[d]]
        offsets.extend(np.arange(1, n + 1)[:, np.newaxis] * M_c)
        coefs.extend(a_d[d] * np.array(laplace[n][1:]))
        offsets.extend(np.arange(-1, -n - 1, -1)[:, np.newaxis] * M_c)
        coefs.extend(a_d[d] * np.array(laplace[n][1:]))

    return FDOperator(coefs, offsets, gd, dtype, allocate)


if extra_parameters.get('newgucstencil'):
    Laplace = GUCLaplace
