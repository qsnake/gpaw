# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from __future__ import division
from math import pi

import numpy as npy

from gpaw import debug
from gpaw.utilities import contiguous, is_contiguous
import _gpaw
    
class _Operator:
    def __init__(self, coef_p, offset_pc, gd, dtype=float):
        """Operator(coefs, offsets, gd, dtype) -> Operator object.
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
        stride_c = npy.array([M_c[1] * M_c[2], M_c[2], 1])
        offset_p = npy.dot(offset_pc, stride_c)
        coef_p = contiguous(coef_p, float)
        neighbor_cd = gd.domain.neighbor_cd
        assert npy.rank(coef_p) == 1
        assert coef_p.shape == offset_p.shape
        assert dtype in [float, complex]
        self.dtype = dtype
        self.shape = tuple(n_c)
        
        if gd.comm.size > 1:
            comm = gd.comm
            if debug:
                # get the C-communicator from the debug-wrapper:
                comm = comm.comm
        else:
            comm = None

        self.operator = _gpaw.Operator(coef_p, offset_p, n_c, mp,
                                          neighbor_cd, dtype == float,
                                          comm, cfd)

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


if debug:
    Operator = _Operator
else:
    def Operator(coef_p, offset_pc, gd, dtype=float):
        return _Operator(coef_p, offset_pc, gd, dtype).operator


def Gradient(gd, c, scale=1.0, dtype=float):
    h = gd.h_c[c]
    a = 0.5 / h * scale
    coef_p = [-a, a]
    offset_pc = npy.zeros((2, 3), int)
    offset_pc[0, c] = -1
    offset_pc[1, c] = 1
    return Operator(coef_p, offset_pc, gd, dtype)


# Expansion coefficients for finite difference Laplacian.  The numbers are
# from J. R. Chelikowsky et al., Phys. Rev. B 50, 11355 (1994):
laplace = [[0],
           [-2, 1],
           [-5/2, 4/3, -1/12],
           [-49/18, 3/2, -3/20, 1/90],
           [-205/72, 8/5, -1/5, 8/315, -1/560],
           [-5269/1800, 5/3, -5/21, 5/126, -5/1008, 1/3150],
           [-5369/1800, 12/7, -15/56, 10/189, -1/112, 2/1925, -1/16632]]

# Check numbers:
if debug:
    for coefs in laplace:
        assert abs(coefs[0] + 2 * sum(coefs[1:])) < 1e-11


def Laplace(gd, scale=1.0, n=1, dtype=float):
    """Central finite diference Laplacian.

    Uses 6*n neighbors."""

    n = int(n)
    h = gd.h_c
    h2 = h**2
    offsets = [(0, 0, 0)]
    coefs = [scale * npy.sum(npy.divide(laplace[n][0], h2))]
    for d in range(1, n + 1):
        offsets.extend([(-d, 0, 0), (d, 0, 0),
                        (0, -d, 0), (0, d, 0),
                        (0, 0, -d), (0, 0, d)])
        c = scale * npy.divide(laplace[n][d], h2)
        coefs.extend([c[0], c[0],
                      c[1], c[1],
                      c[2], c[2]])
    return Operator(coefs, offsets, gd, dtype)


def LaplaceA(gd, scale, dtype=float):
    c = npy.divide(-1/12, gd.h_c**2) * scale  # Why divide? XXX
    c0 = c[1] + c[2]
    c1 = c[0] + c[2]
    c2 = c[1] + c[0]
    a = -16.0 * npy.sum(c)
    b = 10.0 * c + 0.125 * a
    return Operator([a,
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
                    gd, dtype)

def LaplaceB(gd, dtype=float):
    a = 0.5
    b = 1.0 / 12.0
    return Operator([a,
                     b, b, b, b, b, b],
                    [(0, 0, 0),
                     (-1, 0, 0), (1, 0, 0),
                     (0, -1, 0), (0, 1, 0),
                     (0, 0, -1), (0, 0, 1)],
                    gd, dtype)
