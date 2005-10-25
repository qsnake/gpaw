# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from __future__ import division
from math import pi

import Numeric as num

from gridpaw import debug
from gridpaw.utilities import contiguous, is_contiguous
import _gridpaw

    
class _Operator:
    def __init__(self, coefs, offsets, gd, cfd, typecode=num.Float):
        """Operator(coefs, offsets, gd, typecode) -> Operator object.
        """

        maxoffsets = [max([offset[axis] for offset in offsets])
                     for axis in range(3)]
        mp = maxoffsets[0]
        if maxoffsets[1] != mp or maxoffsets[2] != mp:
##            print 'Warning: this should be optimized XXXX', maxoffsets, mp
            mp = max(maxoffsets)
##        ng = num.array(gd.myng, typecode=num.Int)
        ng = gd.myng
        ng2 = ng + 2 * mp
        strides = num.array((ng2[1] * ng2[2], ng2[2], 1))
        offsets = num.dot(offsets, strides)
        coefs = contiguous(coefs, num.Float)
        neighbors = gd.domain.get_neighbor_processors()
        assert len(coefs.shape) == 1
        assert coefs.shape == offsets.shape
        assert typecode in [num.Float, num.Complex]
        self.typecode = typecode
        self.shape = tuple(gd.myng)
        
        if gd.domain.comm.size>1:
            comm = gd.domain.comm
            if debug:
                # get the C-communicator from the debug-wrapper:
                comm = comm.comm
        else:
            comm = None

        if gd.domain.angle is None:
            angle = 0
        else:
            angle = int(angle / (pi / 2) + 0.5)

        self.operator = _gridpaw.Operator(coefs, offsets, ng, mp,
                                          neighbors, typecode == num.Float,
                                          comm, cfd, angle)

    def apply(self, arrays, results, phases=None):
        assert arrays.shape == results.shape
        assert arrays.shape[-3:] == self.shape
        assert is_contiguous(arrays, self.typecode)
        assert is_contiguous(results, self.typecode)
        assert self.typecode is num.Float or (phases.typecode() == num.Complex and
                                         phases.shape == (6,))
        self.operator.apply(arrays, results, phases)

    def get_diagonal_element(self):
        return self.operator.get_diagonal_element()


if debug:
    Operator = _Operator
else:
    def Operator(coefs, offsets, gd, cfd, typecode=num.Float):
        return _Operator(coefs, offsets, gd, cfd, typecode).operator


def Gradient(gd, axis, scale=1.0, n=1, typecode=num.Float):
    h = gd.h[axis]
    a = 0.5 / h * scale
    coefs = [-a, a]
    offsets = num.zeros((2, 3))
    offsets[0, axis] = -1
    offsets[1, axis] = 1
    return Operator(coefs, offsets, gd, True, typecode)


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
    for c in laplace:
        sum = c[0]
        for x in c[1:]:
            sum += 2 * x
        assert abs(sum) < 1e-11


def Laplace(gd, scale=1.0, n=1, typecode=num.Float):
    """Central finite diference Laplacian.

    Uses 6*n neighbors."""
    
    h = gd.h
    h2 = h**2
    offsets = [(0, 0, 0)]
    coefs = [scale * num.sum(num.divide(laplace[n][0], h2))]
    for d in range(1, n + 1):
        offsets.extend([(-d, 0, 0), (d, 0, 0),
                        (0, -d, 0), (0, d, 0),
                        (0, 0, -d), (0, 0, d)])
        c = scale * num.divide(laplace[n][d], h2)
        coefs.extend([c[0], c[0],
                      c[1], c[1],
                      c[2], c[2]])
    return Operator(coefs, offsets, gd, True, typecode)


def LaplaceA(gd, scale, typecode=num.Float):
    c = num.divide(-1/12, gd.h**2) * scale
    c0 = c[1] + c[2]
    c1 = c[0] + c[2]
    c2 = c[1] + c[0]
    a = -16.0 * num.sum(c)
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
                    gd, False, typecode)

def LaplaceB(gd, typecode=num.Float):
    a = 0.5
    b = 1.0 / 12.0
    return Operator([a,
                     b, b, b, b, b, b],
                    [(0, 0, 0),
                     (-1, 0, 0), (1, 0, 0),
                     (0, -1, 0), (0, 1, 0),
                     (0, 0, -1), (0, 0, 1)],
                    gd, True, typecode)
