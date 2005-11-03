# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import sqrt

import Numeric as num
import LinearAlgebra as linalg

from gridpaw.spherical_harmonics import Y



s = sqrt(0.5)
# Points on the unit sphere:
sphere_lm = [ \
    num.array([(1, 0, 0)]), # s
    num.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)]), # p
    num.array([(s, s, 0), (0, s, s), (s, 0, s), (1, 0, 0), (0, 0, 1)])] # d

def Y_matrix(l, symmetry):
    """YMatrix(l, symmetry) -> matrix.

    For 2*l+1 points on the unit sphere (m1=0,...,2*l) calculate the
    value of Y_lm2 for m2=0,...,2*l.  The points are those from the
    list sphere_lm[l] rotated as described by symmetry = (swap,
    mirror)."""
    
    swap, mirror = symmetry
    Y_m1m2 = num.zeros((2 * l + 1, 2 * l + 1), num.Float)
    for m1, point in enumerate(sphere_lm[l]):
        x, y, z = num.take(point * mirror, swap)
        for m2 in range(2 * l + 1):
            L = l**2 + m2
            Y_m1m2[m1, m2] = Y(L, x, y, z)
    return Y_m1m2


identity = ((0, 1, 2), (1, 1, 1))
iY_lm1m2 = [linalg.inverse(Y_matrix(l, identity)) for l in range(3)]
         

def rotation(l, symmetry):
    """Rotation(l, symmetry) -> transformation matrix.

    Find the transformation from Y_lm1 to Y_lm2."""
    
    return num.dot(iY_lm1m2[l], Y_matrix(l, symmetry))


del s, identity
