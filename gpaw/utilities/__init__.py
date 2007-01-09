# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Utility functions and classes."""

import os
from math import sqrt

import Numeric as num

import _gpaw
from gpaw import debug


elementwise_multiply_add = _gpaw.elementwise_multiply_add
utilities_vdot = _gpaw.utilities_vdot
utilities_vdot_self = _gpaw.utilities_vdot_self


# Error function:
erf = _gpaw.erf


# Factorials:
fac = [1, 1, 2, 6, 24, 120, 720, 5040, 40320,
       362880, 3628800, 39916800, 479001600]


def gcd(a, b):
    """Return greatest common divisor of a and b, using the
       euclidian algorithm.
    """
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def contiguous(array, typecode):
    """Convert a sequence to a contiguous Numeric array."""
    array = num.asarray(array, typecode)
    if array.iscontiguous():
        return array
    else:
        return num.array(array)


def is_contiguous(array, typecode=None):
    """Check for contiguity and type."""
    if typecode is None:
        return array.iscontiguous()
    else:
        return array.iscontiguous() and array.typecode() == typecode


# Radial-grid Hartree solver:
#
#                       l
#             __  __   r
#     1      \   4||    <   * ^    ^
#   ------ =  )  ---- ---- Y (r)Y (r'),
#    _ _     /__ 2l+1  l+1  lm   lm
#   |r-r'|    lm      r
#                      >
# where
#
#   r = min(r, r')
#    <
#
# and
#
#   r = max(r, r')
#    >
#
if debug:
    def hartree(l, nrdr, beta, N, vr):
        """Calculates radial Coulomb integral.

        The following integral is calculated::
        
                                      ^
                             n (r')Y (r')
                  ^    / _    l     lm
          v (r)Y (r) = |dr --------------,
           l    lm     /       _   _
                              |r - r'|

        where input and output arrays `nrdr` and `vr`::

                  dr
          n (r) r --  and  v (r) r,
           l      dg        l

        are defined on radial grids as::

              beta g
          r = ------,  g = 0, 1, ..., N - 1.
              N - g

        """
        assert is_contiguous(nrdr, num.Float)
        assert is_contiguous(vr, num.Float)
        assert nrdr.shape == vr.shape and len(vr.shape) == 1
        return _gpaw.hartree(l, nrdr, beta, N, vr)
else:
    hartree = _gpaw.hartree


def packed_index(i1, i2, ni):
    """Return a packed index"""
    if i1 > i2:
        return (i2 * (2 * ni - 1 - i2) // 2) + i1
    else:
        return (i1 * (2 * ni - 1 - i1) // 2) + i2

def unpack(M):
    """Unpack 1D array to 2D, assuming a packing as in ``pack2``."""
    assert is_contiguous(M, num.Float)
    n = int(sqrt(0.25 + 2.0 * len(M)))
    M2 = num.empty((n, n), num.Float)
    _gpaw.unpack(M, M2)
    return M2

    
def unpack2(M):
    """Unpack 1D array to 2D, assuming a packing as in ``pack``."""
    assert is_contiguous(M)
    n = int(sqrt(0.25 + 2.0 * len(M)))
    M2 = num.zeros((n, n), M.typecode())
    if M.typecode() == num.Complex:
        _gpaw.unpack_complex(M, M2)
    else:
        _gpaw.unpack(M, M2)
    M2 *= 0.5 # divide all by 2
    M2.flat[0::n + 1] *= 2 # rescale diagonal to original size
    return M2

    
def pack(M2, tolerance=1e-10):
    """Pack a 2D array to 1D, adding offdiagonal terms.
    The matrix      / a00 a01 a02 \
                 M2=| a10 a11 a12 |
                    \ a20 a21 a22 /
    is transformed to the vector
       M=(a00,a01+a10*,a02+a20*,a11,a12+a21*,a22)
    """
    n = len(M2)
    M = num.zeros(n * (n + 1) // 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = M2[r, c] + num.conjugate(M2[c, r])
            error = abs(M2[r, c] - num.conjugate(M2[c, r]))
            assert error < tolerance, 'Pack not symmetric by %s'%error+' %'
            p += 1
    assert p == len(M)
    return M


def element_from_packed(M,i,j):
    """Return a specific element from a packed array (by ``pack``)."""
    n=int(sqrt(2*len(M)+.25)) 
    assert i<n
    assert j<n
    if i > j:
        p = (j * (2 * n - 1 - j) // 2) + i
    else: 
        p = (i * (2 * n - 1 - i) // 2) + j
    if i==j:  return M[p]
    elif i>j: return .5*M[p]
    else:     return .5*num.conjugate(M[p])


def pack2(M2, tolerance=1e-10):
    """Pack a 2D array to 1D, averaging offdiagonal terms."""
    n = len(M2)
    M = num.zeros(n * (n + 1) // 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = (M2[r, c] + num.conjugate(M2[c, r])) / 2. # note / 2.
            error = abs(M2[r, c] - num.conjugate(M2[c, r]))
            assert error < tolerance, 'Pack not symmetric by %s'%error+' %'
            p += 1
    assert p == len(M)
    return M


def check_unit_cell(cell):
    """Check that the unit cell (3*3 matrix) is orthorhombic (diagonal)."""
    c = cell.copy()
    # Zero the diagonal:
    c.flat[::4] = 0.0
    if num.sometrue(c.flat):
        raise RuntimeError('Unit cell not orthorhombic')
    

class DownTheDrain:
    """Definition of a stream that throws away all output."""
    
    def write(self, string):
        pass
    
    def flush(self):
        pass


"""
class OutputFilter:
    def __init__(self, out, threshold, level=500):
        self.threshold = threshold
        self.verbosity = verbosity

    def write(self, string):
        if kfdce

"""

def warning(msg):
    r"""Put string in a box.

    >>> print Warning('Watch your step!')
     /\/\/\/\/\/\/\/\/\/\/\
     \                    /
     /  WARNING:          \
     \  Watch your step!  /
     /                    \
     \/\/\/\/\/\/\/\/\/\/\/
    """
    
    n = len(msg)
    if n % 2 == 1:
        n += 1
        msg += ' '
    bar = (n / 2 + 3) * '/\\'
    space = (n / 2 + 2) * '  '
    format = ' %s\n \\%s/\n /  WARNING:%s\\\n \\  %s  /\n /%s\\\n %s/'
    return format % (bar, space, space[10:], msg, space, bar[1:])


def center(atoms):
    """Method for centering atoms in input ListOfAtoms"""
    pos = atoms.GetCartesianPositions()
    cntr = 0.5 * (num.minimum.reduce(pos) + num.maximum.reduce(pos))
    cell = num.diagonal(atoms.GetUnitCell())
    atoms.SetCartesianPositions(pos - cntr + 0.5 * cell)


# Function used by test-suite:
def equal(a, b, e=0):
    assert abs(a - b) <= e, '%g != %g (error: %g > %g)' % (a, b, abs(a - b), e)



def locked(filename):
    try:
        os.open(filename, os.O_EXCL | os.O_RDWR | os.O_CREAT)
    except OSError:
        return True
    os.remove(filename)
    return False

def fix(formula):
    """Convert chemical formula to LaTeX"""
    s = '$'
    j = 0
    for i in range(len(formula)):
        c = formula[i]
        if c.isdigit():
            s += r'\rm{' + formula[j:i] + '}_' + c
            j = i + 1
    if s == '$':
        return s + r'\rm{' + formula + '}$'
    return s + '$'

def fix2(formula):
    """Convert chemical formula to reStructureText"""
    s = ''
    j = 0
    for i in range(len(formula)):
        c = formula[i]
        if c.isdigit():
            s += r'\ `' + c + '`:sub:\ '
        else:
            s += c
    return s
