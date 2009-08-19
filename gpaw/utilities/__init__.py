# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Utility functions and classes."""

import os
from math import sqrt, exp

import numpy as np

import _gpaw
from gpaw import debug
from numpy import linalg

elementwise_multiply_add = _gpaw.elementwise_multiply_add
utilities_vdot = _gpaw.utilities_vdot
utilities_vdot_self = _gpaw.utilities_vdot_self


erf = np.vectorize(_gpaw.erf, (float,), 'Error function')


# Factorials:
fac = [1, 1, 2, 6, 24, 120, 720, 5040, 40320,
       362880, 3628800, 39916800, 479001600]


def h2gpts(h, cell_cv):
    """Convert grid spacing to number of grid points divisible by 4.

    Note that units of h and cell_cv must match!
    
    h: float
        Desired grid spacing in.
    cell_cv: 3x3 ndarray
        Unit cell.
    """

    L_c = (cell_cv**2).sum(1)**0.5
    d = h * (L_c.prod() / abs(np.linalg.det(cell_cv)))**(1.0 / 3.0)
    return np.maximum(4, (L_c / d / 4 + 0.5).astype(int) * 4)

def gcd(a, b):
    """Return greatest common divisor of a and b, using the
    euclidian algorithm.
    """
    while b != 0:
        a, b = b, a % b
    return a


def contiguous(array, dtype):
    # XXX Use numpy.ascontiguousarray(array, dtype=None) instead!
    """Convert a sequence to a contiguous Numpy array."""
    array = np.asarray(array, dtype)
    if array.flags.contiguous:
        return array
    else:
        return np.array(array)


def is_contiguous(array, dtype=None):
    """Check for contiguity and type."""
    if dtype is None:
        return array.flags.contiguous
    else:
        return array.flags.contiguous and array.dtype == dtype


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
def hartree(l, nrdr, beta, N, vr):
    """Calculates radial Coulomb integral.

    The following integral is calculated::

                                   ^
                          n (r')Y (r')
              ^    / _     l     lm
      v (r)Y (r) = |dr' --------------,
       l    lm     /        _   _
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
    assert is_contiguous(nrdr, float)
    assert is_contiguous(vr, float)
    assert nrdr.shape == vr.shape and len(vr.shape) == 1
    return _gpaw.hartree(l, nrdr, beta, N, vr)


def wignerseitz(index_G, atom_ac, beg_c, end_c):
    """Determine which atom is closest to each grid point.

    For a uniform grid defined by the first and last grid point indices
    beg_c and end_c, determine for each grid point, which atom, specified
    by the atomic coordinates in atom_ac, is the closest. Return result as
    atomic indices on the grid index_G.
    """
    assert is_contiguous(index_G, int)
    assert is_contiguous(atom_ac, float)
    assert atom_ac.shape[1] == len(beg_c) == len(end_c) == 3
    assert index_G.shape == tuple(end_c - beg_c)
    return _gpaw.wigner_seitz_grid(index_G, atom_ac, beg_c, end_c)


def packed_index(i1, i2, ni):
    """Return a packed index"""
    if i1 > i2:
        return (i2 * (2 * ni - 1 - i2) // 2) + i1
    else:
        return (i1 * (2 * ni - 1 - i1) // 2) + i2


def unpacked_indices(p, ni):
    """Return unpacked indices corresponding to upper triangle"""
    assert 0 <= p < ni * (ni + 1) // 2
    i1 = int(ni + .5 - sqrt((ni - .5)**2 - 2 * (p - ni)))
    return i1, p - i1 * (2 * ni - 1 - i1) // 2


packing_conventions = """\n
In the code, the convention is that density matrices are constructed using
pack / unpack2, and anything that should be multiplied onto such, e.g.
corrections to the Hamiltonian, are constructed according to pack2 / unpack.
"""


def unpack(M):
    """Unpack 1D array to 2D, assuming a packing as in ``pack2``."""
    assert is_contiguous(M)
    n = int(sqrt(0.25 + 2.0 * len(M)))
    M2 = np.zeros((n, n), M.dtype.char)
    if M.dtype.char == complex:
        _gpaw.unpack_complex(M, M2)
    else:
        _gpaw.unpack(M, M2)
    return M2

    
def unpack2(M):
    """Unpack 1D array to 2D, assuming a packing as in ``pack``."""
    M2 = unpack(M)
    M2 *= 0.5 # divide all by 2
    M2.flat[0::len(M2) + 1] *= 2 # rescale diagonal to original size
    return M2

    
def pack(M2, tolerance=1e-10):
    """Pack a 2D array to 1D, adding offdiagonal terms.
    
    The matrix::
    
           / a00 a01 a02 \ 
      M2 = | a10 a11 a12 |
           \ a20 a21 a22 /
                
    is transformed to the vector::
    
      M = (a00, a01 + a10*, a02 + a20*, a11, a12 + a21*, a22)
    """
    n = len(M2)
    M = np.zeros(n * (n + 1) // 2, M2.dtype.char)
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = M2[r, c] + np.conjugate(M2[c, r])
            error = abs(M2[r, c] - np.conjugate(M2[c, r]))
            assert error < tolerance, 'Pack not symmetric by %s' % error + ' %'
            p += 1
    assert p == len(M)
    return M


def pack2(M2, tolerance=1e-10):
    """Pack a 2D array to 1D, averaging offdiagonal terms."""
    n = len(M2)
    M = np.zeros(n * (n + 1) // 2, M2.dtype.char)
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = (M2[r, c] + np.conjugate(M2[c, r])) / 2. # note / 2.
            error = abs(M2[r, c] - np.conjugate(M2[c, r]))
            assert error < tolerance, 'Pack not symmetric by %s' % error + ' %'
            p += 1
    assert p == len(M)
    return M


for method in (unpack, unpack2, pack, pack2):
    method.__doc__ += packing_conventions


def element_from_packed(M, i, j):
    """Return a specific element from a packed array (by ``pack``)."""
    n = int(sqrt(2 * len(M) + .25)) 
    assert i < n and j < n
    p = packed_index(i, j, n)
    if i == j:
        return M[p]
    elif i > j:
        return .5 * M[p]
    else:
        return .5 * np.conjugate(M[p])


def check_unit_cell(cell):
    """Check that the unit cell (3*3 matrix) is orthorhombic (diagonal)."""
    c = cell.copy()
    # Zero the diagonal:
    c.flat[::4] = 0.0
    if np.sometrue(c.flat):
        raise RuntimeError('Unit cell not orthorhombic')
    

class _DownTheDrain:
    """Definition of a stream that throws away all output."""
    
    def write(self, string):
        pass
    
    def flush(self):
        pass

    def close(self):
        pass

devnull = _DownTheDrain()

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
    
    lines = ['', 'WARNING:'] + msg.split('\n')
    n = max([len(line) for line in lines])
    n += n % 2
    bar = (n / 2 + 3) * '/\\'
    start, end = ' \\ ', ' / '
    msg = ' %s\n' % bar
    for line in lines + (len(lines) % 2) * ['']:
        msg += '%s %s %s%s\n' % (start, line, (n - len(line)) * ' ', end)
        start, end = end, start
    msg += ' %s/' % bar[1:]
    return msg


def divrl(a_g, l, r_g):
    """Return array divided by r to the l'th power."""
    b_g = a_g.copy()
    if l > 0:
        b_g[1:] /= r_g[1:]**l
        b1, b2 = b_g[1:3]
        r0, r1, r2 = r_g[0:3]
        b_g[0] = b2 + (b1 - b2) * (r0 - r2) / (r1 - r2)
    return b_g


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
    remainder = formula[j:]
    if remainder:
        s += r'\rm{' + remainder + '}'
    return s + '$'


def fix2(formula):
    """Convert chemical formula to reStructuredText"""
    s = ''
    j = 0
    for i in range(len(formula)):
        c = formula[i]
        if c.isdigit():
            s += r'\ `' + c + '`:sub:\ '
        else:
            s += c
    return s


def scalapack():
    return _gpaw.compiled_with_sl()

def load_balance(paw, atoms):
    try:
        paw.initialize(atoms)
    except SystemExit:
        pass
    spos_ac = paw.atoms.get_scaled_positions()
    atoms_r = np.zeros(paw.wfs.world.size)
    rnk_a = paw.gd.get_ranks_from_positions(spos_ac)
    for rnk in rnk_a:
        atoms_r[rnk] += 1
    max_atoms = max(atoms_r)
    min_atoms = min(atoms_r)
    ave_atoms = atoms_r.sum()/paw.wfs.world.size
    stddev_atoms = sqrt((atoms_r**2).sum()/paw.wfs.world.size - ave_atoms**2)
    print "Information about load balancing"
    print "--------------------------------"
    print "Number of atoms:", len(spos_ac)
    print "Number of CPUs:", paw.wfs.world.size
    print "Max. number of atoms/CPU:   ", max_atoms
    print "Min. number of atoms/CPU:   ", min_atoms
    print "Average number of atoms/CPU:", ave_atoms
    print "    standard deviation:     %5.1f" % stddev_atoms

if not debug:
    hartree = _gpaw.hartree
    wignerseitz = _gpaw.wigner_seitz_grid

def interpolate_mlsqr(dg_c, vt_g, order):
    """Interpolate a point using moving least squares algorithm.

    dg_c:    The grid point index (from (0,0,0) to (Bg_g - bg_g)
    vt_g:    The array to be interpolated
    order:   Polynomial order
    """

    # Define the weight function
    lsqr_weight = lambda r2 : exp(-r2)

    # Define the polynomial basis
    if order == 1:
        b = lambda x : np.array([1, x[0], x[1], x[2]])
    elif order == 2:
        b = lambda x :  np.array([1, x[0], x[1], x[2],
                                  x[0]*x[1], x[1]*x[2], x[2]*x[0],
                                  x[0]**2, x[1]**2, x[2]**2])
    else:
        raise NotImplementedError

    def fill_X(x,y,z):
        result = None
        for i,j,k in zip(x.ravel(), y.ravel(), z.ravel()):
            r = b(np.array([i,j,k]))*lsqr_weight(np.sum((dg_c-np.array([i,j,k]))**2))
            if result == None:
                result = r
            else:
                result = np.vstack((result, r))
        return result


    def fill_w(x,y,z):
        result = []
        for i,j,k in zip(x.ravel(), y.ravel(), z.ravel()):
            weight = lsqr_weight(np.sum((dg_c-np.array([i,j,k]))**2))
            result.append(weight * vt_g[i][j][k])
        return np.array(result)
    
    X = np.fromfunction(fill_X, vt_g.shape)
    y = np.fromfunction(fill_w, vt_g.shape)

    X2 = np.dot(X.transpose(), X)
    y2 = np.dot(X.transpose(), y)
    c = linalg.solve(X2, y2)
    a = np.dot(b(dg_c), c)
    return a


def crop_array(v_g, bg_c, Bg_c):
    # XXX TODO: Make more efficient using special features of numpy
    N_c = v_g.shape
    size_c = Bg_c-bg_c+1
    result = np.zeros(size_c)
    for i in range(0, size_c[0]):
        for j in range(0, size_c[1]):
            for k in range(0, size_c[2]):
                result[i][j][k] = v_g[(i + bg_c[0]) % N_c[0]][(j + bg_c[1]) % N_c[1]][(k + bg_c[2]) % N_c[2]]
                
    return result
