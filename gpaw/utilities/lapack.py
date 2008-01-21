# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Linear Algebra PACKage (LAPACK)
"""

import numpy as npy

from gpaw import debug
import _gpaw

from gpaw.utilities.blas import gemm

def diagonalize(a, w, b=None):
    """Diagonalize a symmetric/hermitian matrix.

    Uses dsyevd/zheevd to diagonalize symmetric/hermitian matrix
    `a`. The eigenvectors are returned in the rows of `a`, and the
    eigenvalues in `w` in ascending order. Only the lower triangle of
    `a` is considered.

    If a symmetric/hermitian positive definite matrix b is given, then
    dsygvd/zhegvd is used to solve a generalized eigenvalue
    problem: a*v=b*v*w."""

    assert a.flags.contiguous
    assert w.flags.contiguous
    assert a.dtype in [float, complex]
    assert w.dtype == float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)
    if b is not None:
        assert b.flags.contiguous
        assert b.dtype == a.dtype
        assert b.shape == a.shape
        info = _gpaw.diagonalize(a, w, b)
    else:
        info = _gpaw.diagonalize(a, w)
    return info

def inverse_cholesky(a):
    """Calculate the inverse of the Cholesky decomposition of
    a symmetric/hermitian positive definete matrix `a`.

    Uses dpotrf/zpotrf to calculate the decomposition and then
    dtrtri/ztrtri for the inversion"""

    assert a.flags.contiguous
    assert a.dtype in [float, complex]
    n = len(a)
    assert a.shape == (n, n)
    info = _gpaw.inverse_cholesky(a)
    return info


def right_eigenvectors(a, w, v):
    """Get right eigenvectors and eigenvalues from a square matrix
    using LAPACK dgeev.

    The right eigenvector corresponding to eigenvalue w[i] is v[i]."""

    assert a.flags.contiguous
    assert w.flags.contiguous
    assert v.flags.contiguous
    assert a.dtype == float
    assert w.dtype == float
    assert v.dtype == float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)
    assert w.shape == (n,n)
    return _gpaw.right_eigenvectors(a, w, v)

def pm(M):
    """print a matrix or a vector in mathematica style"""
    string=''
    s = M.shape
    if len(s) > 1:
        (n,m)=s
        string += '{'
        for i in range(n):
            string += '{'
            for j in range(m):
                string += str(M[i,j])
                if j == m-1:
                    string += '}'
                else:
                    string += ','
            if i == n-1:
                string += '}'
            else:
                string += ','
    else:
        n=s[0]
        string += '{'
        for i in range(n):
            string += str(M[i])
            if i == n-1:
                string += '}'
            else:
                string += ','
    return string

def sqrt_matrix(a, preserve=False):
    """Get the sqrt of a symmetric matrix a (diagonalize is used).
    The matrix is kept if preserve=True, a=sqrt(a) otherwise."""
    n = len(a)
    if debug:
         assert a.flags.contiguous
         assert a.dtype == float
         assert a.shape == (n, n)
    if preserve:
        b = a.copy()
    else:
        b = a

    # diagonalize to get the form b = Z * D * Z^T
    # where D is diagonal
    D = npy.empty((n,))
    diagonalize(b, D)
    ZT = b.copy()
    Z = npy.transpose(b)

    # c = Z * sqrt(D)
    c = Z * npy.sqrt(D)

    # sqrt(b) = c * Z^T
    gemm(1., ZT, c, 0., b)
    
    return b

if not debug:
    diagonalize = _gpaw.diagonalize
    right_eigenvectors = _gpaw.right_eigenvectors
    inverse_cholesky = _gpaw.inverse_cholesky
