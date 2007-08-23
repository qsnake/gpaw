# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Linear Algebra PACKage (LAPACK)
"""

import Numeric as num

from gpaw import debug
import _gpaw


def diagonalize(a, w, b=None):
    """Diagonalize a symmetric/hermitian matrix.

    Uses dsyevd/zheevd to diagonalize symmetric/hermitian matrix
    `a`. The eigenvectors are returned in the rows of `a`, and the
    eigenvalues in `w` in ascending order. Only the lower triangle of
    `a` is considered.

    If a symmetric/hermitian positive definite matrix b is given, then
    dsygvd/zhegvd is used to solve a generalized eigenvalue
    problem: a*v=b*v*w."""

    assert a.iscontiguous()
    assert w.iscontiguous()
    assert a.typecode() in [num.Float, num.Complex]
    assert w.typecode() == num.Float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)
    if b:
        assert b.iscontiguous()
        assert b.typecode() == a.typecode()
        assert b.shape == a.shape
        info = _gpaw.diagonalize(a, w, b)
    else:
        info = _gpaw.diagonalize(a, w)
    return info

def right_eigenvectors(a, w, v):
    """Get right eigenvectors and eigenvalues from a square matrix
    using LAPACK dgeev.

    The right eigenvector corresponding to eigenvalue w[i] is v[i]."""

    assert a.iscontiguous()
    assert w.iscontiguous()
    assert v.iscontiguous()
    assert a.typecode() == num.Float
    assert w.typecode() == num.Float
    assert v.typecode() == num.Float
    n = len(a)
    assert a.shape == (n, n)
    assert w.shape == (n,)
    assert w.shape == (n,n)
    return _gpaw.right_eigenvectors(a, w, v)

if not debug:
    diagonalize = _gpaw.diagonalize
    right_eigenvectors = _gpaw.right_eigenvectors
    
