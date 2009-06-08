# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Basic Linear Algebra Subroutines (BLAS)

See also:
http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
and
http://www.netlib.org/lapack/lug/node145.html
"""

import numpy as np

from gpaw.utilities import is_contiguous
from gpaw import debug
import _gpaw


def gemm(alpha, a, b, beta, c, transa='n'):
    """General Matrix Multiply.

    Performs the operation::
    
      c <- alpha * b.a + beta * c

    If transa is "n", ``b.a`` denotes the matrix multiplication defined by::
    
                      _
                     \  
      (b.a)        =  ) b  * a
           ijkl...   /_  ip   pjkl...
                      p
    
    If transa is "t" or "c", ``b.a`` denotes the matrix multiplication
    defined by::
    
                      _
                     \  
      (b.a)        =  ) b    *    a
           ij        /_  iklm...   jklm...
                     klm... 

    where in case of "c" also complex conjugate of a is taken.
    """
    assert np.isfinite(c).all()
    
    assert (a.dtype == float and b.dtype == float and c.dtype == float and
            isinstance(alpha, float) and isinstance(beta, float) or
            a.dtype == complex and b.dtype == complex and c.dtype == complex)
    assert a.flags.contiguous
    if transa == 'n':
        assert c.flags.contiguous or c.ndim == 2 and c.strides[1] == c.itemsize
        assert b.ndim == 2
        assert b.strides[1] == b.itemsize
        assert a.shape[0] == b.shape[1]
        assert c.shape == b.shape[0:1] + a.shape[1:]
    else:
        assert b.flags.contiguous
        assert c.strides[1] == c.itemsize
        assert a.shape[1:] == b.shape[1:]
        assert c.shape == (b.shape[0], a.shape[0])
    _gpaw.gemm(alpha, a, b, beta, c, transa)

    
def axpy(alpha, x, y):
    """alpha x plus y.

    Performs the operation::

      y <- alpha * x + y
      
    """
    if isinstance(alpha, complex):
        assert is_contiguous(x, complex) and is_contiguous(y, complex)
    else:
        assert isinstance(alpha, float)
        assert x.dtype in [float, complex]
        assert x.dtype == y.dtype
        assert x.flags.contiguous and y.flags.contiguous
    assert x.shape == y.shape
    _gpaw.axpy(alpha, x, y)


def rk(alpha, a, beta, c):
    """Rank-k update of a matrix.

    Performs the operation::
    
                        dag
      c <- alpha * a . a    + beta * c

    where ``a.b`` denotes the matrix multiplication defined by::

                 _
                \  
      (a.b)   =  ) a         * b
           ij   /_  ipklm...     pjklm...
               pklm...

    ``dag`` denotes the hermitian conjugate (complex conjugation plus a
    swap of axis 0 and 1).
    
    Only the lower triangle of ``c`` will contain sensible numbers.
    """
    assert np.isfinite(c).all()

    assert (a.dtype == float and c.dtype == float or
            a.dtype == complex and c.dtype == complex)
    assert a.flags.contiguous
    assert a.ndim > 1
    assert c.shape == (a.shape[0], a.shape[0])
    assert c.strides[1] == c.itemsize
    _gpaw.rk(alpha, a, beta, c)

    
def r2k(alpha, a, b, beta, c):
    """Rank-2k update of a matrix.

    Performs the operation::

                        dag        cc       dag
      c <- alpha * a . b    + alpha  * b . a    + beta * c

    where ``a.b`` denotes the matrix multiplication defined by::

                 _
                \ 
      (a.b)   =  ) a         * b
           ij   /_  ipklm...     pjklm...
               pklm...

    ``cc`` denotes complex conjugation.
    
    ``dag`` denotes the hermitian conjugate (complex conjugation plus a
    swap of axis 0 and 1).

    Only the lower triangle of ``c`` will contain sensible numbers.
    """
    assert np.isfinite(c).all()
        
    assert (a.dtype == float and b.dtype == float and c.dtype == float or
            a.dtype == complex and b.dtype == complex and c.dtype == complex)
    assert a.flags.contiguous and b.flags.contiguous
    assert np.rank(a) > 1
    assert a.shape == b.shape
    assert c.shape == (a.shape[0], a.shape[0])
    assert c.strides[1] == c.itemsize
    _gpaw.r2k(alpha, a, b, beta, c)

def dotc(a, b):
    """Dot product, conjugating the first vector with complex arguments.

    Returns the value of the operation::

        _
       \   cc   
        ) a       * b
       /_  ijk...    ijk...       
       ijk...

    ``cc`` denotes complex conjugation.
    """
    assert ((is_contiguous(a, float) and is_contiguous(b, float)) or
            (is_contiguous(a, complex) and is_contiguous(b,complex)))
    assert a.shape == b.shape
    return _gpaw.dotc(a, b)
    

def dotu(a, b):
    """Dot product, NOT conjugating the first vector with complex arguments.

    Returns the value of the operation::

        _
       \ 
        ) a       * b
       /_  ijk...    ijk...
       ijk...


    """
    assert ((is_contiguous(a, float) and is_contiguous(b, float)) or
            (is_contiguous(a, complex) and is_contiguous(b,complex)))
    assert a.shape == b.shape
    return _gpaw.dotu(a, b)
    

def _gemmdot(a, b, alpha=1.0, trans='n'):
    """Matrix multiplication using gemm.

    For the 2D matrices a, b, return::

      c = alpha * a . b

    where '.' denotes matrix multiplication.
    If trans='t'; b is replaced by its transpose.
    If trans='c'; b is replaced by its hermitian conjugate.
    """
    isvector = b.ndim == 1
    if trans == 'n':
        if isvector:
            b = b.reshape(-1, 1)
        shape = a.shape[0], b.shape[1]
    else: # 't' or 'c'
        if isvector:
            b = b.reshape(1, -1)
        shape = a.shape[0], b.shape[0]

    # (ATLAS can't handle uninitialized output array)
    c = np.zeros(shape, a.dtype)
    gemm(alpha, b, a, 0.0, c, trans)
    if isvector:
        c = c.reshape(shape[0])
    return c


def _gemmdot2(a, b, alpha=1.0, beta=1.0, out=None):
    """Matrix multiplication using gemm.

    return reference to out, where::

      out <- beta * out + alpha * a . b

    and ``a.b`` denotes the matrix multiplication defined by::

                           _
                          \  
      (a.b)             =  ) a        * b
           ijk...pqr...   /_  ijk...l     lpqr...
                           l

    I.e. the product-sum if over the last dimension of a, and first dimension
    of b.
    
    If out is None, a suitable array will be created.
    """
    aisvector = a.ndim == 1
    bisvector = b.ndim == 1
    if aisvector and bisvector:
        assert out is None
        return beta * _dotu(a, b)
    elif aisvector:
        a = a.reshape(1, -1)
    elif bisvector:
        b = b.reshape(-1, 1)
    outshape = a.shape[:-1] + b.shape[1:]
    if out is None:
        out = np.zeros(outshape, a.dtype)
    else:
        out = out.reshape(outshape)
    gemm(alpha, b, a, beta, out, 'n')
    if aisvector:
        outshape = outshape[1:]
    elif bisvector:
        outshape = outshape[:-1]
    return out.reshape(outshape)


def _rotate(in_jj, U_ij, a=1., b=0., out_ii=None, work_ij=None):
    """Perform matrix rotation using gemm

    For the 2D input matrices in, U, do the rotation::

      out <- a * U . in . U^d + b * out

    where '.' denotes matrix multiplication and '^d' the hermitian conjugate.

    work_ij is a temporary work array for storing the intermediate product.
    out_ii, and work_ij are created if not given.

    The method returns a reference to out.
    """
    if work_ij is None:
        work_ij = np.empty_like(U_ij)
    if out_ii is None:
        out_ii = np.empty(U_ij.shape[:1] * 2, U_ij.dtype)
    if in_jj.dtype == float:
        trans = 't'
    else:
        trans = 'c'
    gemm(1., in_jj, U_ij, 0., work_ij, 'n')
    gemm(a, U_ij, work_ij, b, out_ii, trans)
    return out_ii


if not debug:
    gemm = _gpaw.gemm
    axpy = _gpaw.axpy
    rk = _gpaw.rk
    r2k = _gpaw.r2k
    dotc = _gpaw.dotc
    dotu = _gpaw.dotu
    gemmdot = _gemmdot
    gemmdot2 = _gemmdot2
    rotate = _rotate
else:
    def gemmdot(a, b, alpha=1., trans='n'):
        assert a.flags.contiguous
        assert b.flags.contiguous
        assert a.dtype == b.dtype
        assert a.ndim == 2
        assert 1 <= b.ndim <= 2
        if trans == 'n':
            assert a.shape[1] == b.shape[0]
        elif trans == 't':
            assert a.shape[1] == b.shape[-1]
        else: # 'c'
            assert a.shape[1] == b.shape[-1]
        return _gemmdot(a, b, alpha, trans)

    def gemmdot2(a, b, alpha=1.0, beta=1.0, out=None):
        assert a.flags.contiguous
        assert b.flags.contiguous
        assert a.dtype == b.dtype
        assert a.shape[-1] == b.shape[0]
        if out is not None:
            assert out.flags.contiguous
            assert a.dtype == out.dtype
            assert a.ndim > 1 or b.ndim > 1
            if a.ndim == 1:
                assert out.shape == b.shape[1:]
            elif b.ndim == 1:
                assert out.shape == a.shape[:-1]
            else:
                assert out.shape == a.shape[:-1] + b.shape[1:]
        return _gemmdot2(a, b, alpha, beta, out)

    def rotate(in_jj, U_ij, a=1., b=0., out_ii=None, work_ij=None):
        assert in_jj.dtype == U_ij.dtype
        assert in_jj.flags.contiguous
        assert U_ij.flags.contiguous
        assert in_jj.shape == U_ij.shape[:1] * 2
        if out_ii is not None:
            assert out_ii.dtype == in_jj.dtype
            assert out_ii.flags.contiguous
            assert out_jj.shape == U_ij.shape[1:] * 2
        if work_ij is not None:
            assert work_ij.dtype == in_jj.dtype
            assert work_ij.flags.contiguous
            assert work_ij.shape == U_ij.shape
        return _rotate(in_jj, U_ij, a, b, out_ii, work_ij)
        
