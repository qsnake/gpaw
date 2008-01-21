# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Basic Linear Algebra Subroutines (BLAS)
"""

import numpy as npy

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
    
    If transa is "t" or "c", ``b.a`` denotes the matrix multiplication defined by::
    
                      _
                     \  
      (b.a)        =  ) b    *    a
           ij        /_  iklm...   jklm...
                     klm... 

     where in case of "c" also complex conjugate of a is taken.
    """
    assert ((is_contiguous(a, float) and
             is_contiguous(b, float) and
             is_contiguous(c, float) and
             isinstance(alpha, float) and isinstance(beta, float)) or
            (is_contiguous(a, complex) and
             is_contiguous(b, complex) and
             is_contiguous(c, complex)))
    if transa == "n":   
        assert npy.rank(b) == 2
        assert a.shape[0] == b.shape[1]
        assert c.shape == b.shape[0:1] + a.shape[1:]
    else:
        assert a.shape[1:] == b.shape[1:]
        assert c.shape == b.shape[0:1] + a.shape[0:1]
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
    assert (is_contiguous(a, float) and is_contiguous(c, float) or
            is_contiguous(a, complex) and is_contiguous(c, complex))
    assert npy.rank(a) > 1
    assert c.shape == (a.shape[0], a.shape[0])
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
    assert ((is_contiguous(a, float) and is_contiguous(b, float) and
             is_contiguous(c, float) and isinstance(alpha, float)) or
            (is_contiguous(a, complex) and is_contiguous(b,complex) and
             is_contiguous(c, complex)))
    assert npy.rank(a) > 1
    assert a.shape == b.shape
    assert c.shape == (a.shape[0], a.shape[0])
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
    
if not debug:
    gemm = _gpaw.gemm
    axpy = _gpaw.axpy
    rk = _gpaw.rk
    r2k = _gpaw.r2k
    dotc = _gpaw.dotc;
