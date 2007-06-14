# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Basic Linear Algebra Subroutines (BLAS)
"""

import Numeric as num

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
    assert ((is_contiguous(a, num.Float) and
             is_contiguous(b, num.Float) and
             is_contiguous(c, num.Float) and
             isinstance(alpha, float) and isinstance(beta, float)) or
            (is_contiguous(a, num.Complex) and
             is_contiguous(b, num.Complex) and
             is_contiguous(c, num.Complex)))
    if transa == "n":   
        assert num.rank(b) == 2
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
        assert is_contiguous(x, num.Complex) and is_contiguous(y, num.Complex)
    else:
        assert isinstance(alpha, float)
        assert x.typecode() in [num.Float, num.Complex]
        assert x.typecode() == y.typecode()
        assert x.iscontiguous() and y.iscontiguous()
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
    assert (is_contiguous(a, num.Float) and is_contiguous(c, num.Float) or
            is_contiguous(a, num.Complex) and is_contiguous(c, num.Complex))
    assert num.rank(a) > 1
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
    assert ((is_contiguous(a, num.Float) and is_contiguous(b, num.Float) and
             is_contiguous(c, num.Float) and isinstance(alpha, float)) or
            (is_contiguous(a, num.Complex) and is_contiguous(b,num.Complex) and
             is_contiguous(c, num.Complex)))
    assert num.rank(a) > 1
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
    assert ((is_contiguous(a, num.Float) and is_contiguous(b, num.Float)) or
            (is_contiguous(a, num.Complex) and is_contiguous(b,num.Complex)))
    assert a.shape == b.shape
    return _gpaw.dotc(a, b)
    
if not debug:
    gemm = _gpaw.gemm
    axpy = _gpaw.axpy
    rk = _gpaw.rk
    r2k = _gpaw.r2k
    dotc = _gpaw.dotc;

if __name__ == '__main__':
    a = num.array(((1.0, 3.0, 0.0),
                   (0.0, -0.5, 0.0)))
    b = num.array(((0, 1),(2.0, 0)))
    c = num.zeros((2, 3), num.Float)
    gemm(2.0, a, b, 1.0, c)
    print c
    a = num.array(((1.0, 3.0 + 2j, 0.0),
                   (0.0, -0.5j, 0.0)))
    b = num.array(((0, 1),(2.0+1j, 0)))
    c = num.zeros((2, 3), num.Complex)
    gemm(2.0, a, b, 1.0, c)
    print c
    a = num.array(((1.0, 3.0 +2j, 0.0),
                   (0.0, -0.5j, 0.0)))
    c = num.ones((2, 2), num.Complex)
    rk(2.0, a, 0.0, c)
    print c
    import time, RandomArray
    a = RandomArray.random((30, 23**3))
    b = RandomArray.random((30, 30))
    c = num.zeros((30, 23**3), num.Float)
    t = time.clock()
    for i in range(100):
        gemm(1.0, a, b, 0.0, c)
    print time.clock() - t
