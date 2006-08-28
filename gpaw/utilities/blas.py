# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gpaw.utilities import is_contiguous
from gpaw import debug
import _gpaw


def gemm(alpha, a, b, beta, c):
    assert ((is_contiguous(a, num.Float) and
             is_contiguous(b, num.Float) and
             is_contiguous(c, num.Float) and
             isinstance(alpha, float) and isinstance(beta, float)) or
            (is_contiguous(a, num.Complex) and
             is_contiguous(b, num.Complex) and
             is_contiguous(c, num.Complex)))
    assert num.rank(b) == 2
    assert a.shape[0] == b.shape[1]
    assert c.shape == b.shape[0:1] + a.shape[1:]
    _gpaw.gemm(alpha, a, b, beta, c)

    
def axpy(alpha, x, y):
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
    assert (is_contiguous(a, num.Float) and is_contiguous(c, num.Float) or
            is_contiguous(a, num.Complex) and is_contiguous(c, num.Complex))
    assert num.rank(a) > 1
    assert c.shape == (a.shape[0], a.shape[0])
    _gpaw.rk(alpha, a, beta, c)

    
def r2k(alpha, a, b, beta, c):
    assert ((is_contiguous(a, num.Float) and is_contiguous(b, num.Float) and
             is_contiguous(c, num.Float) and isinstance(alpha, float)) or
            (is_contiguous(a, num.Complex) and is_contiguous(b, num.Complex) and
             is_contiguous(c, num.Complex)))
    assert num.rank(a) > 1
    assert a.shape == b.shape
    assert c.shape == (a.shape[0], a.shape[0])
    _gpaw.r2k(alpha, a, b, beta, c)


if not debug:
    gemm = _gpaw.gemm
    axpy = _gpaw.axpy
    rk = _gpaw.rk
    r2k = _gpaw.r2k


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


                   
