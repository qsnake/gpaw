# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gridpaw.utilities import is_contiguous
from gridpaw import debug
import _gridpaw


def gemm(alpha, a, b, beta, c):
    assert ((is_contiguous(a, num.Float) and is_contiguous(b, num.Float) and
             is_contiguous(c, num.Float) and
             type(alpha) is float and type(beta) is float) or
            (is_contiguous(a, num.Complex) and is_contiguous(b, num.Complex) and
             is_contiguous(c, num.Complex)))
##    assert len(a.shape) == 2 and len(b.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[0] == b.shape[1]
    assert c.shape == b.shape[0:1] + a.shape[1:]
    _gridpaw.gemm(alpha, a, b, beta, c)
    
def axpy(alpha, x, y):
    if type(alpha) is float:
        assert is_contiguous(x, num.Float) and is_contiguous(y, num.Float)
    else:
        assert type(alpha) is complex
        assert is_contiguous(x, num.Complex) and is_contiguous(y, num.Complex)
    assert x.shape == y.shape
    _gridpaw.axpy(alpha, x, y)


def rk(alpha, a, beta, c):
    assert (is_contiguous(a, num.Float) and is_contiguous(c, num.Float) or
            is_contiguous(a, num.Complex) and is_contiguous(c, num.Complex))
    assert len(a.shape) > 1
    assert c.shape == (a.shape[0], a.shape[0])
    _gridpaw.rk(alpha, a, beta, c)

    
def r2k(alpha, a, b, beta, c):
    assert ((is_contiguous(a, num.Float) and is_contiguous(b, num.Float) and
             is_contiguous(c, num.Float) and type(alpha) is float) or
            (is_contiguous(a, num.Complex) and is_contiguous(b, num.Complex) and
             is_contiguous(c, num.Complex)))
    assert len(a.shape) > 1
    assert a.shape == b.shape
    assert c.shape == (a.shape[0], a.shape[0])
    _gridpaw.r2k(alpha, a, b, beta, c)


if not debug:
    gemm = _gridpaw.gemm
    axpy = _gridpaw.axpy
    rk = _gridpaw.rk
    r2k = _gridpaw.r2k


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


                   
