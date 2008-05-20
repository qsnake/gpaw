# Written by Lauri Lehtovaara 2008

from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc

class MultiBlas:
    def __init__(self, gd):
        self.gd = gd

    # Multivector ZAXPY: a x + y => y
    def multi_zaxpy(self, a,x,y, nvec):
        if isinstance(a, (float, complex)):
            for i in range(nvec):
                axpy(a*(1+0J), x[i], y[i])
        else:
            for i in range(nvec):
                axpy(a[i]*(1.0+0.0J), x[i], y[i])

    # Multivector dot product, a^H b, where ^H is transpose
    def multi_zdotc(self, s, x,y, nvec):
        for i in range(nvec):
            s[i] = dotc(x[i],y[i])
        self.gd.comm.sum(s)
        return s
            
    # Multiscale: a x => x
    def multi_scale(self, a,x, nvec):
        if isinstance(a, (float, complex)):
            x *= a
        else:
            for i in range(nvec):
                x[i] *= a[i]
