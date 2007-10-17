import Numeric as num
from gpaw.utilities.blas import axpy
from gpaw.utilities.blas import dotc

class DefaultBlas:

    def zeros(self, shape, type):
        return num.zeros(shape,type)

    def array(self, array, type):
        return num.array(array,type)

    # Vector dot product, a^H b
    # where ^H is conjugate transpose ^*^T
    # BLAS operation ZDOTC
    def zdotc(self, x, y):
        return num.vdot(x,y)
    
    # Scalar multiplication and vector addition, y = a x + y
    # BLAS operation ZAXPY
    def zaxpy(self, a, x, y):
        y += a * x
        return y


class GpawBlas:
    
    def __init__(self, gd):
        self.gd = gd

    def zeros(self, shape, type):
        #return num.zeros(shape,type)
        return self.gd.zeros(typecode=type)

    def array(self, array, type):
        #return num.array(array,type)
        tmp = self.gd.empty(typecode=type)
        tmp[:] = array
        return tmp

    # Vector dot product, a^H b
    # where ^H is conjugate transpose ^*^T
    # BLAS operation ZDOTC
    def zdotc(self, x, y):
        # return num.vdot(x,y)
        return self.gd.comm.sum(dotc(x,y))
    
    # Scalar multiplication and vector addition, y = a x + y
    # BLAS operation ZAXPY
    def zaxpy(self, a, x, y):
        #y += a * x
        axpy(a*(1+0J), x, y)
        return y
    
