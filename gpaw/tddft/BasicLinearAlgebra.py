import Numeric as num
from gpaw.utilities.blas import axpy
#from gpaw.utilities.blas import dotc

class BLAS:
    
    # Vector dot product, a^H b
    # where ^H is conjugate transpose ^*^T
    # BLAS operation ZDOTC
    # !!! FIX ME !!! better way to do this? BLAS!!!
    def zdotc(self, x, y):
        # return dotc(x,y)
        tmp = num.conjugate(x)*y
        while len(tmp.shape) > 1:
            tmp = num.sum(tmp)
        return num.sum(tmp)
    
    # Scalar multiplication and vector addition, y = a x + y
    # BLAS operation ZAXPY
    def zaxpy(self, a, x, y):
        axpy(a*(1+0J), x, y)
        #y += a * x
        return y
    
