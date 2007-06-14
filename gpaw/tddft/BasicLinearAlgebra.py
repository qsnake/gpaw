import Numeric as num
from gpaw.utilities.blas import axpy
#from gpaw.utilities.blas import dotc

class BLAS:
    
    # Vector dot product, a^H b
    # where ^H is conjugate transpose ^*^T
    # BLAS operation ZDOTC
    def zdotc(self, x, y):
        return num.vdot(x,y)
        # return dotc(x,y)
    
    # Scalar multiplication and vector addition, y = a x + y
    # BLAS operation ZAXPY
    def zaxpy(self, a, x, y):
        axpy(a*(1+0J), x, y)
        #y += a * x
        return y
    
