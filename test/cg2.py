import Numeric as num
from gpaw.utilities.cg import CG
def A(x, b):
    b[:] = num.reshape(num.dot(num.reshape(x, (2, 4)),
                               num.array([[1.0, 0.1, 0.0, 0.0],
                                          [0.1, 1.1, 0.1, 0.1],
                                          [0.0, 0.1, 0.8, 0.1],
                                          [0.0, 0.1, 0.1, 0.9]])),
                       (2, 2, 1, 2))
A.sum = lambda x: x
b = num.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
b.shape = (2, 2, 1, 2)
x = b.copy()
niter, error = CG(A, x, b, verbose=1)
assert niter < 5
def A(x, b):
    b[:] = num.reshape(num.dot(num.reshape(x, (2, 4)),
                               num.array([[1.0, 0.1, 0.0, 0.0],
                                          [0.1, 1.1, 0.1, 0.1],
                                          [0.0, 0.1, 0.8, 0.1],
                                          [0.0, 0.1, 0.1, 0.9]])),
                       (2, 2, 1, 2))
A.sum = lambda x: x

b = num.array([[1.0, 0.1j, 0.01+0.1j, 0.0], [0.0, 1.0, 0.0, 0.0]])
b.shape = (2, 2, 1, 2)
x = b.copy()
niter, error = CG(A, x, b, verbose=1)
assert niter < 5
