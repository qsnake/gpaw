import numpy as np
from gpaw.utilities.blas import gemm, axpy, r2k, rk, gemmdot, rotate

a = np.arange(5 * 7, dtype=float).reshape(5, 7)
b = np.arange(7, dtype=float)

ab_np = np.dot(a, b)
a2_np = np.dot(a, a.T)
b2_np = np.dot(b, b)

ab_blas = gemmdot(a, b)
a2_blas = gemmdot(a, a.T.copy())
b2_blas = gemmdot(b[None].copy(), b)

assert np.all(ab_np == ab_blas)
assert np.all(a2_np == a2_blas)
assert np.all(b2_np == b2_blas)
