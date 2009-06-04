import numpy as np
from gpaw.utilities.blas import gemm, axpy, r2k, rk, gemmdot, rotate

a = np.arange(5 * 7, dtype=float).reshape(5, 7)
b = np.arange(7, dtype=float)

ab_np = np.dot(a, b)
a2_np = np.dot(a, a.T)
b2_np = np.dot(b, b)

ab_blas = gemmdot(a, b)
a2_blas = gemmdot(a, a, trans='t')
b2_blas = gemmdot(b[None].copy(), b)

assert np.all(ab_np == ab_blas)
assert np.all(a2_np == a2_blas)
assert np.all(b2_np == b2_blas)

a = a * (2 + 1.j)
b = b * (3 - 2.j)

ab_np = np.dot(a, b)
a2_np = np.dot(a, a.T.conj())
b2_np = np.dot(b, b.conj())

ab_blas = gemmdot(a, b)
a2_blas = gemmdot(a, a, trans='c')
b2_blas = gemmdot(b[None].copy(), b, trans='c')

assert np.all(ab_np == ab_blas)
assert np.all(a2_np == a2_blas)
assert np.all(b2_np == b2_blas)
