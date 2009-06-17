import numpy as np
a = np.eye(3, dtype=complex)
a[1:, 0] = 0.01j
w0 = [0.98585786, 1.0, 1.01414214]
from numpy.linalg import eigh
w = eigh(a)[0]
print w
assert abs(w - w0).max() < 1e-8
from gpaw.utilities.lapack import diagonalize
diagonalize(a, w)
print w
assert abs(w - w0).max() < 1e-8
