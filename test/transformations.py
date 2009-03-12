# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.transformers import Transformer
import time


n = 20
gd = GridDescriptor((n,n,n))
np.random.seed(8)
a = np.random.random((n, n, n))

gd2 = gd.refine()
b = gd2.zeros()
for k in [2, 4, 6]:
    inter = Transformer(gd, gd2, k // 2).apply
    inter(a, b)
    print k, np.sum(a.ravel()) - np.sum(b.ravel()) / 8
    assert abs(np.sum(a.ravel()) - np.sum(b.ravel()) / 8) < 2e-11

gd2 = gd.coarsen()
b = gd2.zeros()
for k in [2, 4, 6]:
    restr = Transformer(gd, gd2, k // 2).apply
    restr(a, b)
    print k, np.sum(a.ravel()) - np.sum(b.ravel()) * 8
    assert abs(np.sum(a.ravel()) - np.sum(b.ravel()) * 8) < 3e-12

# complex versions
a = gd.empty(dtype=complex)
a.real = np.random.random((n, n, n))
a.imag = np.random.random((n, n, n))

phase = np.ones((3, 2), complex)

gd2 = gd.refine()
b = gd2.zeros(dtype=complex)
for k in [2, 4, 6]:
    inter = Transformer(gd, gd2, k // 2, complex).apply
    inter(a, b, phase)
    print k, np.sum(a.ravel()) - np.sum(b.ravel()) / 8
    assert abs(np.sum(a.ravel()) - np.sum(b.ravel()) / 8) < 4e-11

gd2 = gd.coarsen()
b = gd2.zeros(dtype=complex)
for k in [2, 4, 6]:
    restr = Transformer(gd, gd2, k // 2, complex).apply
    restr(a, b, phase)
    print k, np.sum(a.ravel()) - np.sum(b.ravel()) * 8
    assert abs(np.sum(a.ravel()) - np.sum(b.ravel()) * 8) < 2e-11

