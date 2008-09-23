# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as npy
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.transformers import Transformer
import time


n = 20
gd = GridDescriptor(Domain((1,1,1)), (n,n,n))
npy.random.seed(8)
a = npy.random.random((n, n, n))

gd2 = gd.refine()
b = gd2.zeros()
for k in [2, 4, 6]:
    inter = Transformer(gd, gd2, k // 2).apply
    inter(a, b)
    print k, npy.sum(a.ravel()) - npy.sum(b.ravel()) / 8
    assert abs(npy.sum(a.ravel()) - npy.sum(b.ravel()) / 8) < 2e-11

gd2 = gd.coarsen()
b = gd2.zeros()
for k in [2, 4, 6]:
    restr = Transformer(gd, gd2, k // 2).apply
    restr(a, b)
    print k, npy.sum(a.ravel()) - npy.sum(b.ravel()) * 8
    assert abs(npy.sum(a.ravel()) - npy.sum(b.ravel()) * 8) < 3e-12

# complex versions
a = gd.empty(dtype=complex)
a.real = npy.random.random((n, n, n))
a.imag = npy.random.random((n, n, n))

phase = npy.ones((3, 2), complex)

gd2 = gd.refine()
b = gd2.zeros(dtype=complex)
for k in [2, 4, 6]:
    inter = Transformer(gd, gd2, k // 2, complex).apply
    inter(a, b, phase)
    print k, npy.sum(a.ravel()) - npy.sum(b.ravel()) / 8
    assert abs(npy.sum(a.ravel()) - npy.sum(b.ravel()) / 8) < 4e-11

gd2 = gd.coarsen()
b = gd2.zeros(dtype=complex)
for k in [2, 4, 6]:
    restr = Transformer(gd, gd2, k // 2, complex).apply
    restr(a, b, phase)
    print k, npy.sum(a.ravel()) - npy.sum(b.ravel()) * 8
    assert abs(npy.sum(a.ravel()) - npy.sum(b.ravel()) * 8) < 2e-11

