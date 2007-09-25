# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num
import RandomArray as ra
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.transformers import Transformer
import time


n = 20
gd = GridDescriptor(Domain((1,1,1)), (n,n,n))
ra.seed(7, 8)
a = ra.random((n, n, n))

gd2 = gd.refine()
b = gd2.zeros()
for k in [2, 4, 6]:
    inter = Transformer(gd, gd2, k // 2).apply
    inter(a, b)
    print k, num.sum(a.flat) - num.sum(b.flat) / 8
    assert abs(num.sum(a.flat) - num.sum(b.flat) / 8) < 3e-11

gd2 = gd.coarsen()
b = gd2.zeros()
for k in [2, 4, 6]:
    restr = Transformer(gd, gd2, k // 2).apply
    restr(a, b)
    print k, num.sum(a.flat) - num.sum(b.flat) * 8
    assert abs(num.sum(a.flat) - num.sum(b.flat) * 8) < 5.1e-12
