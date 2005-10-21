# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num
import RandomArray as ra
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.domain import Domain
from gridpaw.transformers import Interpolator, Restrictor
import time


n = 20
gd = GridDescriptor(Domain((1,1,1)), (n,n,n))
ra.seed(7, 8)
a = ra.random((n, n, n))
b = num.zeros((n * 2, n * 2, n * 2), num.Float)
for k in [2, 4, 6]:
    inter = Interpolator(gd, k - 1).apply
    inter(a, b)
    print k, num.sum(a.flat), num.sum(b.flat) / 8
for p in [2, 5]:
    b = num.zeros((n / p, n / p, n / p), num.Float)
    for k in [2, 4, 6]:
        restr = Restrictor(gd, k - 1, p=p).apply
        restr(a, b)
        print k, p, num.sum(a.flat), num.sum(b.flat) * p**3
