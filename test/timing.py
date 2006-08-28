# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num
import RandomArray as ra
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.transformers import Interpolator, Restrictor
import time


n = 12
gd = GridDescriptor(Domain((1,1,1)), (n,n,n))
a = ra.random((n / 2, n / 2, n / 2))
b = ra.random((n, n, n))
c = ra.random((n * 2, n * 2, n * 2))

inter = Interpolator(gd, 3).apply
restr = Restrictor(gd, 3).apply

t = time.clock()
for i in range(8*300):
    inter(b, c)
print time.clock() - t

t = time.clock()
for i in range(8*3000):
    restr(b, a)
print time.clock() - t
