# This test takes approximately 1.6 seconds
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num
import RandomArray as ra
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.transformers import Transformer
import time


n = 6
gda = GridDescriptor(Domain((1,1,1)), (n,n,n))
gdb = gda.refine()
gdc = gdb.refine()
a = gda.new_array()
b = gdb.new_array()
c = gdc.new_array()

inter = Transformer(gdb, gdc, 2).apply
restr = Transformer(gdb, gda, 2).apply

t = time.clock()
for i in range(8*300):
    inter(b, c)
print time.clock() - t

t = time.clock()
for i in range(8*3000):
    restr(b, a)
print time.clock() - t
