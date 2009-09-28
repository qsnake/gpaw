from gpaw.operators import Gradient
import numpy as np
from gpaw.grid_descriptor import GridDescriptor

gd = GridDescriptor((8, 1, 1), (8.0, 1.0, 1.0))
a = gd.zeros()
dadx = gd.zeros()
a[:, 0, 0] = np.arange(gd.beg_c[0], gd.end_c[0])
gradx = Gradient(gd, c=0)
print a.itemsize, a.dtype, a.shape
print dadx.itemsize, dadx.dtype, dadx.shape
gradx.apply(a, dadx)

#   a = [ 0.  1.  2.  3.  4.  5.  6.  7.]
#
#   da
#   -- = [-2.5  1.   1.   1.   1.   1.  1.  -2.5]
#   dx

dadx = gd.collect(dadx, broadcast=True)
assert dadx[3, 0, 0] == 1.0 and np.sum(dadx[:, 0, 0]) == 0.0

gd = GridDescriptor((1, 8, 1), (1.0, 8.0, 1.0), (1, 0, 1))
dady = gd.zeros()
a = gd.zeros()
grady = Gradient(gd, c=1)
a[0, :, 0] = np.arange(gd.beg_c[1], gd.end_c[1]) - 1
grady.apply(a, dady)

#   da
#   -- = [0.5  1.   1.   1.   1.   1.  -2.5]
#   dy

dady = gd.collect(dady, broadcast=True)
assert dady[0, 0, 0] == 0.5 and np.sum(dady[0, :, 0]) == 3.0

