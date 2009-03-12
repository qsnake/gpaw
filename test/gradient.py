from gpaw.operators import Gradient
import numpy as np
from gpaw.grid_descriptor import GridDescriptor

gd = GridDescriptor((7, 1, 1), (7.0, 1.0, 1.0))
a = gd.zeros()
dadx = gd.zeros()
a[:, 0, 0] = np.arange(7)
gradx = Gradient(gd, c=0)
print a.itemsize, a.dtype, a.shape
print dadx.itemsize, dadx.dtype, dadx.shape
gradx.apply(a, dadx)

#   a = [ 0.  1.  2.  3.  4.  5.  6.]
#
#   da
#   -- = [-2.5  1.   1.   1.   1.   1.  -2.5]
#   dx

if dadx[3, 0, 0] != 1.0 or np.sum(dadx[:, 0, 0]) != 0.0:
    raise AssertionError

gd = GridDescriptor((1, 7, 1), (1.0, 7.0, 1.0), (1, 0, 1))
dady = gd.zeros()
a = gd.zeros()
grady = Gradient(gd, c=1)
a[0, :, 0] = np.arange(6)
grady.apply(a, dady)

#   da
#   -- = [0.5  1.   1.   1.   1.  -2.]
#   dy

if dady[0, 0, 0] != 0.5 or np.sum(dady[0, :, 0]) != 2.5:
    raise AssertionError

