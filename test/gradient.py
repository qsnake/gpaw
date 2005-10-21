from gridpaw.operators import Gradient
import Numeric as num
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.domain import Domain


domain = Domain((7.0, 1.0, 1.0))
gd = GridDescriptor(domain, (7, 1, 1))
a = gd.array()
dadx = gd.array()
a[:, 0, 0] = num.arange(7)
gradx = Gradient(gd, axis=0)
gradx.apply(a, dadx)

#   a = [ 0.  1.  2.  3.  4.  5.  6.]
#
#   da
#   -- = [-2.5  1.   1.   1.   1.   1.  -2.5]
#   dx

if dadx[3, 0, 0] != 1.0 or num.sum(dadx[:, 0, 0]) != 0.0:
    raise AssertionError

domain = Domain((1.0, 7.0, 1.0), periodic=(1, 0, 1))
gd = GridDescriptor(domain, (1, 7, 1))
dady = gd.array()
a = gd.array()
grady = Gradient(gd, axis=1)
a[0, :, 0] = num.arange(7)
grady.apply(a, dady)

#   da
#   -- = [0.5  1.   1.   1.   1.   1.  -2.5]
#   dy

if dady[0, 0, 0] != 0.0 or num.sum(dady[0, :, 0]) != 2.5:
    raise AssertionError

