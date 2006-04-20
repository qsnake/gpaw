from math import pi

import Numeric as num
import RandomArray as ra

from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.domain import Domain
from gridpaw.operators import Gradient, Laplace


domain = Domain((4.0, 4.0, 4.0), (1, 0, 0), angle=pi)
gd = GridDescriptor(domain, (4, 4, 4))
a = gd.new_array()
b = gd.new_array()
a[-1, 1, 1] = 1

gradx = Gradient(gd, c=0)
gradx.apply(a, b)
print b[0, 3, 3]
assert abs(b[0, 3, 3] + 0.5) < 1e-12

lap = Laplace(gd)
lap.apply(a, b)
assert abs(b[0, 3, 3] - 1.0) < 1e-12

domain = Domain((8.0, 4.0, 4.0), (1, 0, 0))
gd = GridDescriptor(domain, (8, 4, 4))
a2 = gd.new_array()
b2 = gd.new_array()
ra.seed(1, 2)
a2[:] = ra.random(a2.shape)
a2[:, 0] = 0.0
a2[:, :, 0] = 0.0
a2[4:, 1:, 1:] = a2[:4, :0:-1, :0:-1]
lap.apply(a2[:4], b)
lap2 = Laplace(gd)
lap2.apply(a2, b2)
print num.sum(b.flat) * 2 - num.sum(b2.flat)
assert abs(num.sum(b.flat) * 2 - num.sum(b2.flat)) < 1.5e-14

