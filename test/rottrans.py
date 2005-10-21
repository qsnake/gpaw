from math import pi

import Numeric as num
import RandomArray as ra

from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.domain import Domain
from gridpaw.operators import Gradient, Laplace


domain = Domain((4.0, 4.0, 4.0), (1, 0, 0), angle=pi)
gd = GridDescriptor(domain, (4, 4, 4))
a = gd.array()
b = gd.array()
a[-1, 1, 1] = 1

gradx = Gradient(gd, axis=0)
gradx.apply(a, b)
assert b[0, 3, 3] == -0.5

lap = Laplace(gd)
lap.apply(a, b)
assert b[0, 3, 3] == 1.0

domain = Domain((8.0, 4.0, 4.0), (1, 0, 0))
gd = GridDescriptor(domain, (8, 4, 4))
a2 = gd.array()
b2 = gd.array()
a2[:] = ra.random(a2.shape)
a2[:, 0] = 0.0
a2[:, :, 0] = 0.0
a2[4:, 1:, 1:] = a2[:4, :0:-1, :0:-1]
lap.apply(a2[:4], b)
lap2 = Laplace(gd)
lap2.apply(a2, b2)
assert num.sum(b.flat) * 2 == num.sum(b2.flat)
