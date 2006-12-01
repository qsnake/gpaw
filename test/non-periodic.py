from gpaw.transformers import Interpolator, Restrictor
import Numeric as num
import RandomArray as ra
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain

p = 0
domain = Domain((8.0, 8.0, 8.0), periodic=(p, p, p))
n = 20
gd1 = GridDescriptor(domain, (n, n, n))
a1 = gd1.new_array()
ra.seed(1, 2)
a1[:] = ra.random(a1.shape)
gd2 = gd1.refine()
a2 = gd2.new_array()
i = Interpolator(gd1, 1).apply
i(a1, a2)
assert abs(num.sum(a1.flat) - num.sum(a2.flat) / 8) < 1e-10
r = Restrictor(gd2, 1).apply
a2[0] = 0.0
a2[:, 0] = 0.0
a2[:, :, 0] = 0.0
a2[-1] = 0.0
a2[:, -1] = 0.0
a2[:, :, -1] = 0.0
r(a2, a1)
assert abs(num.sum(a1.flat) - num.sum(a2.flat) / 8) < 1e-10
