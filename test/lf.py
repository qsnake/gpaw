import numpy as np
from gpaw.utilities import equal
from gpaw.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.localized_functions import create_localized_functions, \
     LocFuncBroadcaster
from gpaw.spline import Spline
import gpaw.mpi as mpi

s = Spline(0, 1.0, [1.0, 0.5, 0.0])
n = 40
a = 8.0
domain = Domain((a, a, a))
gd = GridDescriptor(domain, (n, n, n))

lfbc = LocFuncBroadcaster(mpi.world)
p = [create_localized_functions([s], gd, (0.5, 0.5, 0.25 + 0.25 * i),
                                lfbc=lfbc)
     for i in [0, 1, 2]]
lfbc.broadcast()
c = np.ones(1)
a = gd.zeros()
for q in p:
    q.add(a, c)
x = np.sum(a.ravel())

p = [create_localized_functions([s], gd, (0.75, 0.25, 0.25 * i))
     for i in [0, 1, 2]]
a[:] = 0.0
for q in p:
    q.add(a, c)
equal(x, np.sum(a.ravel()), 1e-13)
