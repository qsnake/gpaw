import Numeric as num
from gridpaw.utilities import equal
from gridpaw.domain import Domain
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.localized_functions import create_localized_functions, \
     LocFuncBroadcaster
from gridpaw.spline import Spline
import gridpaw.utilities.mpi as mpi

s = Spline(0, 1.0, [1.0, 0.5, 0.0])
n = 40
a = 8.0
domain = Domain((a, a, a))
gd = GridDescriptor(domain, (n, n, n))

lfbc = LocFuncBroadcaster(mpi.world)
p = [create_localized_functions([s], gd, (0.5, 0.5, 0.25 + 0.25 * i),
                                onohirose=1, lfbc=lfbc)
     for i in [0, 1, 2]]
lfbc.broadcast()
c = num.identity(1, num.Float)
a = gd.new_array()
for q in p:
    q.add(a, c)
x = num.sum(a.flat)

p = [create_localized_functions([s], gd, (0.75, 0.25, 0.25 * i), onohirose=1)
     for i in [0, 1, 2]]
a[:] = 0.0
for q in p:
    q.add(a, c)
equal(x, num.sum(a.flat), 1e-13)
