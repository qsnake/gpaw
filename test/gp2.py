import Numeric as num
from gridpaw.domain import Domain
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.localized_functions import create_localized_functions
from gridpaw.spline import Spline

s=Spline(0, 1.2, [1, 0.6, 0.1, 0.0])
a = 4.0
domain = Domain((a, a, a))
n = 24
gd = GridDescriptor(domain, (n, n, n))
print gd.get_boxes((0, 0, 0), 1.2, 0)
if 0:
    p = create_localized_functions([s], gd, (0.0, 0.0, 0.0), cut=True)
    a = num.zeros((n, n, n), num.Float)
    p.add(a, num.array([2.0]))
    print a[1,0]
