from math import pi
from gpaw.grid_descriptor import RadialGridDescriptor, GridDescriptor
from gpaw.domain import Domain
from gpaw.xc_functional import XCOperator
import Numeric as num
from gpaw.utilities import equal


for name in ['LDA', 'PBE']:
    r = 0.01 * num.arange(100)
    dr = 0.01 * num.ones(100, num.Float)
    rgd = RadialGridDescriptor(r, dr)
    xc = XCOperator(name, rgd)
    n = num.exp(-r**2)
    v = num.zeros(100, num.Float)
    E = xc.get_energy_and_potential(n, v)
    print E
    n2 = 1.0 * n
    i = 23
    n2[i] += 0.000001
    x = v[i] * rgd.dv_g[i]
    E2 = xc.get_energy_and_potential(n2, v)
    x2 = (E2 - E) / 0.000001
    print i, x, x2, x - x2
    equal(x, x2, 2e-8)

    N = 20
    a = 1.0
    gd = GridDescriptor(Domain((a, a, a)), (N, N, N))
    xc = XCOperator(name, gd)
    n = 0.02 * num.ones((N, N, N), num.Float)
    n += 0.01 * num.sin(num.arange(N) * 2 * pi / N)
    v = 0.0 * n
    E = xc.get_energy_and_potential(n, v)

    n2 = 1.0 * n
    i = 17
    n2[i, i, i] += 0.000001
    x = v[i, i, i] * gd.dv
    E2 = xc.get_energy_and_potential(n2, v)
    x2 = (E2 - E) / 0.000001
    print i, x, x2, x - x2
    equal(x, x2, 2e-8)
