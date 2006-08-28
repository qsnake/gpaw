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
    xc = XCOperator(name, rgd, nspins=2)
    naa = num.exp(-r**2)
    nb = 0.5 * num.exp(-0.5 * r**2)
    va = num.zeros(100, num.Float)
    vb = num.zeros(100, num.Float)
    E = xc.get_energy_and_potential(naa, va, nb, vb)

    n2 = 1.0 * nb
    i = 23
    n2[i] += 0.000001
    x = vb[i] * rgd.dv_g[i]
    E2 = xc.get_energy_and_potential(naa, va, n2, vb)
    x2 = (E2 - E) / 0.000001
    equal(x, x2, 3e-7)

    N = 20
    a = 1.0
    gd = GridDescriptor(Domain((a, a, a)), (N, N, N))
    xc = XCOperator(name, gd, nspins=2)
    naa = 0.02 * num.ones((N, N, N), num.Float)
    nb = 0.03 * num.ones((N, N, N), num.Float)
    naa += 0.01 * num.cos(num.arange(N) * 2 * pi / N)
    nb += 0.01 * num.sin(num.arange(N) * 2 * pi / N)
    va = 0.0 * naa
    vb = 0.0 * nb
    E = xc.get_energy_and_potential(naa, va, nb, vb)
    n2 = 1.0 * nb
    i = 17
    n2[i, i, i] += 0.0000001
    x = vb[i, i, i] * gd.dv
    E2 = xc.get_energy_and_potential(naa, va, n2, vb)
    x2 = (E2 - E) / 0.0000001
    equal(x, x2, 3e-9)
