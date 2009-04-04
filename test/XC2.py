from math import pi
from gpaw.grid_descriptor import RadialGridDescriptor, GridDescriptor
from gpaw.xc_functional import XC3DGrid, XCRadialGrid
import numpy as np
from gpaw.utilities import equal


for name in ['LDA', 'PBE']:
    r = 0.01 * np.arange(100)
    dr = 0.01 * np.ones(100)
    rgd = RadialGridDescriptor(r, dr)
    xc = XCRadialGrid(name, rgd)
    n = np.exp(-r**2)
    v = np.zeros(100)
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
    gd = GridDescriptor((N, N, N), (a, a, a))
    xc = XC3DGrid(name, gd)
    xc.allocate()
    n = gd.empty()
    n.fill(0.02)
    n += 0.01 * np.sin(np.arange(gd.beg_c[2], gd.end_c[2]) * 2 * pi / N)
    v = 0.0 * n
    E = xc.get_energy_and_potential(n, v)

    n2 = 1.0 * n
    here = (gd.beg_c[0] <= 1 < gd.end_c[0] and
            gd.beg_c[1] <= 2 < gd.end_c[1] and
            gd.beg_c[2] <= 3 < gd.end_c[2])
    if here:
        n2[1, 2, 3] += 0.000001
        x = v[1, 2, 3] * gd.dv
    E2 = xc.get_energy_and_potential(n2, v)
    x2 = (E2 - E) / 0.000001
    if here:
        print x, x2, x - x2
        equal(x, x2, 2e-8)
