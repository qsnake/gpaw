from math import pi
from gpaw.grid_descriptor import RadialGridDescriptor, GridDescriptor
from gpaw.xc_functional import XC3DGrid, XCRadialGrid
import numpy as np
from gpaw.utilities import equal


for name in ['LDA', 'PBE']:
    r = 0.01 * np.arange(100)
    dr = 0.01 * np.ones(100)
    rgd = RadialGridDescriptor(r, dr)
    xc = XCRadialGrid(name, rgd, nspins=2)
    na = np.exp(-r**2)
    nb = 0.5 * np.exp(-0.5 * r**2)
    va = np.zeros(100)
    vb = np.zeros(100)
    E = xc.get_energy_and_potential(na, va, nb, vb)

    n2 = 1.0 * nb
    i = 23
    n2[i] += 0.000001
    x = vb[i] * rgd.dv_g[i]
    E2 = xc.get_energy_and_potential(na, va, n2, vb)
    x2 = (E2 - E) / 0.000001
    equal(x, x2, 3e-7)

    N = 20
    a = 1.0
    gd = GridDescriptor((N, N, N), (a, a, a))
    xc = XC3DGrid(name, gd, nspins=2)
    xc.allocate()
    na = gd.empty()
    na.fill(0.02)
    nb = 1.5 * na
    na += 0.01 * np.cos(np.arange(gd.beg_c[2], gd.end_c[2]) * 2 * pi / N)
    nb += 0.01 * np.sin(np.arange(gd.beg_c[2], gd.end_c[2]) * 2 * pi / N)
    va = 0.0 * na
    vb = 0.0 * nb
    E = xc.get_energy_and_potential(na, va, nb, vb)
    n2 = 1.0 * nb
    here = (gd.beg_c[0] <= 1 < gd.end_c[0] and
            gd.beg_c[1] <= 2 < gd.end_c[1] and
            gd.beg_c[2] <= 3 < gd.end_c[2])
    if here:
        n2[1, 2, 3] += 0.0000001
        x = vb[1, 2, 3] * gd.dv
    E2 = xc.get_energy_and_potential(na, va, n2, vb)
    x2 = (E2 - E) / 0.0000001
    if here:
        equal(x, x2, 3e-9)
