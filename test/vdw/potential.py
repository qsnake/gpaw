"""Test correctness of vdW-DF potential."""
import os
from math import pi
from gpaw.grid_descriptor import RadialGridDescriptor, GridDescriptor
from gpaw.domain import Domain
from gpaw.xc_functional import XC3DGrid, XCRadialGrid
import numpy as np
from gpaw.utilities import equal
from gpaw.vdw import FFTVDWFunctional

N = 8
a = 2.0
gd = GridDescriptor((N, N, N), (a, a, a))

# Spin paired:
def paired():
    vdw = FFTVDWFunctional(verbose=0)
    vdw.set_grid_descriptor(gd)
    xc = XC3DGrid(vdw, gd)
    xc.allocate()
    n = 0.3 * np.ones((N, N, N))
    n += 0.01 * np.cos(np.arange(N) * 2 * pi / N)
    v = 0.0 * n
    E = xc.get_energy_and_potential(n, v)
    n2 = 1.0 * n
    i = 1
    n2[i, i, i] += 0.00002
    x = v[i, i, i] * gd.dv
    E2 = xc.get_energy_and_potential(n2, v)
    n2[i, i, i] -= 0.00004
    E2 -= xc.get_energy_and_potential(n2, v)
    x2 = E2 / 0.00004
    print i, x, x2, x - x2, x / x2
    equal(x, x2, 1e-12)

# Spin polarized:
def polarized():
    vdw = FFTVDWFunctional(nspins=2, verbose=0)
    vdw.set_grid_descriptor(gd)
    xc = XC3DGrid(vdw, gd, nspins=2)
    xc.allocate()
    na = 0.04 * np.ones((N, N, N))
    nb = 0.3 * np.ones((N, N, N))
    na += 0.02 * np.sin(np.arange(N) * 2 * pi / N)
    nb += 0.2 * np.cos(np.arange(N) * 2 * pi / N)
    va = 0.0 * na
    vb = 0.0 * nb
    E = xc.get_energy_and_potential(na, va, nb, vb)
    n2a = 1.0 * na
    i = 1
    n2a[i, i, i] += 0.00002
    x = va[i, i, i] * gd.dv
    E2 = xc.get_energy_and_potential(n2a, va, nb, vb)
    n2a[i, i, i] -= 0.00004
    E2 -= xc.get_energy_and_potential(n2a, va, nb, vb)
    x2 = E2 / 0.00004
    print i, x, x2, x - x2, x / x2
    equal(x, x2, 1e-10)

if 'GPAW_VDW' in os.environ:
    paired()
    polarized()
