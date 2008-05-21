from gpaw.vdw import VanDerWaals
import numpy as npy
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain

g = 4
h = 0.3
a = g * h
for R in [1, 2, 3]:
    dom = Domain([R * a, a, a], pbc=(True, False, False))
    gd = GridDescriptor(dom, (R * g, g, g))
    n = gd.zeros()
    n[:] = 0.001
    vdw = VanDerWaals(n, gd=gd)
    for r in range(0, 10):
        E1, E2 = vdw.get_energy(repeat=(r, 0, 0))
        print R, r, E1 / R, E2 / R
