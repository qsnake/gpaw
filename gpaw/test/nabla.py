import numpy as np
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.grid_descriptor import GridDescriptor, RadialGridDescriptor
from gpaw.spline import Spline
from gpaw.setup import Setup

rc = 2.0
a = 2.5 * rc
n = 64
lmax = 2
b = 8.0
m = (lmax + 1)**2
gd = GridDescriptor([n, n, n], [a, a, a])
r = np.linspace(0, rc, 200)
g = np.exp(-(r / rc * b)**2)
splines = [Spline(l=l, rmax=rc, f_g=g) for l in range(lmax + 1)]
c = LFC(gd, [splines])
c.set_positions([(0, 0, 0)])
psi = gd.zeros(m)
c.add(psi, {0: np.identity(m)})
d1 = c.dict(m, derivative=True)
c.derivative(psi, d1)
d1 = d1[0]
class TestSetup(Setup):
    l_j = range(lmax + 1)
    nj = lmax + 1
    ni = m
    def __init__(self):
        pass
rgd = RadialGridDescriptor(r, np.ones_like(r) * r[1])
g = [np.exp(-(r / rc * b)**2) * r**l for l in range(lmax + 1)]
d2 = TestSetup().get_derivative_integrals(rgd, g, np.zeros_like(g))
assert abs(d1 - d2).max() < 2e-6



