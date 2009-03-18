from math import sqrt
import numpy as np
from gpaw.spline import Spline
from gpaw.poisson import PoissonSolver
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC

L = 2.87 / 0.529177
def f(n):
    N = 2 * n
    gd = GridDescriptor((N, N, N), (L, L, L))
    a = gd.zeros()
    print a.shape
    p = PoissonSolver(nn=1, relax='J')
    p.initialize(gd)
    cut = N / 2.0 * 0.9
    s = Spline(l=0, rmax=cut, f_g=np.array([1, 0.5, 0.0]))
    c = LFC(gd, [[s], [s]])
    c.set_positions([(0, 0, 0), (0.5, 0.5, 0.5)])
    c.add(a)

    I0 = gd.integrate(a)
    a -= gd.integrate(a) / L**3

    I = gd.integrate(a)
    b = gd.zeros()
    p.solve(b, a)#, eps=1e-20)
    return gd.collect(b, broadcast=1)

b = f(8)
assert abs(b[0,0,0]-b[8,8,8]) < 6e-16
