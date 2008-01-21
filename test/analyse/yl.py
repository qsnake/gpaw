import numpy as npy

from ASE import Atom

from gpaw.utilities import equal
from gpaw.cluster import Cluster
from gpaw import Calculator
from gpaw.analyse.expandyl import ExpandYl

R = 1.
H2 = Cluster([Atom('H',(0,0,0)), Atom('H',(0,0,R))])
H2.minimal_box(3.)

calc = Calculator(h=0.2, width=0.01, nbands=2)
H2.SetCalculator(calc)
H2.GetPotentialEnergy()

yl = ExpandYl(H2.center_of_mass(), calc.gd, Rmax=2.5)
gl = []
for n in range(calc.nbands):
    psit_G = calc.kpt_u[0].psit_nG[n]
    norm = calc.gd.integrate(psit_G**2)
    g = yl.expand(psit_G)
    gsum = npy.sum(g)

    # allow for 10 % inaccuracy in the norm
    print "norm, sum=", norm, gsum
    equal(norm, gsum, 0.1)
    
    gl.append(g/gsum*100)

# 1 sigma_g has s-symmetry mainly
print gl[0]
equal( gl[0][0], 100, 10)
# 1 sigma_u has p-symmetry mainly
print gl[1]
equal( gl[1][1], 100, 10)
