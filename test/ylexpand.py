import os

from ASE import Atom
from gpaw import Calculator
from gpaw.utilities import equal
from gpaw.cluster import Cluster
from gpaw.analyse.expandyl import ExpandYl

R=0.7 # approx. experimental bond length
a = 2
c = 3
H2 = Cluster([Atom('H', (a/2,a/2,(c-R)/2)),
              Atom('H', (a/2,a/2,(c+R)/2))],
             cell=(a,a,c))
calc = Calculator(xc='LDA',nbands=2,spinpol=False,
                  convergence={'eigenstates':1.e-6})
H2.SetCalculator(calc)
H2.GetPotentialEnergy()

yl = ExpandYl(H2.center_of_mass(), calc.gd, Rmax=1.5)

def max_index(l):
    mi = 0
    limax = l[0]
    for i, li in enumerate(l):
        if limax < li:
            limax = li
            mi = i
    return mi

# check numbers
for n in [0,1]:
    gl, w = yl.expand(calc.kpt_u[0].psit_nG[n])
    print 'max_index(gl), n=', max_index(gl), n
    assert(max_index(gl) == n)

# io
fname = 'expandyl.dat'
yl.to_file(calc,fname)
os.remove(fname)
