import os
import sys

from ase import *
from ase.units import Bohr, Hartree
from ase.io.cube import write_cube
from gpaw import Calculator
from gpaw.utilities import equal

from gpaw.cluster import Cluster
from gpaw.point_charges import PointCharges, ConstantPotential

cp = ConstantPotential()

sc = 2.
R=0.7 # approx. experimental bond length
R=1.
a = 2 * sc
c = 3 * sc
at='Na'
H2 = Atoms([Atom(at, (a/2, a/2, (c-R)/2)),
            Atom(at, (a/2, a/2, (c+R)/2))],
           cell=(a,a,c), pbc=False)
print at, 'dimer'
nelectrons = 2 * H2[0].number

txt='/dev/null'
#txt='-'

# without potential
if True:
##    print '\n################## no potential'
    c00 = Calculator(h=0.3, nbands=-1, txt=txt)
    c00.calculate(H2)
    eps00_n = c00.kpt_u[0].eps_n

# 0 potential
if False:
##    print '\n################## 0 potential'
    cp0 = ConstantPotential(0.0)
    c01 = Calculator(h=0.3, nbands=-2, external=cp0, txt=txt)
    c01.calculate(H2)

# 1 potential
if True:
##    print '################## 1 potential'
    cp1 = ConstantPotential(-1.0/Hartree)
    c1 = Calculator(h=0.3, nbands=-2, external=cp1, txt=txt)
    c1.calculate(H2)

for i in range(c00.nbands):
    f00 = c00.kpt_u[0].f_n[i]
    if f00 > 0.01:
        e00 = c00.kpt_u[0].eps_n[i] * Hartree
        e1 = c1.kpt_u[0].eps_n[i] * Hartree
        print 'Eigenvalues no pot, expected, error=', e00, e1 + 1, e00 - e1 - 1
        equal(e00, e1 + 1., 0.002)

DeltaE = c00.get_potential_energy() - c1.get_potential_energy()
print 'Energy diff, expected, error=', DeltaE, nelectrons, DeltaE - nelectrons
equal(DeltaE, nelectrons, 0.002)
