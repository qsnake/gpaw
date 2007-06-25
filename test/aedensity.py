# This test takes approximately 50.4 seconds
#!/usr/bin/env python
import Numeric as num

from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import center, equal

h = 0.17 # gridspacing
a = [6.5, 6.5, 7.5] # unit cell
d = 2.3608 # experimental bond length
gridrefinement = 2 # grid-refinement-factor for all-electron density

NaCl = ListOfAtoms([Atom('Na', [0, 0, 0]),
                    Atom('Cl', [0, 0, d])],
                   periodic=False, cell=a)
center(NaCl)
calc = Calculator(h=h, xc='LDA', nbands=5, lmax=0, tolerance=1e-6, spinpol=1)

NaCl.SetCalculator(calc)
NaCl.GetPotentialEnergy()

nt = calc.GetDensityArray()
n = calc.GetAllElectronDensity(gridrefinement)

Zt = num.sum(nt.flat) * num.product(calc.GetGridSpacings())
Z = num.sum(n.flat) * num.product(calc.GetGridSpacings() / gridrefinement)

print 'Integral of pseudo density:', Zt
print 'Integral of all-electron density:', Z

equal(Z, 28, 1e-5)
