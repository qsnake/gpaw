from ASE import Atom, ListOfAtoms
from gpaw.utilities import equal
import Numeric as num
from gpaw import Calculator

bulk = ListOfAtoms([Atom('Li')], periodic=True)
a = 2.7
bulk.SetUnitCell((a, a, a))
k = 4
g = 8
calc = Calculator(gpts=(g, g, g), kpts=(k, k, k), nbands=2, poissonsolver='GS',
 	                  txt=None)
bulk.SetCalculator(calc)
bulk.GetPotentialEnergy()
ave_pot = num.sum(calc.hamiltonian.vHt_g.flat) / (2 * g)**3
equal(ave_pot, 0.0, 1e-8)

calc = Calculator(gpts=(g, g, g), kpts=(k, k, k), nbands=2, poissonsolver='J',
 	                  txt=None)
bulk.SetCalculator(calc)
bulk.GetPotentialEnergy()
ave_pot = num.sum(calc.hamiltonian.vHt_g.flat) / (2 * g)**3
equal(ave_pot, 0.0, 1e-8)

