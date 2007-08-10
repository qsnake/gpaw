from math import log
from ASE import ListOfAtoms, Atom
from gpaw import Calculator
from gpaw.utilities import equal

a = 4.0
h = 0.2
hydrogen = ListOfAtoms([Atom('H', (a / 2, a / 2, a / 2))],
                       cell=(a, a, a))

calc = Calculator(h=h, nbands=1)
hydrogen.SetCalculator(calc)
e1 = hydrogen.GetPotentialEnergy()

calc.set(kpts=(1, 1, 1))
e2 = hydrogen.GetPotentialEnergy()
print e1 - e2
equal(e1, e2, 2e-13)

kT = 0.0001
calc.set(width=kT)
e3 = hydrogen.GetPotentialEnergy()
equal(e1, e3 + log(2) * kT, 2.0e-13)
