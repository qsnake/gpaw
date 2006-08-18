from gridpaw import Calculator
from gridpaw.utilities import equal
from ASE import ListOfAtoms, Atom

a = 4.0
h = 0.2
hydrogen = ListOfAtoms([Atom('H', (a / 2, a / 2, a / 2))],
                       cell=(a, a, a))

calc = Calculator(h=h, nbands=1)
hydrogen.SetCalculator(calc)
e1 = hydrogen.GetPotentialEnergy()

calc.Set(kpts=(1, 1, 1))
e2 = hydrogen.GetPotentialEnergy()
print e1 - e2
equal(e1, e2, 3e-15)

calc.Set(width=0.00001)
e3 = hydrogen.GetPotentialEnergy()
equal(e1, e3, 1e-4)
