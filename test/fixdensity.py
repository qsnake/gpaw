from ASE import Atom, ListOfAtoms
from gpaw import Calculator

a = 5.0
H = ListOfAtoms([Atom('H', (a/2, a/2, a/2))],
                                periodic=False,
                                cell=(a, a, a))
calc = Calculator(fixdensity=True)
H.SetCalculator(calc)
H.GetPotentialEnergy()
calc = Calculator(fixdensity=3)
H.SetCalculator(calc)
H.GetPotentialEnergy()
