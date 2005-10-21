from ASE import Atom, ListOfAtoms
from gridpaw import Calculator

a = 6.0
H = ListOfAtoms([Atom('H',magmom=1)],
                periodic=True,
                cell=(a, a, a))
calc = Calculator(nbands=1, h=0.20, onohirose=5, tolerance=0.001, softgauss=0)
H.SetCalculator(calc)
print H.GetPotentialEnergy()

