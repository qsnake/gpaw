from ASE import Atom, ListOfAtoms
from gridpaw import Calculator
from math import pi

a = 4.0
d = 1.0
H1 = ListOfAtoms([Atom('H', (d / 2, a / 2, a / 2 + 0.25))],
                 periodic=(1, 0, 0),
                 cell=(d, a, a),
                 angle=pi)
calc1 = Calculator(nbands=1, h=0.25, kpts=(4, 1, 1), softgauss=0, usesymm=0)
H1.SetCalculator(calc1)
e1 = H1.GetPotentialEnergy()

H2 = ListOfAtoms([Atom('H', (d / 2, a / 2, a / 2 + 0.25)),
                  Atom('H', (3 * d / 2, a / 2, a / 2 - 0.25))],
                 periodic=(1, 0, 0),
                 cell=(2 * d, a, a))
calc2 = Calculator(nbands=2, h=0.25, kpts=(2, 1, 1), softgauss=0)
H2.SetCalculator(calc2)
e2 = H2.GetPotentialEnergy()

assert abs(e1 - e2 / 2) < 0.00015
