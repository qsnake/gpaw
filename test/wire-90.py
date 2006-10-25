from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from math import pi

a = 4.0
d = 1.0
x = 0.25
H1 = ListOfAtoms([Atom('H', (d / 2, a / 2, a / 2 + x))],
                 periodic=(1, 0, 0),
                 cell=(d, a, a),
                 angle=-pi / 2
                 )
calc1 = Calculator(nbands=1, h=0.25, kpts=(8, 1, 1), softgauss=0, usesymm=0,
                   tolerance=1e-11)
H1.SetCalculator(calc1)
e1 = H1.GetPotentialEnergy()

H4 = ListOfAtoms([Atom('H', (1 * d / 2, a / 2, a / 2 + x)),
                  Atom('H', (3 * d / 2, a / 2 + x, a / 2)),
                  Atom('H', (5 * d / 2, a / 2, a / 2 - x)),
                  Atom('H', (7 * d / 2, a / 2 - x, a / 2))],
                 periodic=(1, 0, 0),
                 cell=(4 * d, a, a))
calc4 = Calculator(nbands=4, h=0.25, kpts=(2, 1, 1), softgauss=0,
                   tolerance=1e-11)
H4.SetCalculator(calc4)
e4 = H4.GetPotentialEnergy()

print e1 - e4 / 4
assert abs(e1 - e4 / 4) < 3e-5
