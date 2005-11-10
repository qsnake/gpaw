from ASE import Atom, ListOfAtoms
from gridpaw import Calculator
from math import pi

d = 1.4
a = 4 * d
x = 0.2
C1 = ListOfAtoms([Atom('C', (d / 2, a / 2, a / 2 + x))],
                 periodic=(1, 0, 0),
                 cell=(d, a, a),
                 angle=-pi / 2
                 )
calc1 = Calculator(nbands=3, h=d / 8, kpts=(8, 1, 1), softgauss=1, usesymm=0,
                   tolerance=1e-11)
C1.SetCalculator(calc1)
e1 = C1.GetPotentialEnergy()

C4 = ListOfAtoms([Atom('C', (1 * d / 2, a / 2, a / 2 + x)),
                  Atom('C', (3 * d / 2, a / 2 + x, a / 2)),
                  Atom('C', (5 * d / 2, a / 2, a / 2 - x)),
                  Atom('C', (7 * d / 2, a / 2 - x, a / 2))],
                 periodic=(1, 0, 0),
                 cell=(4 * d, a, a))
calc4 = Calculator(nbands=12, h=d / 8, kpts=(2, 1, 1), softgauss=1,
                   tolerance=1e-11)
C4.SetCalculator(calc4)
e4 = C4.GetPotentialEnergy()

print e1 - e4 / 4
assert abs(e1 - e4 / 4) < 0.000015
