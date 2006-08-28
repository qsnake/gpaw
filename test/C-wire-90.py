from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from math import pi
#from ASE.Visualization.RasMol import RasMol

d = 1.4
a = 4 * d
x = 0.52
y = x / 2**0.5
C2 = ListOfAtoms([Atom('C', (0 * d / 2, a / 2, a / 2 + x)),
                  Atom('C', (2 * d / 2, a / 2 + y, a / 2 + y))],
                 periodic=(1, 0, 0),
                 cell=(2 * d, a, a),
                 angle=pi / 2
                 )
calc2 = Calculator(nbands=6, h=d / 8, kpts=(8, 1, 1),
                   tolerance=1e-11)
C2.SetCalculator(calc2)
e2 = C2.GetPotentialEnergy()

C8 = ListOfAtoms([Atom('C', (0 * d / 2, a / 2, a / 2 + x)),
                  Atom('C', (2 * d / 2, a / 2 + y, a / 2 + y)),
                  Atom('C', (4 * d / 2, a / 2 + x, a / 2)),
                  Atom('C', (6 * d / 2, a / 2 + y, a / 2 - y)),
                  Atom('C', (8 * d / 2, a / 2, a / 2 - x)),
                  Atom('C', (10 * d / 2, a / 2 - y, a / 2 - y)),
                  Atom('C', (12 * d / 2, a / 2 - x, a / 2)),
                  Atom('C', (14 * d / 2, a / 2 - y, a / 2 + y))],
                 periodic=(1, 0, 0),
                 cell=(8 * d, a, a))
calc8 = Calculator(nbands=24, h=d / 8, kpts=(2, 1, 1),
                   tolerance=1e-11)
C8.SetCalculator(calc8)
e8 = C8.GetPotentialEnergy()

print e2 / 2 - e8 / 8
assert abs(e2 / 2 - e8 / 8) < 6e-6
