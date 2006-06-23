from ASE import Atom, ListOfAtoms
from gridpaw import Calculator

h = 0.2
n = 24
a = n * h
H2 = ListOfAtoms([Atom('He', [0.123, 0.234, 0.345]),
                 Atom('He', [2.523, 2.634, 0.345])],
                periodic=True,
                cell=(a, a, a))
calc = Calculator(nbands=2, gpts=(n, n, n), hosts=8, out='tmp')
H2.SetCalculator(calc)
e0 = H2.GetPotentialEnergy()
#for i in range(51):
for i in range(3):
    e = H2.GetPotentialEnergy()
    print i * a / 25, e - e0
    H2[0].SetCartesianPosition(H2[0].GetCartesianPosition() + (a / 25, 0, 0))
    H2[1].SetCartesianPosition(H2[1].GetCartesianPosition() + (0, 0, a / 25))
#assert abs(e - e0) < 1e-5
