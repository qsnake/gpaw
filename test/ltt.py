import os
from ASE import Atom, ListOfAtoms
from gridpaw import Calculator
from gridpaw.utilities import equal
h = 0.2
n = 24
a = n * h
b = a / 2
H = ListOfAtoms([Atom('H', [b, b, b])],
                periodic=True,
                cell=(a, a, a))
calc = Calculator(nbands=1, gpts=(n, n, n), out=None)
H.SetCalculator(calc)
e0 = H.GetPotentialEnergy()
cmd = 'ps -eo comm,pmem | grep python'
mem0 = float(os.popen(cmd).readlines()[-1].split()[-1])
for i in range(50):
    e = H.GetPotentialEnergy()
    H[0].SetCartesianPosition(H[0].GetCartesianPosition() +
                              (0.0123456789, 0.023456789, 0.03456789))
mem = float(os.popen(cmd).readlines()[-1].split()[-1])
equal(e, e0, 0.0002)
assert mem < mem0 + 0.11
