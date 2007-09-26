from ASE import Atom, ListOfAtoms
from gpaw import Calculator
bulk = ListOfAtoms([Atom('Li')], periodic=True)
k = 4
g = 8
calc = Calculator(gpts=(g, g, g), kpts=(k, k, k), nbands=2)
bulk.SetCalculator(calc)
e = []
A = [2.6, 2.65, 2.7, 2.75, 2.8]
for a in A:
    bulk.SetUnitCell((a, a, a))
    e.append(bulk.GetPotentialEnergy())
print e

try:
    import numpy as npy
except ImportError:
    pass
else:
    a = npy.roots(npy.polyder(npy.polyfit(A, e, 2), 1))[0]
    print 'a =', a
    assert abs(a - 2.6503) < 0.0001
