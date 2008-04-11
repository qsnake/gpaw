from ase import *
from gpaw import Calculator
bulk = Atoms([Atom('Li')], pbc=True)
k = 4
g = 8
calc = Calculator(gpts=(g, g, g), kpts=(k, k, k), nbands=2)
bulk.set_calculator(calc)
e = []
A = [2.6, 2.65, 2.7, 2.75, 2.8]
for a in A:
    bulk.set_cell((a, a, a))
    e.append(bulk.get_potential_energy())
print e

import numpy as npy
a = npy.roots(npy.polyder(npy.polyfit(A, e, 2), 1))[0]
print 'a =', a
assert abs(a - 2.64931934872) < 0.0001
