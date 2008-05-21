from ase import *
from gpaw import Calculator
from gpaw.atom.basis import BasisMaker

sym = 'Li'
bm = BasisMaker(sym, run=False)
bm.generator.N = 300
bm.generator.run(write_xml=False)
basis = bm.generate(2)

bulk = Atoms([Atom(sym)], pbc=True)
k = 4
g = 8
calc = Calculator(gpts=(g, g, g), kpts=(k, k, k),
                  eigensolver='lcao', basis={sym : basis})
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
assert abs(a - 2.8498) < 0.0001
