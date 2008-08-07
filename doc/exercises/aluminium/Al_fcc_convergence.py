"""Bulk Al(fcc) test"""

from ase import *
from gpaw import *

name = 'Al-fcc'
a = 4.05   # fcc lattice paramter
b = a / 2 
bulk = Atoms(symbols='4Al',
             positions=[(0, 0, 0),
                        (b, b, 0),
                        (0, b, b),
                        (b, 0, b)],
             cell=(a, a, a),
             pbc=True)

# Convergence with respect to k-points:
calc = GPAW(nbands=16, h=0.3, txt=name + '-k.txt')
bulk.set_calculator(calc)
f = open(name + '-k.dat', 'w')
for k in [2, 4, 6, 8]: 
    calc.set(kpts=(k, k, k))
    energy = bulk.get_potential_energy() 
    print k, energy
    print >> f, k, energy

# Convergence with respect to grid spacing:
k = 4
calc = GPAW(nbands=16, kpts=(k, k, k), txt=name + '-h.txt')
bulk.set_calculator(calc)
f = open(name + '-h.dat', 'w')
for g in [12, 16, 20]:
    h = a / g
    calc.set(h=h)
    energy = bulk.get_potential_energy() 
    print h, energy
    print >> f, h, energy
