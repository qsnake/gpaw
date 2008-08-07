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

view(bulk)

k = 4
calc = GPAW(nbands=16,          # number of electronic bands
            h=0.2,              # grid spacing
            kpts=(k, k, k),     # k-points
            txt=name + '.txt')  # output file
bulk.set_calculator(calc)

energy = bulk.get_potential_energy()
calc.write(name + '.gpw')
print 'Energy:', energy, 'eV'
