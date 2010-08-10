"""Bulk Al(fcc) test"""

from ase import Atoms
from ase.visualize import view
from gpaw import GPAW

name = 'Al-fcc'
a = 4.05  # fcc lattice paramter
b = a / 2 

bulk = Atoms('Al',
             cell=[[0, b, b],
                   [b, 0, b],
                   [b, b, 0]],
             pbc=True)

view(bulk)

k = 4
calc = GPAW(h=0.2,              # grid spacing
            kpts=(k, k, k),     # k-points
            txt=name + '.txt')  # output file

bulk.set_calculator(calc)

energy = bulk.get_potential_energy()
calc.write(name + '.gpw')
print 'Energy:', energy, 'eV'
