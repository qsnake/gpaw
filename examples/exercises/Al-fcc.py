"""Bulk Al(fcc) test"""

from gpaw import Calculator
from ASE import Atom, ListOfAtoms

filename = 'Al-fcc'

a = 4.05   # fcc lattice paramter
b = a / 2. 

bulk = ListOfAtoms([Atom('Al', (0, 0, 0)),
                    Atom('Al', (b, b, 0)),
                    Atom('Al', (0, b, b)),
                    Atom('Al', (b, 0, b)),],
                   cell=(a, a, a),
                   periodic=(1, 1, 1))

calc = Calculator(nbands=16,                 # Set the number of electronic bands
                  h=0.2,                     # Set the grid spacing
                  kpts=(6,6,6),              # Set the k-points
                  txt=filename+'.txt')       # Set output file

bulk.SetCalculator(calc)

energy = bulk.GetPotentialEnergy()

calc.write(filename+'.gpw')

print energy
