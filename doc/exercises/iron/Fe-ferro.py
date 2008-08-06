from ase import *
from gpaw import Calculator

a = 2.87
bulk = Atoms([Atom('Fe', (0,   0,   0),   magmom=?),
              Atom('Fe', (a/2, a/2, a/2), magmom=?)],
             cell=(a, a, a), pbc=True)
calc = Calculator(kpts=(8, 8, 8),
                  h = 0.25,
                  nbands=?,
                  txt='ferro.txt')
bulk.set_calculator(calc)
print bulk.get_potential_energy()
calc.write('ferro.gpw')
