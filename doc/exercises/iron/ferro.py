from ase import *
from gpaw import GPAW

a = 2.87
m = 2.2
bulk = Atoms([Atom('Fe', (0,   0,   0),   magmom=m),
              Atom('Fe', (a/2, a/2, a/2), magmom=m)],
             cell=(a, a, a),
             pbc=True)
calc = GPAW(kpts=(6, 6, 6),
            h=0.20,
            nbands=18,
            eigensolver='cg',
            txt='ferro.txt')
bulk.set_calculator(calc)
print bulk.get_potential_energy()
calc.write('ferro.gpw')
