from ase import *
from gpaw import Calculator
from gpaw.utilities import equal

a = 4.05
d = a / 2**0.5
bulk = Atoms([Atom('Al', (0, 0, 0)),
              Atom('Al', (0.5, 0.5, 0.5))],
             pbc=True)
bulk.set_cell((d, d, a), scale_atoms=True)
h = 0.25
calc = Calculator(h=h,
                  nbands=2*8,
                  kpts=(2, 2, 2),
                  convergence={'energy': 1e-5})
bulk.set_calculator(calc)
e0 = bulk.get_potential_energy()
calc = Calculator(h=h,
                  nbands=2*8,
                  kpts=(2, 2, 2),
                  convergence={'energy': 1e-5},
                  eigensolver='cg')
bulk.set_calculator(calc)
e1 = bulk.get_potential_energy()
equal(e0, e1, 3.6e-5)
