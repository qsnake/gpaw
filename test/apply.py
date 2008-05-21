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
conv = {'density': 1.0e-1, 'eigenstates': 1.0e-4, 'energy': 1e-2}
calc = Calculator(h=h,
                  kpts=(2, 2, 2),
                  convergence=conv,
                  eigensolver='rmm-diis2')
bulk.set_calculator(calc)
e0 = bulk.get_potential_energy()
calc = Calculator(h=h,
                  kpts=(2, 2, 2),
                  convergence=conv)
bulk.set_calculator(calc)
e1 = bulk.get_potential_energy()
equal(e0, e1, 1.0e-10)
