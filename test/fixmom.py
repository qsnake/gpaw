from gpaw import Calculator
from ase import *
from gpaw.utilities import equal

a = 2.87
bulk = Atoms([Atom('Fe', (0, 0, 0), magmom=2.20),
              Atom('Fe', (0.5, 0.5, 0.5), magmom=2.20)],
             pbc=True)
bulk.set_cell((a, a, a))
mom0 = sum(bulk.get_magnetic_moments())
h = 0.20
calc = Calculator(h=h, nbands=11, kpts=(3, 3, 3),
                  convergence={'eigenstates': 0.02}, fixmom=True)
bulk.set_calculator(calc)
e = bulk.get_potential_energy()
mom = calc.get_magnetic_moment()
equal(mom, mom0, 1e-5)

