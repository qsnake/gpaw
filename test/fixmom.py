from gpaw import Calculator
from ase import *
from gpaw.utilities import equal

a = 2.87
bulk = Atoms([Atom('Fe', (0, 0, 0), magmom=2.20),
              Atom('Fe', (0.5, 0.5, 0.5), magmom=2.20)],
             pbc=True)
bulk.set_cell((a, a, a), scale_atoms=True)
mom0 = sum(bulk.get_initial_magnetic_moments())
h = 0.20
conv = {'eigenstates': 0.1, 'density':0.1, 'energy':0.1}
calc = Calculator(h=h, nbands=11, kpts=(3, 3, 3),
                  convergence=conv, fixmom=True)
bulk.set_calculator(calc)
e = bulk.get_potential_energy()
mom = calc.get_magnetic_moment()
equal(mom, mom0, 1e-5)

