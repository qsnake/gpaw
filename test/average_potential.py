from ase import *
from gpaw.utilities import equal
import numpy as np
from gpaw import GPAW
from gpaw.poisson import PoissonSolver

a = 2.7
bulk = Atoms('Li', pbc=True, cell=(a, a, a))
k = 4
g = 8
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2,
                  poissonsolver=PoissonSolver(relax='GS'),
                  txt=None)
bulk.set_calculator(calc)
bulk.get_potential_energy()
ave_pot = np.sum(calc.hamiltonian.vHt_g.ravel()) / (2 * g)**3
equal(ave_pot, 0.0, 1e-8)

calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2,
                  poissonsolver=PoissonSolver(relax='J'),
                  txt=None)
bulk.set_calculator(calc)
bulk.get_potential_energy()
ave_pot = np.sum(calc.hamiltonian.vHt_g.ravel()) / (2 * g)**3
equal(ave_pot, 0.0, 1e-8)

