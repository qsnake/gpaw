from ase import *
from gpaw import GPAW
from gpaw.utilities import equal


a = 4.0
n = 16
hydrogen = Atoms([Atom('H')], cell=(a, a, a), pbc=True)
hydrogen.center()
calc = GPAW(gpts=(n, n, n), nbands=1, convergence={'energy': 1e-5})#, mode='lcao')
hydrogen.set_calculator(calc)
e1 = hydrogen.get_potential_energy()
hydrogen.set_initial_magnetic_moments([1.0])
e2 = hydrogen.get_potential_energy()
de = e1 - e2
print de
equal(de, 0.7918, 1.e-4)
