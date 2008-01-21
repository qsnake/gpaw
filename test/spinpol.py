from ase import *
from gpaw import Calculator
from gpaw.utilities import equal


a = 4.0
n = 16
hydrogen = Atoms([Atom('H')], cell=(a, a, a), pbc=True)
calc = Calculator(gpts=(n, n, n), nbands=1, convergence={'energy': 1e-5})
hydrogen.set_calculator(calc)
e1 = hydrogen.get_potential_energy()
calc.set(spinpol=True)
hydrogen.set_magnetic_moments([1.0])
e2 = hydrogen.get_potential_energy()
de = e1 - e2
print de
equal(de, 0.7918, 1.e-4)
