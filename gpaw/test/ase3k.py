from ase import Atom, Atoms
from ase.io import read
from gpaw import *
from gpaw.test import equal
a = 2.0
calc = GPAW(gpts=(12, 12, 12), txt='H.txt')
H = Atoms([Atom('H')],
          cell=(a, a, a),
          pbc=True,
          calculator=calc)
e0 = H.get_potential_energy()
niter0 = calc.get_number_of_iterations()
del H
H = read('H.txt')
print H.get_potential_energy()

energy_tolerance = 0.00007
niter_tolerance = 0
equal(e0, -6.55685, energy_tolerance)
equal(niter0, 12, niter_tolerance)

print calc.get_xc_functional()
