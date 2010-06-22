from math import log
from ase import Atom, Atoms
from gpaw import GPAW, FermiDirac
from gpaw.test import equal

a = 4.0
h = 0.2
hydrogen = Atoms([Atom('H', (a / 2, a / 2, a / 2))],
                 cell=(a, a, a))

calc = GPAW(h=h, nbands=1, convergence={'energy': 1e-7})
hydrogen.set_calculator(calc)
e1 = hydrogen.get_potential_energy()
niter1 = calc.get_number_of_iterations()

calc.set(kpts=(1, 1, 1))
e2 = hydrogen.get_potential_energy()
niter2 = calc.get_number_of_iterations()
print e1 - e2
equal(e1, e2, 3e-7)

kT = 0.001
calc.set(occupations=FermiDirac(width=kT))
e3 = hydrogen.get_potential_energy()
niter3 = calc.get_number_of_iterations()
equal(e1, e3 + log(2) * kT, 3.0e-7)

energy_tolerance = 0.00006
niter_tolerance = 0
equal(e1, 0.526939, energy_tolerance)
assert 24 <= niter1 <= 26, niter1
equal(e2, 0.526938, energy_tolerance)
equal(niter2, 3, niter_tolerance)
equal(e3, 0.526245, energy_tolerance)
equal(niter3, 11, niter_tolerance)
