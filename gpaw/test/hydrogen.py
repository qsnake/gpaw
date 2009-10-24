from math import log
from ase import *
from gpaw import GPAW
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
calc.set(width=kT)
e3 = hydrogen.get_potential_energy()
niter3 = calc.get_number_of_iterations()
equal(e1, e3 + log(2) * kT, 3.0e-7)

energy_tolerance = 0.000001
niter_tolerance = 0
equal(e1, 0.521945345886, energy_tolerance) # svnversion 5252
equal(niter1, 25, niter_tolerance) # svnversion 5252
equal(e2, 0.521945343668, energy_tolerance) # svnversion 5252
equal(niter2, 3, niter_tolerance) # svnversion 5252
equal(e3, 0.521252179047, energy_tolerance) # svnversion 5252
equal(niter3, 11, niter_tolerance) # svnversion 5252
