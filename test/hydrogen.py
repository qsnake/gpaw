from math import log
from ase import *
from gpaw import Calculator
from gpaw.utilities import equal

a = 4.0
h = 0.2
hydrogen = Atoms([Atom('H', (a / 2, a / 2, a / 2))],
                 cell=(a, a, a))

calc = Calculator(h=h, nbands=1, convergence={'energy': 1e-6})
hydrogen.set_calculator(calc)
e1 = hydrogen.get_potential_energy()

calc.set(kpts=(1, 1, 1))
e2 = hydrogen.get_potential_energy()
print e1 - e2
equal(e1, e2, 3e-7)

kT = 0.001
calc.set(width=kT)
e3 = hydrogen.get_potential_energy()
equal(e1, e3 + log(2) * kT, 3.0e-7)
