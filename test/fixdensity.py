from ase import *
from gpaw import Calculator

a = 5.0
H = Atoms([Atom('H', (a/2, a/2, a/2))],
                                pbc=False,
                                cell=(a, a, a))
calc = Calculator(fixdensity=True)
H.set_calculator(calc)
H.get_potential_energy()
calc = Calculator(fixdensity=3)
H.set_calculator(calc)
H.get_potential_energy()
