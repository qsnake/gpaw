from ase import Atoms
from gpaw import GPAW
import numpy as np

a = 2.0
L = 5.0

a1 = Atoms('H', pbc=(1, 0, 0), cell=(a, L, L))
a1.center()
atoms = a1.repeat((4, 1, 1))

def energy(usesymm):
    atoms.set_calculator(GPAW(h=0.3, width=0.1,
                              usesymm=usesymm,
                              kpts=(3,1,1),
                              mode='lcao'))
    return atoms.get_potential_energy()

e1 = energy(False)
e2 = energy(True)

print e1
print e2

assert abs(e2 - e1) < 1e-4
