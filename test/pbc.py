"""Make sure we get an exception when an atom is too close to the boundary."""
from ase import *
from gpaw import *
a = 4.0
x = 0.1
hydrogen = Atoms('H', [(x, x, x)],
                 cell=(a, a, a),
                 calculator=GPAW(maxiter=7))
try:
    e1 = hydrogen.get_potential_energy()
except RuntimeError:
    pass
else:
    assert False
