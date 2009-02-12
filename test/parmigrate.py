from ase import *
from gpaw import GPAW

h = 0.2
n = 24
a = n * h
H2 = Atoms([Atom('He', [0.123, 0.234, 0.345]),
                  Atom('He', [2.523, 2.634, 0.345])],
                 pbc=True,
                 cell=(a, a, a))
calc = GPAW(nbands=2, gpts=(n, n, n), #hosts=8,
                  txt='tmp')
H2.set_calculator(calc)
e0 = H2.get_potential_energy()
for i in range(51):
    e = H2.get_potential_energy()
    print i * a / 25, e - e0
    calc.txt.flush()
    H2[0].set_cartesian_position(H2[0].get_cartesian_position() + (a / 25, 0, 0))
    H2[1].set_cartesian_position(H2[1].get_cartesian_position() + (0, 0, a / 25))
assert abs(e - e0) < 1e-5
