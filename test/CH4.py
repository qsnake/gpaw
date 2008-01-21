from gpaw import Calculator
from ase import *


a = 4.0
n = 20
d = 1.0
x = d / 3**0.5
atoms = Atoms([Atom('C', (0.0, 0.0, 0.0)),
               Atom('H', (x, x, x)),
               Atom('H', (-x, -x, x)),
               Atom('H', (x, -x, -x)),
               Atom('H', (-x, x, -x))],
              cell=(a, a, a), pbc=True)
atoms.set_calculator(Calculator(gpts=(n, n, n), nbands=4, txt='-'))
e0 = atoms.get_potential_energy()

for d in [1.0, 1.05, 1.1, 1.15]:
    x = d / 3**0.5
    atoms.positions[1] = (x, x, x)
    print d, atoms.get_potential_energy() - e0
