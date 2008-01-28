from ase import *
from gpaw import *
a = 2.0
H = Atoms([Atom('H')],
          cell=(a, a, a),
          pbc=True,
          calculator=Calculator(txt='H.txt'))
e0 = H.get_potential_energy()
del H
H = read('H.txt')
print H.get_potential_energy()
