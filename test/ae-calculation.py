from ase import *
from gpaw import Calculator
from gpaw.utilities import equal

a = 5.0
H = Atoms('H', [(a/2, a/2, a/2)], magmoms=[1],
          pbc=False,
          cell=(a, a, a))

H.set_calculator(Calculator(h=0.1, setups='ae', fixmom=True))
e1 = H.get_potential_energy()

c = a / 2.0
d = 0.74
s = d / 2 / 3**0.5
H2 = Atoms('H2', 
           [(c - s, c - s, c - s),
            (c + s, c + s, c + s)],
           pbc=False,
           cell=(a, a, a))

H2.set_calculator(Calculator(h=0.1, setups='ae'))
e2 = H2.get_potential_energy()
print e1, e2, 2 * e1 - e2
equal(2 * e1 - e2, 4.55354238957, 1e-5)
