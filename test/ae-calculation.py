from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import equal

a = 5.0
H = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=1)],
                periodic=False,
                cell=(a, a, a))

H.SetCalculator(Calculator(h=0.1, setups='ae', fixmom=True))
e1 = H.GetPotentialEnergy()

c = a / 2.0
d = 0.74
s = d / 2 / 3**0.5
H2 = ListOfAtoms([Atom('H', (c - s, c - s, c - s)),
                  Atom('H', (c + s, c + s, c + s))],
                 periodic=False,
                 cell=(a, a, a))

H2.SetCalculator(Calculator(h=0.1, setups='ae'))
e2 = H2.GetPotentialEnergy()
print e1, e2, 2 * e1 - e2
equal(2 * e1 - e2, 4.55354160381, 1e-5)
