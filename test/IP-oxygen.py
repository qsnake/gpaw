#!/usr/bin/env python
from ase import *
from gpaw import Calculator

a = 6.0
o = Atoms([Atom('O', (a/2, a/2 + 0.5, a/2), magmom=2)],
                pbc=0,
                cell=(a, a + 1, a))
calc = Calculator(nbands=4, h=0.2, charge=0, hund=1, fixdensity=0)
o.set_calculator(calc)
e0 = o.get_potential_energy()

o = Atoms([Atom('O', (a/2, a/2 + 0.5, a/2), magmom=2)],
                pbc=0,
                cell=(a, a + 1, a))
calc = Calculator(nbands=4, h=0.2, charge=1)
o.set_calculator(calc)
e1 = o.get_potential_energy()

print e1 - e0
assert abs(e1 - e0 - 13.989) < 0.04

# The first ionization energy for LDA oxygen is from this paper:
# In-Ho Lee, Richard M. Martin, Phys. Rev. B 56 7197 (1997)
