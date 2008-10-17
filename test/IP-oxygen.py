#!/usr/bin/env python
from ase import *
from gpaw import GPAW

a = 6.0
calc = GPAW(nbands=4)
O = Atoms([Atom('O', (a/2, a/2 + 0.5, a/2), magmom=2)],
          pbc=False, cell=(a, a + 1, a), calculator=calc)
e0 = O.get_potential_energy()

# calc.set(charge=1) # XXX For some reason changing charge doesn't reset WF
O.set_calculator(GPAW(nbands=4, charge=1)) # XXX should not be needed

e1 = O.get_potential_energy()

print e1 - e0
assert abs(e1 - e0 - 13.989) < 0.04

# The first ionization energy for LDA oxygen is from this paper:
# In-Ho Lee, Richard M. Martin, Phys. Rev. B 56 7197 (1997)
