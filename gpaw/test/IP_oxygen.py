#!/usr/bin/env python
from ase import *
from gpaw import GPAW
from gpaw.test import equal

a = 6.0
calc = GPAW(nbands=4)
O = Atoms([Atom('O', (a/2, a/2 + 0.5, a/2), magmom=2)],
          pbc=False, cell=(a, a + 1, a), calculator=calc)
e0 = O.get_potential_energy()
niter0 = calc.get_number_of_iterations()

# calc.set(charge=1) # XXX For some reason changing charge doesn't reset WF
calc = GPAW(nbands=4, charge=1)
O.set_calculator(calc) # XXX should not be needed

e1 = O.get_potential_energy()
niter1 = calc.get_number_of_iterations()

print e1 - e0
assert abs(e1 - e0 - 13.989) < 0.04

energy_tolerance = 0.00001
niter_tolerance = 0
equal(e0, -1.68870788141, energy_tolerance) # svnversion 5252
equal(niter0, 19, niter_tolerance) # svnversion 5252
equal(e1, 12.2762985769, energy_tolerance) # svnversion 5252
equal(niter1, 17, niter_tolerance) # svnversion 5252

# The first ionization energy for LDA oxygen is from this paper:
# In-Ho Lee, Richard M. Martin, Phys. Rev. B 56 7197 (1997)
