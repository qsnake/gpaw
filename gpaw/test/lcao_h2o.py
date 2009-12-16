#!/usr/bin/env python
from ase import *
from gpaw import GPAW
from gpaw.test import equal
from gpaw.hs_operators import Operator

a = 6
b = a / 2
mol = Atoms([Atom('O',(b, b, 0.1219 + b)),
             Atom('H',(b, 0.7633 + b, -0.4876 + b)),
             Atom('H',(b, -0.7633 + b, -0.4876 + b))],
            pbc=False, cell=[a, a, a])
calc = GPAW(nbands=4, h=0.2, mode='fd')
mol.set_calculator(calc)
e = mol.get_potential_energy()
niter = calc.get_number_of_iterations()

eref = -10.3852568107
err = abs(e - eref)

print 'Energy', e
print 'Ref', eref
print 'Err', err

assert err < 1e-4

energy_tolerance = 0.00005
niter_tolerance = 0
equal(e, -10.3852568107, energy_tolerance) # svnversion 5252
equal(niter, 8, niter_tolerance) # svnversion 5252
