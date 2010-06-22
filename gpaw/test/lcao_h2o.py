#!/usr/bin/env python
from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.test import equal

a = 6.
b = a / 2
mol = Atoms([Atom('O',(b, b, 0.1219 + b)),
             Atom('H',(b, 0.7633 + b, -0.4876 + b)),
             Atom('H',(b, -0.7633 + b, -0.4876 + b))],
            pbc=False, cell=[a, a, a])
calc = GPAW(gpts=(32, 32, 32), nbands=4, mode='lcao')
mol.set_calculator(calc)
e = mol.get_potential_energy()
niter = calc.get_number_of_iterations()

eref = -10.39054
err = abs(e - eref)

print 'Energy', e
print 'Ref', eref
print 'Err', err

assert err < 1e-4

equal(niter, 8, 0)
