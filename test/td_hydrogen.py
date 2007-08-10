#!/usr/bin/env python
from cmath import exp
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
import Numeric as num
from gpaw.tddft import TDDFT
import os

a = 5.0
atoms = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=1)],
                    periodic=False,
                    cell=(a, a, a))
calc = Calculator(nbands=1, h=0.2, tolerance=1e-14)
atoms.SetCalculator(calc);
e = atoms.GetPotentialEnergy()

calc.write('hydrogen.gpw', 'all')
calc = Calculator('hydrogen.gpw')

paw = calc
td_atoms = TDDFT(paw, tolerance=1e-11)

print 'eps = ', paw.kpt_u[0].eps_n[0]
eps = paw.kpt_u[0].eps_n[0]
time_step = 0.05
psi0 = num.array(paw.kpt_u[0].psit_nG[0])
print '%8s  %16s  %16s' % ('time  ', 'Error of norm', 'Error of phase')
for i in range(50):
    time = i * time_step
    c = num.vdot(psi0, paw.kpt_u[0].psit_nG[0]) / abs(num.vdot(psi0, psi0))
    err = abs(c - exp(-1j * eps * time))
    #print time, c, exp(-1j * eps * time), err, abs(c) - 1.0
    print '%8lf  %16.6le  %16.6le' % (time, abs(c) - 1.0, err)
    td_atoms.propagate(time_step)

os.remove('hydrogen.gpw')

assert(err < 1e-5)
