#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
import gpaw
import Numeric as num
from gpaw.tddft import TDDFT
import math
import sys
import os

a = 5.0
atoms = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=1)],
                    periodic=False,
                    cell=(a, a, a))
calc = Calculator(nbands=1, h=0.2, tolerance=1e-14)
atoms.SetCalculator(calc);
e = atoms.GetPotentialEnergy()

calc.Write('hydrogen.nc')
atoms = Calculator.ReadAtoms('hydrogen.nc')
calc = atoms.GetCalculator()

paw = calc.paw
td_atoms = TDDFT(paw,tolerance=1e-11)

print 'eps = ', paw.kpt_u[0].eps_n[0]
eps = paw.kpt_u[0].eps_n[0]
time_step = .05
psi0 = num.array( paw.kpt_u[0].psit_nG[0] )
print '%8s  %16s  %16s' \
        % ('time  ', 'Error of norm', 'Error of phase')
for i in range(50):
    time = i*time_step
    c = num.sum(num.sum(num.sum( num.conjugate(psi0) * paw.kpt_u[0].psit_nG[0]))) / abs(num.sum(num.sum(num.sum( num.conjugate(psi0) * psi0))))
    err = math.sqrt( (c.real - math.cos(eps*time)) * (c.real - math.cos(eps*time)) \
                + (c.imag + math.sin(eps*time)) * (c.imag + math.sin(eps*time)) )
    #print '%8lf  %20.6le  %20.6le  %18.6le  %18.6le  %12.6le  %12.6le' % (time, c.real, c.imag, math.cos(eps*time), -math.sin(eps*time), err, abs(c) - 1.)
    print '%8lf  %16.6le  %16.6le' % (time, abs(c) - 1., err)
    sys.stdout.flush()
    td_atoms.propagate(time_step)

os.remove('hydrogen.nc')

assert(err < 1e-5)
