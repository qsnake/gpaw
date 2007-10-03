#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
import gpaw
import Numeric as num
from gpaw.tddft import TDDFT
import math
import sys
import os
import gpaw.io.array as ioarray

if 1:
    a = 9.0
    atoms = ListOfAtoms([Atom('Be',(a/2, a/2, a/2), magmom=0)],
                        periodic=False,
                        cell=(a, a, a))
    calc = Calculator(nbands=1, h=0.2, convergence={'eigenstates': 1e-14}, setups='paw')
    atoms.SetCalculator(calc)
    e = atoms.GetPotentialEnergy()
    calc.write('Be.gpw', 'all')
else:
    calc = Calculator('Be.gpw')

paw = calc
rhot0 = calc.density.rhot_g.copy()
dm0 = calc.finegd.calculate_dipole_moment(rhot0)
norm0 = calc.finegd.integrate(rhot0)

td_atoms = TDDFT(paw, propagator='SICN', tolerance=1e-8)
td_atoms.absorption_kick(strength=1.0e-4)
calc.density.charge_eps = 1e-6

time_step = 0.05

print ('%16.6le  %16.6le %16.6le %16.6le' %
       (norm0, dm0[0], dm0[1], dm0[2]))

print '%8s  %16s  %16s' \
        % ('time  ', 'Error of norm', 'Dipole moment')

for i in range(10000):
    time = i*time_step

    rhot = calc.density.rhot_g
    dm = calc.finegd.calculate_dipole_moment(rhot) - dm0
    norm = calc.finegd.integrate(rhot) - norm0
    print ('%8lf  %16.6le  %16.6le %16.6le %16.6le' %
           (time, norm, dm[0], dm[1], dm[2]))
    sys.stdout.flush()
    td_atoms.propagate(time_step)

drho = rhot - rhot0
print 'Density'
nc = calc.finegd.N_c[0]
for i in range(nc - 1):
    print '%8lf  %16.6le %16.6le' % (0.529177 * (i + 1) * calc.finegd.h_c[0],
                                     drho[i, nc/2, nc/2],
                                     drho[nc/2, nc/2, i])


