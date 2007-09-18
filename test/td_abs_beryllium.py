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

a = 7.0
atoms = ListOfAtoms([Atom('Be',(a/2, a/2, a/2), magmom=0)],
                    periodic=False,
                    cell=(a, a, a))
calc = Calculator(nbands=1, h=0.2, tolerance=1e-16, setups='paw')
atoms.SetCalculator(calc)
e = atoms.GetPotentialEnergy()

calc = atoms.GetCalculator()

paw = calc
td_atoms = TDDFT(paw, propagator='SICN', tolerance=1e-8)
td_atoms.absorption_kick(strength=1.0e-4)
calc.density.charge_eps = 1e-3

print 'eps = ', paw.kpt_u[0].eps_n[0]
eps = paw.kpt_u[0].eps_n[0]
time_step = 0.05

rho_ae0 = calc.density.get_all_electron_density()
dm0 = calc.finegd.calculate_dipole_moment(rho_ae0)
norm0 = calc.finegd.integrate(rho_ae0)

print '%8s  %16s  %16s' \
        % ('time  ', 'Error of norm', 'Dipole moment')
for i in range(200):
    time = i*time_step
    rho_ae = calc.density.get_all_electron_density()
    dm = calc.finegd.calculate_dipole_moment(rho_ae) - dm0
    norm = calc.finegd.integrate(rho_ae) - norm0
    print '%8lf  %16.6le  %16.6le %16.6le %16.6le' % (time, norm, dm[0], dm[1], dm[2])
    sys.stdout.flush()
    td_atoms.propagate(time_step)

rho_ae = calc.density.get_all_electron_density()
drho_ae = rho_ae - rho_ae0
print 'Density'
nc = calc.finegd.N_c[0]
for i in range(nc-1):
    print '%8lf  %16.6le %16.6le' % ( 0.529177*(i+1)*calc.finegd.h_c[0], drho_ae[i,nc/2,nc/2], drho_ae[nc/2,nc/2,i] )


