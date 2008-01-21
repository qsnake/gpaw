#!/usr/bin/env python
from ase import *
from gpaw import Calculator, PoissonSolver
import gpaw
import numpy as npy
from gpaw.tddft import TDDFT
import math
import sys
import os
import gpaw.io.array as ioarray

from gpaw.mpi import rank

if 0:
    a = 9.0
    atoms = Atoms([Atom('Be',(a/2, a/2, a/2), magmom=0)],
                        pbc=False,
                        cell=(a, a, a))
    calc = Calculator(nbands=1, h=0.3,
                      convergence={'eigenstates': 1e-15},
                      xc='X-C_PW',
                      poissonsolver=PoisonSolver(relax='J'))
    atoms.set_calculator(calc)
    e = atoms.get_potential_energy()
    calc.write('Be.nc', 'all')
else:
    calc = Calculator('Be.nc')

#print 'Ground state found.'

rhot0 = calc.density.rhot_g.copy()
dm0 = calc.finegd.calculate_dipole_moment(rhot0)
norm0 = calc.finegd.integrate(rhot0)

#print 'TDDFT'
td_atoms = TDDFT(calc, propagator='SICN',tolerance=1e-12)
#print '-> kick'
td_atoms.absorption_kick(strength=1.0e-4)
calc.density.charge_eps = 1e-6

time_step = 0.05

if rank == 0:
    print ( '%16.6le  %16.6le %16.6le %16.6le' %
            (norm0, dm0[0], dm0[1], dm0[2]) )

#print '-> iteration'

if rank == 0:
    print ( '%8s  %16s  %16s'
            % ('time  ', 'Error of norm', 'Dipole moment') )

for i in range(100):
    time = i*time_step

    rhot = calc.density.rhot_g
    dm = calc.finegd.calculate_dipole_moment(rhot) - dm0
    norm = calc.finegd.integrate(rhot) - norm0
    if rank == 0:
        print ( '%8lf  %16.6le  %16.6le %16.6le %16.6le' %
                (time, norm, dm[0], dm[1], dm[2]) )
        sys.stdout.flush()
    #print i+1
    td_atoms.propagate(time_step)
