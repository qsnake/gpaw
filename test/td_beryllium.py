#!/usr/bin/env python
from ase import *
from gpaw import *
from gpaw.poisson import PoissonSolver
import numpy as npy
from gpaw.tddft import TDDFT
import math
import sys
import os
import gpaw.io.array as ioarray

from gpaw.mpi import rank


if ( not os.path.exists('be_gs.gpw') ):
    atoms = Atoms(symbols='Be', positions=[(0,0,0)], pbc=False)
    atoms.center(vacuum=4.0)
    calc = Calculator(nbands=1, h=0.3,
                      convergence={'eigenstates': 1e-15},
                      poissonsolver=PoissonSolver(relax='J'))
    atoms.set_calculator(calc)
    e = atoms.get_potential_energy()
    calc.write('be_gs.gpw', 'all')


time_step = 1.0 # 1 as = 0.041341 autime
iters = 1000     # 1000 x 1 as => 1 fs

td_atoms = TDDFT('be_gs.gpw')
td_atoms.absorption_kick()
td_atoms.propagate(time_step, iters, 'be_dm.dat', 'be_td.gpw')
TDDFT.photoabsorption_spectrum('be_dm.dat', 'be_spectrum.dat')
