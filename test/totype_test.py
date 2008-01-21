#!/usr/bin/env python
from ase import *
from gpaw import Calculator
import numpy as npy
import math
import sys
import os
import gpaw.io.array as ioarray

from gpaw.mpi import rank

a = 7.0
atoms = Atoms([Atom('Be',(a/2, a/2, a/2), magmom=0)],
                     pbc=False,
                     cell=(a, a, a))
calc = Calculator(nbands=1, h=0.3, convergence={'eigenstates': 1e-10},xc='X-C_PW',poissonsolver='J')
atoms.set_calculator(calc)
e = atoms.get_potential_energy()
calc.write('Be.nc', 'all')

calc = Calculator('Be.nc')
calc.totype(complex)