#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
import Numeric as num
import math
import sys
import os
import gpaw.io.array as ioarray

from gpaw.mpi import rank

a = 7.0
atoms = ListOfAtoms([Atom('Be',(a/2, a/2, a/2), magmom=0)],
                     periodic=False,
                     cell=(a, a, a))
calc = Calculator(nbands=1, h=0.3, convergence={'eigenstates': 1e-10},xc='X-C_PW',poissonsolver='J')
atoms.SetCalculator(calc)
e = atoms.GetPotentialEnergy()
calc.write('Be.nc', 'all')

calc = Calculator('Be.nc')
calc.totype(num.Complex)