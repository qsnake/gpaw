#!/usr/bin/env python
from gpaw import Calculator
from ase import *
import numpy as num


########################################
#This script produces the projected density of states
########################################
filename='Fe_nonmag.gpw'
calc = Calculator(filename)

energies, ldos = calc.get_orbital_ldos(a=0, spin=0, angular='s')

# Plot LDOS
import pylab
plot(energies, ldos)
show()


