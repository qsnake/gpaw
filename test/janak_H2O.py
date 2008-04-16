#!/usr/bin/env python
from ase import *
from gpaw import Calculator
from gpaw.mpi import run
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

# Generate setups for oxygen and hydrogen:
g = Generator('O', xcname='GLLBLDARCR', scalarrel=True, nofiles=True)
g.run(**parameters['O'])
g = Generator('H', xcname='GLLBLDARCR', scalarrel=True, nofiles=True)
g.run(**parameters['H'])
setup_paths.insert(0, '.')

# Then, calculate the core-eigenvalue of H2O molecule

a = 5.0
d = 0.9575
t = pi / 180 * 104.51
H2O = Atoms([Atom('O', (0, 0, 0)),
             Atom('H', (d, 0, 0)),
             Atom('H', (d * cos(t), d * sin(t), 0))],
            cell=(a, a, a), pbc=False)
H2O.center()
calc = Calculator(nbands=10, h=0.2, xc='GLLBLDARCR')
H2O.set_calculator(calc)
e = H2O.get_potential_energy()

Ec = calc.nuclei[0].coreref_k[0]
turboeig = -18.60729
print "Calculated core eigenvalue              :", Ec, " Ha."
print "Reference from TurboMole calculation    :", turboeig, " Ha:"
print "Difference                              :", Ec-turboeig, "Ha:"
# Eigenvalue from quick TurboMole calculation
# Basis sets used: o (13s8p)[8s5p], h QZVPP
# TODO: Find the true reference value of something to really test core eigenvalues

# This is just for testing purposes
# Using more accurate calculation of
# a = 9.0 and h=0.15, eigenvalue of -18.60716, which makes diff 0.0001270 Ha.

assert ( abs(turboeig - calc.nuclei[0].coreref_k[0]) < 0.04 )
#assert ( abs(turboeig - calc.nuclei[0].coreref_k[0]) < 2e-4 )
