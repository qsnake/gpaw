import os
from math import pi, cos, sin
from ase import *
from gpaw import GPAW
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
from gpaw.mpi import world

# Generate the setups
if world.rank == 0:
    Generator('O', nofiles=True, xcname='LDA').run(**parameters['O'])
    Generator('O', nofiles=True, xcname='GLLBLDA').run(**parameters['O'])
    Generator('H', nofiles=True, xcname='LDA').run(**parameters['H'])
    Generator('H', nofiles=True, xcname='GLLBLDA').run(**parameters['H'])

setup_paths.insert(0, '.')

a = 4.0
d = 0.9575
t = pi / 180 * 104.51

system = [Atom('O', (0, 0, 0)),
          Atom('H', (d, 0, 0)),
          Atom('H', (d * cos(t), d * sin(t), 0))]

c = {'energy':      1e-6,  # eV / atom
     'density':     1.0e-7,
     'eigenstates': 1.0e-11,
     'bands':       'occupied'}


H2O = Atoms(system, cell=(a, a, a), pbc=False)
H2O.center()
calc = GPAW(xc='LDA', convergence=c)
H2O.set_calculator(calc)
e_lda = H2O.get_potential_energy()

calc = GPAW(xc='GLLBLDA', convergence=c)
H2O.set_calculator(calc)
e_gllblda = H2O.get_potential_energy()

del setup_paths[0]
assert abs(e_lda - e_gllblda)<1e-3


