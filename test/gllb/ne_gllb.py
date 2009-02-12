import os
from math import pi, cos, sin
from ase import *
from ase.parallel import rank, barrier
from gpaw import GPAW
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
import numpy as np

atom = 'Ne'

# Generate setup for oxygen with half a core-hole:
g = Generator(atom, xcname ='GLLB', scalarrel=False,nofiles=True)
g.run(**parameters[atom])
eps = g.e_j[-1]

setup_paths.insert(0, '.')

a = 10
He = Atoms([Atom(atom, (0, 0, 0))],
                cell=(a, a, a), pbc=False)
He.center()
calc = GPAW(nbands=4, h=0.15, xc='GLLB')
He.set_calculator(calc)
e = He.get_potential_energy()

eps3d = calc.wfs.kpt_u[0].eps_n[-1]
equal(eps, eps3d, 1e-3)
