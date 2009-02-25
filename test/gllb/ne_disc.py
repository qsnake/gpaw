import os
from math import pi, cos, sin
from ase import *
from ase.parallel import rank, barrier
from gpaw import GPAW
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
import numpy as np

atom = 'Ne'

g = Generator(atom, xcname ='GLLB', scalarrel=False,nofiles=True)
g.run(**parameters[atom])
eps = g.e_j[-1]

setup_paths.insert(0, '.')

a = 14
He = Atoms([Atom(atom, (0, 0, 0))],
                cell=(a, a, a), pbc=False)
He.center()
calc = GPAW(nbands=10, h=0.20, xc='GLLB')
He.set_calculator(calc)
e = He.get_potential_energy()
response = calc.hamiltonian.xc.xcfunc.xc.xcs['RESPONSE']
dxc = response.calculate_delta_xc_perturbation()
equals(dxc, 27.71, 1e-2)
# Hardness of Ne 24.71eV by GLLB+Dxc, experimental I-A = I = 21.56eV

eps3d = calc.wfs.kpt_u[0].eps_n[-1]
equal(eps, eps3d, 1e-3)
