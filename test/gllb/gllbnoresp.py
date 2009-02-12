from ase import *
from gpaw import GPAW
from gpaw.utilities import equal
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths


Generator('Ne', nofiles=True, scalarrel=False, xcname='GLLBNORESP').run(**parameters['Ne'])

setup_paths.insert(0, '.')

a = 8
hydrogen = Atoms([Atom('Ne')], cell=(a, a, a), pbc=False)
hydrogen.center()
calc = GPAW(xc = 'GLLBNORESP')
hydrogen.set_calculator(calc)
e1 = hydrogen.get_potential_energy()
