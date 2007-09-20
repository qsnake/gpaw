#!/usr/bin/env python
from gpaw.utilities import equal
from gpaw.utilities.singleatom import SingleAtom
from gpaw.utilities.molecule import Molecule, molecules

formula = 'Be2'

# setup gpaw calculation
kwargs = {'a': 5.9,  # size of unit cell along x-axis
          'b': 4.8,  # size of unit cell along y-axis
          'c': 5.0,  # size of unit cell along z-axis
          'h': 0.21, # grid spacing
          'forcesymm': False,
          'parameters': {'xc'         : 'EXX',
                         'txt'        : '-',
                         'mix'        : (0.25, 3, 1.0),
                         'lmax'       : 1,
                         'nbands'     : 3,
                         'convergence': {'eigenstates': 1e-6}}}
if formula in molecules:
    mol = Molecule(formula, **kwargs)
    E = mol.energy()
else:
    atom = SingleAtom(formula, **kwargs)
    E = atom.energy()
print E
equal(E, -787.5476, 1e-4)
