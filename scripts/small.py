from gpaw.utilities.singleatom import SingleAtom
from gpaw.utilities.molecule import Molecule, molecules

formula = 'H'

# setup gpaw calculation
kwargs = {'a': 5.5,  # size of unit cell along x-axis
          'b': 5.6,  # size of unit cell along y-axis
          'c': 5.7,  # size of unit cell along z-axis
          'h': 0.19, # grid spacing
          'forcesymm': False,
          'parameters': {'xc'         : 'PBE',
                         'txt'        : '-',
                         'mix'        : (0.25, 3, 1.0),
                         'lmax'       : 2,
                         'nbands'     : 2,
                         'setups'     : {'Li': 'nocore'},
                         'spinpol'    : False,
                         'stencils'   : (2, 'M', 3),
                         'convergence': {'eigenstates': 1e-9},
                         'eigensolver': 'rmm-diis'}}
if formula in molecules:
    mol = Molecule(formula, **kwargs)
    mol.energy()
else:
    atom = SingleAtom(formula, **kwargs)
    atom.energy()
