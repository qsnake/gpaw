from gpaw.utilities.singleatom import SingleAtom
from gpaw.utilities.molecule import Molecule, molecules

formula = 'H2'
a = 6.7
h = 0.19

# setup gpaw calculation
kwargs = {'a': a,    # size of unit cell
          'b': a+.1, # size of unit cell
          'c': a-.1, # size of unit cell
          'h': h,    # grid spacing
          'forcesymm': False,
          'parameters': {'xc'         : 'EXX',
                         'out'        : '-',
                         'mix'        : (0.25, 3, 1.0),
                         'lmax'       : 2,
                         'hosts'      : 1,
                         'nbands'     : 4,
                         'setups'     : {},
                         'spinpol'    : True,
                         'stencils'   : (2, 'M', 3),
                         'softgauss'  : False,
                         'tolerance'  : 1e-9,
                         'eigensolver': 'rmm-diis'}}
if formula in molecules:
    mol = Molecule(formula, **kwargs)
    mol.energy()
else:
    atom = SingleAtom(formula, **kwargs)
    atom.energy()
