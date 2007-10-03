#!/usr/bin/env python
import os
from gpaw.utilities import equal
from gpaw.utilities.singleatom import SingleAtom
from gpaw.utilities.molecule import Molecule, molecules
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths

formula = 'Be2'

# Generate setup
symbol = 'Be'
g = Generator(symbol, 'PBE', scalarrel=True, nofiles=True)
g.run(exx=True, **parameters[symbol])
setup_paths.insert(0, '.')

tolerance = 0.00003 # libxc must reproduce old gpaw energies
# zero Kelvin: in Hartree
reference_1147 = { # version 1147
    'energy': -792.761134812,
    'bands': [-6.03947247, -2.42546926]#, 0.79480484] # unocc. st. is random
    }

# setup gpaw calculation
kwargs = {'a': 5.9,  # size of unit cell along x-axis
          'b': 4.8,  # size of unit cell along y-axis
          'c': 5.0,  # size of unit cell along z-axis
          'h': 0.21, # grid spacing
          'forcesymm': False,
          'parameters': {'xc'         : 'PBE0',
                         'txt'        : '-',
                         'mix'        : (0.25, 3, 1.0),
                         'lmax'       : 1,
                         'nbands'     : 3,
                         'convergence': {'eigenstates': 1e-6}}}
if formula in molecules:
    mol = Molecule(formula, **kwargs)
    E = mol.energy()
    bands = mol.atoms.GetCalculator().GetEigenvalues()
    setupfile = mol.atoms.GetCalculator().nuclei[0].setup.filename
else:
    atom = SingleAtom(formula, **kwargs)
    E = atom.energy()
    bands = atom.atom.GetCalculator().GetEigenvalues()
    setupfile = atom.atom.GetCalculator().nuclei[0].setup.filename

# Remove setup
os.remove(setupfile)
del setup_paths[0]

print E
equal(E, reference_1147['energy'], tolerance)
assert len(reference_1147['bands']) <= len(bands)
print bands
for i in range(len(reference_1147['bands'])):
    equal(bands[i], reference_1147['bands'][i], tolerance)
