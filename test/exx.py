#!/usr/bin/env python
import os
from gpaw.utilities import equal
from gpaw.utilities.singleatom import SingleAtom
from gpaw.utilities.molecule import Molecule, molecules
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths

formula = 'Be2'

symbol = 'Be'
setup_paths.insert(0, '.')

tolerance = 0.00003 # must reproduce old gpaw results
# zero Kelvin: in Hartree
reference_1170 = { # version 1170
    'PBE' : { # libxc must reproduce libxc
    'energy': -792.761113956,
    'bands': [-6.03947075, -2.425467]#, 3.34091088] # unocc. st. is random
    },
    'oldPBE' : { # old gpaw must reproduce libxc
    'energy': -792.761113956,
    'bands': [-6.03947075, -2.425467]#, 3.34091088] # unocc. st. is random
    ## This are real oldPBE results of 1170 version:
    ## 'energy': -792.761134812,
    ## 'bands': [-6.03947247, -2.42546926]#, 3.24562689] # unocc. st. is random
    },
    'LDA' : { # libxc must reproduce libxc
    'energy': -784.058008392,
    'bands': [-7.01715742, -1.62133354]#, 3.13046582] # unocc. st. is random
    }
    }

# translation between setup name and hybrid functional name
setup2xc = {'PBE': 'PBE0',
            'oldPBE': 'oldPBE0',
            'LDA': 'EXX'}

for setup in ['PBE', 'oldPBE', 'LDA']:
    # Generate setup
    g = Generator(symbol, setup, scalarrel=True, nofiles=True)
    g.run(exx=True, **parameters[symbol])
    # setup gpaw calculation
    kwargs = {'a': 5.9,  # size of unit cell along x-axis
              'b': 4.8,  # size of unit cell along y-axis
              'c': 5.0,  # size of unit cell along z-axis
              'h': 0.21, # grid spacing
              'forcesymm': False,
              'parameters': {'xc'         : setup2xc[setup],
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

    print setup, setup2xc[setup], E
    equal(E, reference_1170[setup]['energy'], tolerance)
    assert len(reference_1170[setup]['bands']) <= len(bands)
    print bands
    for i in range(len(reference_1170[setup]['bands'])):
        equal(bands[i], reference_1170[setup]['bands'][i], tolerance)

del setup_paths[0]
