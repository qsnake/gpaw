#!/usr/bin/env python
import os
from gpaw import Calculator
from gpaw.utilities import equal, center
from gpaw.atom.generator import Generator, parameters
from ASE import ListOfAtoms, Atom
from gpaw import setup_paths

loa = ListOfAtoms([Atom('Be', (0, 0, 0)), Atom('Be', (2.45, 0, 0))],
                  cell= [5.9, 4.8, 5.0], periodic=False)
center(loa)
calc = Calculator(h = .21, nbands=3, convergence={'eigenstates': 1e-6})
loa.SetCalculator(calc)

setup_paths.insert(0, '.') # Use setups from this directory
tolerance = 0.0003 # must reproduce old gpaw results

# zero Kelvin: in eV
reference_1170 = {'PBE': {'energy': -792.761113956,
                          'bands': [-6.03947075, -2.425467, 3.34091088]},
                  'oldPBE': {'energy': -792.761134812,
                             'bands': [-6.03947247, -2.42546926, 3.24562689]},
                  'LDA' : {'energy': -784.058008392,
                           'bands': [-7.01715742, -1.62133354, 3.13046582]},
                  }

reference_1212 = {'PBE': {'energy': -792.760534925,
                          'bands': [-6.0389239, -2.42516464, 3.34121398]},
                  'oldPBE': {'energy': -792.76055639,
                             'bands': [-6.03892584, -2.4251677, 0.75559029]},
                  'LDA' : {'energy': -784.058614283,
                           'bands': [-7.01716767, -1.62099917, 5.0301319]},
                  }

# translation between setup name and hybrid functional name
setup2xc = {'PBE': 'PBE0',
            'oldPBE': 'oldPBE0',
            'LDA': 'EXX'}

for setup in ['PBE', 'oldPBE', 'LDA']:
    # Generate setup
    g = Generator('Be', setup, scalarrel=True, nofiles=True)
    g.run(exx=True, **parameters['Be'])

    # setup gpaw calculation
    calc.set(xc=setup2xc[setup])
    E = calc.GetPotentialEnergy()
    bands = calc.GetEigenvalues()
    setupfile = calc.nuclei[0].setup.filename

    # Remove setup
    os.remove(setupfile)

    print setup, setup2xc[setup], E
    print bands

    equal(E, reference_1212[setup]['energy'], tolerance)
    for i in range(2): # not 3 as unoccupied eigenvalues are random!?? XXX
        equal(bands[i], reference_1212[setup]['bands'][i], tolerance)

del setup_paths[0]
