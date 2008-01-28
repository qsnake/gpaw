#!/usr/bin/env python
import os
from ase import *
from gpaw import Calculator
from gpaw.utilities import equal
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

loa = Atoms([Atom('Be', (0, 0, 0)), Atom('Be', (2.45, 0, 0))])
loa.center(vacuum=2.0)
calc = Calculator(h=0.21, nbands=3, convergence={'eigenstates': 1e-6},
                  txt='Be2.txt')
loa.set_calculator(calc)

tolerance = 0.0003 # must reproduce old gpaw results
tolerance = 30.0003 # must reproduce old gpaw results

#for setup in ['LDA', 'PBE']:#, 'oldPBE', 'LDA']:
for setup in ['PBE', 'PBE0', 'EXX', 'PBE']:#, 'oldPBE', 'LDA']:
    # Generate setup
    #g = Generator('Be', setup, scalarrel=True, nofiles=True, txt=None)
    #g.run(exx=True, **parameters['Be'])

    # setup gpaw calculation
    calc.set(xc=setup)
    E = loa.get_potential_energy()
    bands = calc.get_eigenvalues()

    print setup, E
    print bands

    for i in range(2): # not 3 as unoccupied eigenvalues are random!?? XXX
        print(bands[i])
