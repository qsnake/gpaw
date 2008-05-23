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
                  txt=None)
loa.set_calculator(calc)

ref_1871 = { # Values from revision 1871. Not true reference values
    # xc         Energy          eigenvalue 0    eigenvalue 1
    'PBE' : (   5.36043449926, -3.86524565488, -0.989950689923),
    'PBE0': (-790.98523889100, -5.09079115492, -1.826444460780),
    'EXX' : (-785.59780385600, -7.21083921161, -2.782087866830),
    }

current = {} # Current revision
for setup in ['PBE', 'PBE0', 'EXX', 'PBE']:#, 'oldPBE', 'LDA']:
    # Generate setup
    #g = Generator('Be', setup, scalarrel=True, nofiles=True, txt=None)
    #g.run(exx=True, **parameters['Be'])

    # switch to new xc functional
    calc.set(xc=setup)
    E = loa.get_potential_energy()
    bands = calc.get_eigenvalues()[:2] # not 3 as unocc. eig are random!? XXX
    res = (E,) + tuple(bands)
    print setup, res
    
    if setup in current:
        for first, second in zip(current[setup], res):
            equal(first, second, 2e-3)
    else:
        current[setup] = res
        
for setup in current:
    for ref, cur in zip(ref_1871[setup], current[setup]):
        equal(ref, cur, 2e-3)
