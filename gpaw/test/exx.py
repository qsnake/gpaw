#!/usr/bin/env python
import os
from ase import Atom, Atoms
from gpaw import GPAW, PoissonSolver
from gpaw.test import equal

loa = Atoms([Atom('Be', (0, 0, 0)), Atom('Be', (2.45, 0, 0))])
loa.center(vacuum=2.0)
calc = GPAW(h=0.21, nbands=3, convergence={'eigenstates': 1e-8},
            poissonsolver=PoissonSolver(nn='M', relax='J'),
            txt='exx.txt')
            
loa.set_calculator(calc)

ref_1871 = { # Values from revision 1871. Not true reference values
    # xc         Energy          eigenvalue 0    eigenvalue 1
    'PBE' : ( 5.42745031912, -3.84092348806, -0.961920795759),
    'PBE0': (-790.919942,   -4.92321, -1.62948),
    'EXX' : (-785.521919866, -7.18798034247, -2.75482604483)
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
            equal(first, second, 2.5e-3)
    else:
        current[setup] = res

for setup in current:
    for ref, cur in zip(ref_1871[setup], current[setup]):
        print ref, cur, ref-cur
        equal(ref, cur, 2.5e-3)
