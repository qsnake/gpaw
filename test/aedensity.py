#!/usr/bin/env python
import numpy as npy

from ase import *
from ase.parallel import rank

from gpaw import Calculator
from gpaw.utilities import equal

try:
    calc = Calculator('NaCl.gpw')
    NaCl = calc.get_atoms()
except:
    h = 0.17 # gridspacing
    a = [6.5, 6.5, 7.7] # unit cell
    d = 2.3608 # experimental bond length

    NaCl = Atoms([Atom('Na', [0, 0, 0]),
                  Atom('Cl', [0, 0, d])],
                 pbc=False, cell=a)
    NaCl.center()
    calc = Calculator(h=h, xc='LDA', nbands=5, lmax=0,
                      convergence={'eigenstates': 1e-6}, spinpol=1)

    NaCl.set_calculator(calc)
    NaCl.get_potential_energy()
    calc.write('NaCl.gpw')

nt = calc.get_pseudo_valence_density()
gridrefinement = 2 # grid-refinement-factor for all-electron density
n = calc.get_all_electron_density(gridrefinement=gridrefinement)

dv = npy.product(calc.get_grid_spacings())
Zt = npy.sum(nt.ravel()) * dv
Z = npy.sum(n.ravel()) * dv / gridrefinement**3

if rank == 0:
    print 'Integral of pseudo density:', Zt
    print 'Integral of all-electron density:', Z

    equal(Z, 28, 1e-5)
