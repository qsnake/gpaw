#!/usr/bin/env python
from gpaw import Calculator
from gpaw.testing.g2 import get_g2, atoms
from gpaw.testing.atomization_data import atomization_vasp
from ASE.Utilities.Parallel import paropen

cell = [12., 13., 14.]
calc = Calculator(h=.18, xc='PBE', txt='g2test.txt',
                  convergence={'energy': 1e-3,
                               'density': 1e-3,
                               'eigenstates': 1e-9,
                               'bands': 'occupied'})

data = paropen('g2data.txt', 'w')
for formula in atoms.keys() + atomization_vasp.keys():
    loa = get_g2(formula, cell)
    loa.SetCalculator(calc)
    energy = loa.GetPotentialEnergy()
    print >>data, formula, repr(energy)
