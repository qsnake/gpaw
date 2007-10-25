#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator

a = 5
b = a / 2
mol = ListOfAtoms([Atom('O',(b, b, 0.1219 + b)),
                  Atom('H',(b, 0.7633 + b, -0.4876 + b)),
                  Atom('H',(b, -0.7633 + b, -0.4876 + b))],
                  periodic=False, cell=[a, a, a])
calc = Calculator(nbands=4, h=0.2, eigensolver='lcao')
mol.SetCalculator(calc)
e = mol.GetPotentialEnergy()
