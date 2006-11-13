#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator

a = 4.0
H = ListOfAtoms([Atom('H', (a/2, a/2, a/2), magmom=1)],
                periodic=0,
                cell=(a, a, a))
calc = Calculator(nbands=1, h=0.2, charge=1)
H.SetCalculator(calc)
print H.GetPotentialEnergy() + calc.GetReferenceEnergy()
assert abs(H.GetPotentialEnergy() + calc.GetReferenceEnergy()) < 0.013

