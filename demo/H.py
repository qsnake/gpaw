#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator

a = 5.0
H = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=0)],
                periodic=0,
                cell=(a, a, a))
calc = Calculator(nbands=1, h=0.2, onohirose=1, tolerance=0.001, softgauss=0)
H.SetCalculator(calc)
if 0:
    import profile
    profile.run('H.GetPotentialEnergy()', 'H.prof')
else:
    H.GetPotentialEnergy()
