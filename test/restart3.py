import os
from gpaw import Calculator
from ase import *
from gpaw.utilities import equal


h = Atoms([Atom('H')], cell=(4, 4, 4), pbc=1)
calc = Calculator(nbands=1, gpts=(16, 16, 16), kpts = (1,1,1) )
h.set_calculator(calc)
e = h.get_potential_energy()
calc.write('tmp.gpw')


calc2 = Calculator('tmp.gpw', fixdensity = True, kpts = [ (0,0,0) ])

