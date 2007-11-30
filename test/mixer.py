from ASE import Atom, ListOfAtoms
from gpaw import *
a = 2.7
bulk = ListOfAtoms([Atom('Li')], periodic=True, cell=(a, a, a))
k = 2
g = 8
calc = Calculator(gpts=(g, g, g), kpts=(k, k, k), nbands=2,
                  mixer=Mixer(nmaxold=5, metric='new'))
bulk.SetCalculator(calc)
bulk.GetPotentialEnergy()
calc.write('Li.gpw')
calc2 = Calculator('Li.gpw')
