from ase import *
from gpaw import *
a = 2.7
bulk = Atoms([Atom('Li')], pbc=True, cell=(a, a, a))
k = 2
g = 8
calc = Calculator(gpts=(g, g, g), kpts=(k, k, k), nbands=2,
                  mixer=Mixer(nmaxold=5, metric='new'))
bulk.set_calculator(calc)
bulk.get_potential_energy()
calc.write('Li.gpw')
calc2 = Calculator('Li.gpw')
