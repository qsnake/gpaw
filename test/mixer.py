from ase import *
from gpaw import *
a = 2.7
bulk = Atoms([Atom('Li')], pbc=True, cell=(a, a, a))
k = 2
g = 16
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2,
                  mixer=Mixer(nmaxold=5))
bulk.set_calculator(calc)
bulk.get_potential_energy()
calc.write('Li.gpw')
calc2 = GPAW('Li.gpw')
