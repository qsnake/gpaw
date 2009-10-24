from ase import *
from gpaw import *
from gpaw.test import equal

a = 2.7
bulk = Atoms([Atom('Li')], pbc=True, cell=(a, a, a))
k = 2
g = 16
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2,
                  mixer=Mixer(nmaxold=5))
bulk.set_calculator(calc)
e = bulk.get_potential_energy()
niter = calc.get_number_of_iterations()
calc.write('Li.gpw')
calc2 = GPAW('Li.gpw')

energy_tolerance = 0.000001
niter_tolerance = 0
equal(e, -1.20245188797, energy_tolerance) # svnversion 5252
equal(niter, 14, niter_tolerance) # svnversion 5252
