from ase import *
from gpaw import GPAW
from gpaw.test import equal

a = 4.05
d = a / 2**0.5
bulk = Atoms([Atom('Al', (0, 0, 0)),
              Atom('Al', (0.5, 0.5, 0.5))],
             pbc=True)
bulk.set_cell((d, d, a), scale_atoms=True)
h = 0.25
calc = GPAW(h=h,
                  nbands=2*8,
                  kpts=(2, 2, 2),
                  convergence={'energy': 1e-5})
bulk.set_calculator(calc)
e0 = bulk.get_potential_energy()
niter0 = calc.get_number_of_iterations()
calc = GPAW(h=h,
                  nbands=2*8,
                  kpts=(2, 2, 2),
                  convergence={'energy': 1e-5},
                  eigensolver='cg')
bulk.set_calculator(calc)
e1 = bulk.get_potential_energy()
niter1 = calc.get_number_of_iterations()
equal(e0, e1, 3.6e-5)

energy_tolerance = 0.00001
niter_tolerance = 0
equal(e0, -6.97125875119, energy_tolerance) # svnversion 5252
equal(niter0, 24, niter_tolerance) # svnversion 5252
equal(e1, -6.97126179098, energy_tolerance) # svnversion 5252
equal(niter1, 18, niter_tolerance) # svnversion 5252
