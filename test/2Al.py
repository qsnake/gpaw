from gpaw import Calculator
from ase import *
from gpaw.utilities import equal

a = 4.05
d = a / 2**0.5
bulk = Atoms([Atom('Al', (0, 0, 0)),
              Atom('Al', (0, 0, d))],
             cell=(4*d, 4*d, 2*d),
             pbc=1)
n = 16
calc = Calculator(gpts=(2*n, 2*n, 1*n),
                  nbands=1*8,
                  kpts=(1, 1, 4),
                  convergence={'eigenstates': 1e-11},xc='LDA')
bulk.set_calculator(calc)
e2 = bulk.get_potential_energy()

bulk = bulk.repeat((1, 1, 2))
bulk.set_calculator(calc)
calc.set(nbands=16, kpts=(1, 1, 2), gpts=(2*n, 2*n, 2*n))
e4 = bulk.get_potential_energy()

equal(e4 / 2, e2, 48e-6)
