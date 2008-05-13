from gpaw import Calculator
from ase import *
from ase.calculators import numeric_force
from gpaw.utilities import equal

a = 5.404
bulk = Atoms(symbols='Si8',
             positions=[(0, 0, 0.1 / a),
                        (0, 0.5, 0.5),
                        (0.5, 0, 0.5),
                        (0.5, 0.5, 0),
                        (0.25, 0.25, 0.25),
                        (0.25, 0.75, 0.75),
                        (0.75, 0.25, 0.75),
                        (0.75, 0.75, 0.25)],
             pbc=True)
bulk.set_cell((a, a, a))
n = 20
calc = Calculator(gpts=(n, n, n),
                  nbands=8*3,
                  width=0.01,
                  kpts=(2, 2, 2))
bulk.set_calculator(calc)
f1 = bulk.get_forces()[0, 2]
f2 = numeric_force(bulk, 0, 2)
print f1,f2,f1-f2
equal(f1, f2, 0.012)
