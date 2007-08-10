from gpaw import Calculator
from ASE import Crystal, Atom
from ASE.Calculators.CheckForce import CheckForce
from gpaw.utilities import equal

a = 5.404
bulk = Crystal([Atom('Si', (0, 0, 0.1 / a)),
                Atom('Si', (0, 0.5, 0.5)),
                Atom('Si', (0.5, 0, 0.5)),
                Atom('Si', (0.5, 0.5, 0)),
                Atom('Si', (0.25, 0.25, 0.25)),
                Atom('Si', (0.25, 0.75, 0.75)),
                Atom('Si', (0.75, 0.25, 0.75)),
                Atom('Si', (0.75, 0.75, 0.25))])
bulk.SetUnitCell((a, a, a))
n = 20
calc = Calculator(gpts=(n, n, n),
                  nbands=8*3,
                  width=0.01,
                  kpts=(2, 2, 2))
bulk.SetCalculator(calc)
f1, f2 = CheckForce(bulk, 0, 2, 0.001)
equal(f1, f2, 0.012)
