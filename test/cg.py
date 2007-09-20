from gpaw import Calculator
from gpaw.utilities import equal
from ASE import ListOfAtoms, Atom, Crystal

a = 4.05
d = a / 2**0.5
bulk = Crystal([Atom('Al', (0, 0, 0)),
                Atom('Al', (0.5, 0.5, 0.5))])
bulk.SetUnitCell((d, d, a))
h = 0.25
calc = Calculator(h=h,
                  nbands=2*8,
                  kpts=(2, 2, 2),
                  tolerance=1e-10)
bulk.SetCalculator(calc)
e0 = bulk.GetPotentialEnergy()
calc = Calculator(h=h,
                  nbands=2*8,
                  kpts=(2, 2, 2),
                  tolerance=1e-10,
                  eigensolver='cg')
bulk.SetCalculator(calc)
e1 = bulk.GetPotentialEnergy()
equal(e0, e1, 3.6e-5)
