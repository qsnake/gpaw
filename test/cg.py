from gpaw import Calculator
from gpaw.utilities import equal
from ASE import ListOfAtoms, Atom, Crystal

a = 5.0
d = 1.0
x = d / 3**0.5
atoms = ListOfAtoms([Atom('C', (0.0, 0.0, 0.0)),
                     Atom('H', (x, x, x)),
                     Atom('H', (-x, -x, x)),
                     Atom('H', (x, -x, -x)),
                     Atom('H', (-x, x, -x))],
                    cell=(a, a, a),
                    periodic=False)

atoms.SetCartesianPositions(atoms.GetCartesianPositions() + a / 2)
calc = Calculator(h=0.25, nbands=4, tolerance=1e-12)
atoms.SetCalculator(calc)
e0 = atoms.GetPotentialEnergy()
calc.Set(eigensolver="cg")
e1 = atoms.GetPotentialEnergy()
equal(e0, e1, 1e-5)
del atoms

a = 4.05
d = a / 2**0.5
bulk = Crystal([Atom('Al', (0, 0, 0)),
                Atom('Al', (0.5, 0.5, 0.5))])
bulk.SetUnitCell((d, d, a))
h = 0.25
calc = Calculator(h=h,
                  nbands=2*8,
                  kpts=(4, 4, 4),
                  tolerance=1e-12)
bulk.SetCalculator(calc)
e0 = bulk.GetPotentialEnergy()
calc.Set(eigensolver="cg")
e1 = bulk.GetPotentialEnergy()
equal(e0, e1, 1.4e-5)
