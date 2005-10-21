from ASE import Atom, ListOfAtoms
from ASE.Calculators.CheckForce import CheckForce
from gridpaw.utilities import equal
from gridpaw import Calculator

a = 4.0
n = 16
atoms = ListOfAtoms([Atom('H', [1.234, 2.345, 3.456])],
                    cell=(a, a, a), periodic=True)
calc = Calculator(nbands=1, gpts=(n, n, n), out=None,
                  tolerance=1e-13)
atoms.SetCalculator(calc)
for i in range(3):
    f1, f2 = CheckForce(atoms, 0, i, 0.001)
    equal(f1, f2, 0.0029)

