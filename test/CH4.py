from gpaw import Calculator
from ASE import ListOfAtoms, Atom


a = 4.0
n = 20
d = 1.0
x = d / 3**0.5
atoms = ListOfAtoms([Atom('C', (0.0, 0.0, 0.0)),
                     Atom('H', (x, x, x)),
                     Atom('H', (-x, -x, x)),
                     Atom('H', (x, -x, -x)),
                     Atom('H', (-x, x, -x))], cell=(a, a, a), periodic=True)
atoms.SetCalculator(Calculator(gpts=(n, n, n), nbands=4, out='-'))
e0 = atoms.GetPotentialEnergy()

for d in [1.0, 1.05, 1.1, 1.15]:
    x = d / 3**0.5
    atoms[1].SetCartesianPosition((x, x, x))
    print d, atoms.GetPotentialEnergy() - e0
