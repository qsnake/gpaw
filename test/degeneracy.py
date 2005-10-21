from gridpaw import Calculator
from gridpaw.utilities import equal
from ASE import ListOfAtoms, Atom


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
atoms.SetCalculator(Calculator(h=0.25, nbands=4, out=None))
energy = atoms.GetPotentialEnergy()

# The three eigenvalues e[1], e[2], and e[3] must be degenerate:
e = atoms.GetCalculator().paw.wf.kpts[0].eps_n
equal(e[1], e[3], 1e-11)
