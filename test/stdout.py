# This test takes approximately 4.4 seconds
import sys
import StringIO
stdout, sys.stdout = sys.stdout, StringIO.StringIO()
stderr, sys.stderr = sys.stderr, StringIO.StringIO()
from gpaw import Calculator
from ASE import ListOfAtoms, Atom

a = 5.0
h = 0.2
hydrogen = ListOfAtoms([Atom('H', (a / 2, a / 2, a / 2), magmom=1)],
                       cell=(a, a, a))

calc = Calculator(h=h, nbands=1, kpts=(1, 1, 1), width=1e-9, spinpol=True,
                  out=None)
hydrogen.SetCalculator(calc)
f = hydrogen.GetCartesianForces()

stdout, sys.stdout = sys.stdout, stdout
sys.stdout.write(stdout.getvalue())
assert len(stdout.getvalue()) == 0, 'stdout is not silent!'

stderr, sys.stderr = sys.stderr, stderr
sys.stderr.write(stderr.getvalue())
assert len(stderr.getvalue()) == 0, 'stderr is not silent!'
