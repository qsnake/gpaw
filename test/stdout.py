import sys
class Out:
    def write(self, x):
        sys.__stdout__.write(x)
        raise RuntimeError('not silent')

out, err = sys.stdout, sys.stderr
sys.stdout = sys.stderr = Out()

try:
    from gpaw import Calculator
    from ASE import ListOfAtoms, Atom

    a = 5.0
    h = 0.2
    hydrogen = ListOfAtoms([Atom('H', (a / 2, a / 2, a / 2), magmom=1)],
                           cell=(a, a, a))

    calc = Calculator(h=h, nbands=1, kpts=(1, 1, 1), width=1e-9, spinpol=True,
                      txt=None)
    hydrogen.SetCalculator(calc)
    f = hydrogen.GetCartesianForces()
except:
    sys.stdout = out
    sys.stderr = err
    raise

sys.stdout = out
sys.stderr = err
