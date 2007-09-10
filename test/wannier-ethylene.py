import os
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import equal, center
from gpaw.wannier import Wannier
import Numeric as num

# GPAW wannier example for ethylene corresponding to the ASE Wannier
# tutorial.

if 1:
    a = 6.0  # Size of unit cell (Angstrom)

    ethylene = ListOfAtoms([
                       Atom('H', (-1.235,-0.936 , 0 )),
                       Atom('H', (-1.235, 0.936 , 0 )),
                       Atom('C', (-0.660, 0.000 , 0 )),
                       Atom('C', ( 0.660, 0.000 , 0 )),
                       Atom('H', ( 1.235,-0.936 , 0 )),
                       Atom('H', ( 1.235, 0.936 , 0 ))],
                       cell=(a, a, a), periodic=True)
    center(ethylene)
    calc = Calculator(nbands=8, h=0.20, tolerance=1e-6)
    ethylene.SetCalculator(calc)
    ethylene.GetPotentialEnergy()
    calc.write('ethylene.gpw', 'all')
else:
    calc = Calculator('ethylene.gpw', txt=None)

wannier = Wannier(numberofwannier=6,
                  calculator=calc,
                  numberoffixedstates=[6])
wannier.Localize(tolerance=1e-5)

centers = wannier.GetCenters()
print centers
expected = [[1.950, 2.376, 3.000],
            [1.950, 3.624, 3.000],
            [3.000, 3.000, 2.671],
            [3.000, 3.000, 3.329],
            [4.050, 2.376, 3.000],
            [4.050, 3.624, 3.000]]
equal(13.7995, wannier.GetFunctionalValue(), 0.016)
for xi, wi in enumerate(wannier.GetSortedIndices()):
    assert abs(num.sum(expected[xi] - centers[wi]['pos'])) < 0.01

os.remove('ethylene.gpw')
## from ASE.Visualization.PrimiPlotter import PrimiPlotter, X11Window
## ethylene.extend(wannier.GetCentersAsAtoms())
## plot = PrimiPlotter(ethylene)
## plot.SetOutput(X11Window())
## plot.SetRadii(.2)
## plot.SetRotation([15, 0, 0])
## plot.Plot()
