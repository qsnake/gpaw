import os
from ase import *
from gpaw import Calculator
from gpaw.utilities import equal
from gpaw.wannier import Wannier
import numpy as npy

# GPAW wannier example for ethylene corresponding to the ASE Wannier
# tutorial.

a = 6.0  # Size of unit cell (Angstrom)

ethylene = Atoms([Atom('H', (-1.235,-0.936 , 0 )),
                  Atom('H', (-1.235, 0.936 , 0 )),
                  Atom('C', (-0.660, 0.000 , 0 )),
                  Atom('C', ( 0.660, 0.000 , 0 )),
                  Atom('H', ( 1.235,-0.936 , 0 )),
                  Atom('H', ( 1.235, 0.936 , 0 ))],
                 cell=(a, a, a), pbc=True)
ethylene.center()

if not os.path.isfile('ethylene.gpw'):
    calc = Calculator(nbands=6, h=0.20, convergence={'eigenstates': 1e-6})
    ethylene.set_calculator(calc)
    ethylene.get_potential_energy()
    calc.write('ethylene.gpw', 'all')
else:
    calc = Calculator('ethylene.gpw', txt=None)

wannier = Wannier(calc)
wannier.localize()

centers = wannier.get_centers()
print centers
expected = [[1.950, 2.376, 3.000],
            [1.950, 3.624, 3.000],
            [3.000, 3.000, 2.671],
            [3.000, 3.000, 3.329],
            [4.050, 2.376, 3.000],
            [4.050, 3.624, 3.000]]
equal(13.7995, wannier.value, 0.016)
for center in centers:
    i = 0
    while npy.sum((expected[i] - center)**2) > 0.01:
        i += 1
        if i == len(expected):
            raise RuntimeError, 'Correct center not found'
    expected.pop(i)    

os.remove('ethylene.gpw')

## for i in range(6):
##     wannier.write_cube(i, 'ethylene%s.cube' % i, real=True)

## from ASE.Visualization.PrimiPlotter import PrimiPlotter, X11Window
## ethylene.extend(wannier.get_centers_as_atoms())
## plot = PrimiPlotter(ethylene)
## plot.set_output(X11Window())
## plot.set_radii(.2)
## plot.set_rotation([15, 0, 0])
## plot.plot()
