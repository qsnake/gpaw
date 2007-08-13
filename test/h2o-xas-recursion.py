import os
from math import pi, cos, sin
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import center
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
import Numeric as num

if 0:
    # Generate setup for oxygen with half a core-hole:
    g = Generator('O', scalarrel=True, corehole=(1, 0, 0.5), nofiles=True)
    g.run(name='hch1s', **parameters['O'])

setup_paths.insert(0, '.')

if 0:
    a = 5.0
    d = 0.9575
    t = pi / 180 * 104.51
    H2O = ListOfAtoms([Atom('O', (0, 0, 0)),
                       Atom('H', (d, 0, 0)),
                       Atom('H', (d * cos(t), d * sin(t), 0))],
                      cell=(a, a, a), periodic=False)
    center(H2O)
    calc = Calculator(nbands=10, h=0.2, setups={'O': 'hch1s'})
    H2O.SetCalculator(calc)
    e = H2O.GetPotentialEnergy()
    calc.write('h2o.gpw')
else:
    calc = Calculator('h2o.gpw')
    calc.set_positions()

from gpaw.xas import RecursionMethod

if 0:
    r = RecursionMethod(calc)
    r.run(400)
    r.write('h2o.pckl')
else:
    r = RecursionMethod(filename='h2o.pckl')

from pylab import *

x = -30 + 40 * num.arange(300) / 300.0

for n in range(50, 401, 50):
    y = r.get_spectra(x, imax=n)
    plot(x, y[0], label=str(n))
legend()
show()
