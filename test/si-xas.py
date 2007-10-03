import os
from math import pi, cos, sin
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import center
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
from gpaw.xas import XAS, RecursionMethod

if 1:
    # Generate setup for oxygen with half a core-hole:
    g = Generator('Si', scalarrel=True, corehole=(1, 0, 0.5), nofiles=True)
    g.run(name='hch1s', **parameters['Si'])
setup_paths.insert(0, '.')

a = 4.0
b = a / 2
c = b / 2
d = b + c
si = ListOfAtoms([Atom('Si', (0, 0, 0)),
                  Atom('Si', (c, c, c)),
                  Atom('Si', (b, b, 0)),
                  Atom('Si', (d, d, c)),
                  Atom('Si', (b, 0, b)),
                  Atom('Si', (d, c, d)),
                  Atom('Si', (0, b, b)),
                  Atom('Si', (c, d, d))],
                 cell=(a, a, a), periodic=True)

if 1:
    k = 2
    import Numeric as num
    calc = Calculator(nbands=None, h=0.25, kpts=(k, k, k),
                      width=0.05,
                      setups={0: 'hch1s'}, usesymm=True
                      )
    si.SetCalculator(calc)
    e = si.GetPotentialEnergy()
    calc.write('si.gpw')
else:
    calc = Calculator('si.gpw')

xas = XAS(calc)
x, y = xas.get_spectra()
calc.set_positions()
r = RecursionMethod(calc)
r.run(40)

if 0:
    import pylab as p
    p.plot(x, y[0])
    p.plot(x, sum(y))
    p.plot(x, r.get_spectra(x)[0])
    p.show()

os.system('rm si.gpw')
# remove Si.hch1s.* setup
os.remove(calc.nuclei[0].setup.filename)
