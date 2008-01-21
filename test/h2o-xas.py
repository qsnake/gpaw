import os
from math import pi, cos, sin
from ase import *
from gpaw import Calculator
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
from gpaw.xas import XAS

# Generate setup for oxygen with half a core-hole:
g = Generator('O', scalarrel=True, corehole=(1, 0, 0.5), nofiles=True)
g.run(name='hch1s', **parameters['O'])
setup_paths.insert(0, '.')

a = 5.0
d = 0.9575
t = pi / 180 * 104.51
H2O = Atoms([Atom('O', (0, 0, 0)),
             Atom('H', (d, 0, 0)),
             Atom('H', (d * cos(t), d * sin(t), 0))],
            cell=(a, a, a), pbc=False)
H2O.center()
calc = Calculator(nbands=10, h=0.2, setups={'O': 'hch1s'})
H2O.set_calculator(calc)
e = H2O.get_potential_energy()

xas = XAS(calc)
x, y = xas.get_spectra()
e1_n = xas.eps_n

calc.write('h2o-xas.gpw')

calc = Calculator('h2o-xas.gpw', txt=None)
xas = XAS(calc)
x, y = xas.get_spectra()
e2_n = xas.eps_n
w_n = sum(xas.sigma_cn.real**2, axis=0)
de1 = e1_n[1] - e1_n[0]
de2 = e2_n[1] - e2_n[0]

assert de1 == de2
print de1 - 2.0801
assert abs(de1 - 2.0801) < 0.001
print w_n[1] / w_n[0]
assert abs(w_n[1] / w_n[0] - 2.19) < 0.01

if 0:
    import pylab as p
    p.plot(x, y[0])
    p.plot(x, sum(y))
    p.show()

os.remove('h2o-xas.gpw')
# remove O.hch1s.* setup
os.remove(calc.nuclei[0].setup.filename)
del setup_paths[0]
