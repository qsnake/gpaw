import os
from math import pi, cos, sin
from ase import *
from ase.parallel import rank, barrier
from gpaw import GPAW, FermiDirac
from gpaw.test import equal, gen
from gpaw.xas import XAS, RecursionMethod

# Generate setup for oxygen with half a core-hole:
gen('Si', name='hch1s', corehole=(1, 0, 0.5))

a = 4.0
b = a / 2
c = b / 2
d = b + c
si = Atoms([Atom('Si', (0, 0, 0)),
            Atom('Si', (c, c, c)),
            Atom('Si', (b, b, 0)),
            Atom('Si', (d, d, c)),
            Atom('Si', (b, 0, b)),
            Atom('Si', (d, c, d)),
            Atom('Si', (0, b, b)),
            Atom('Si', (c, d, d))],
           cell=(a, a, a), pbc=True)

import numpy as np
calc = GPAW(nbands=None,
            h=0.25,
            occupations=FermiDirac(width=0.05),
            setups={0: 'hch1s'},
            usesymm=True)
si.set_calculator(calc)
e = si.get_potential_energy()
niter = calc.get_number_of_iterations()
calc.write('si.gpw')

# restart from file
calc = GPAW('si.gpw')

import gpaw.mpi as mpi
if mpi.size == 1:
    xas = XAS(calc)
    x, y = xas.get_spectra()
else:
    x = np.linspace(0, 10, 50)

k = 2
calc.set(kpts=(k, k, k))
calc.initialize()
calc.set_positions(si)
assert calc.wfs.dtype == complex

r = RecursionMethod(calc)
r.run(40)
if mpi.size == 1:
    z = r.get_spectra(x)

if 0:
    import pylab as p
    p.plot(x, y[0])
    p.plot(x, sum(y))
    p.plot(x, z[0])
    p.show()

print e, niter
energy_tolerance = 0.0003
niter_tolerance = 0
equal(e, 18.5772, energy_tolerance) # svnversion 5252
equal(niter, 23, niter_tolerance) # svnversion 5252
