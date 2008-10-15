import os
from math import pi, cos, sin
from ase import *
from ase.parallel import rank, barrier
from gpaw import Calculator
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
from gpaw.xas import XAS, RecursionMethod

if rank == 0:
    # Generate setup for oxygen with half a core-hole:
    g = Generator('Si', scalarrel=True, corehole=(1, 0, 0.5), nofiles=True)
    g.run(name='hch1s', **parameters['Si'])
barrier()
setup_paths.insert(0, '.')

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

k = 2
import numpy as npy
calc = Calculator(nbands=None, h=0.25, 
                  width=0.05,
                  setups={0: 'hch1s'}, usesymm=True
                  )
si.set_calculator(calc)
e = si.get_potential_energy()
calc.write('si.gpw')

# restart from file
calc = Calculator('si.gpw', kpts=(k, k, k))

assert calc.dtype == complex

import gpaw.mpi as mpi
if mpi.size == 1:
    xas = XAS(calc)
    x, y = xas.get_spectra()
else:
    x = np.linspace(0, 10, 50)
    
calc.set_positions(si)
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
