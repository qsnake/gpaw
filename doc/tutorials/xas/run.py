import os
from math import pi, cos, sin
from ase import *
from ase.parallel import rank, barrier
from gpaw import GPAW

a = 12.0    # use a large cell
d = 0.9575
t = pi / 180 * 104.51
H2O = Atoms('OH2',
            [(0, 0, 0),
             (d, 0, 0),
             (d * cos(t), d * sin(t), 0)],
            cell=(a, a, a),
            pbc=False)
H2O.center()
calc = GPAW(nbands=-30,
            h=0.2,
            setups={'O': 'hch1s'})
# the number of unoccupied stated will determine how
# high you will get in energy 
H2O.set_calculator(calc)
e = H2O.get_potential_energy()

calc.write('h2o_xas.gpw')
