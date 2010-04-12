from ase import *
from gpaw import *
from gpaw.transport.stm import STM

a = 7.0
d = 3.31
tip = Atoms('Na5', pbc=1, cell=(a, a, a / 2 + 4 * d))
tip.positions[:, 2] = np.arange(5) * d + a / 2
tip.positions[:, :2] = a / 2
calc = GPAW(mode='lcao', basis='sz', width=0.1)
tip.set_calculator(calc)
tip.get_potential_energy()
calc.write('tip')

tip = GPAW('tip')
srf = GPAW('tip')
stm = STM(tip, srf)
stm.initialize(0, dmin=4.0)

from gpaw.spline import Spline
import numpy as np
a = np.array([1, 0.9, 0.1, 0.0])
s = Spline(0, 2.0, a)
p = Spline(1, 2.0, a)
from gpaw.transport.stm import AtomCenteredFunctions as ACF
from gpaw.grid_descriptor import GridDescriptor
a = 10.0
N = 32
gd = GridDescriptor((N, N, N), (a, a, a), (1, 1, 0))

a = ACF(gd, [s], np.array((0.5, 0.5, 0.5)))
b = ACF(gd, [p], np.array((0.6, 0.4, 0.54)))
print a.overlap(a)
print (a|a), (a|b), (b|b)
v = gd.empty()
v[:] = 1.0
print (a|v|a), (a|v|b), (b|v|a), (b|v|b)
