import os
from ase import *
from gpaw import Calculator
from gpaw.utilities import equal
h = 0.2
n = 24
a = n * h
b = a / 2
H = Atoms('H', [(b, b, b)], pbc=True, cell=(a, a, a))
calc = Calculator(nbands=1, gpts=(n, n, n))
H.set_calculator(calc)
e0 = H.get_potential_energy()
cmd = 'ps -eo comm,pmem | grep python'
mem0 = float(os.popen(cmd).readlines()[-1].split()[-1])
for i in range(50):
    e = H.get_potential_energy()
    H.positions += (0.0123456789, 0.023456789, 0.03456789)
mem = float(os.popen(cmd).readlines()[-1].split()[-1])
equal(e, e0, 0.0015)
print mem, mem0
assert mem < mem0 + 0.21
