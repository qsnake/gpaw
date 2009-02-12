import os
from ase import *
from gpaw import GPAW
from gpaw.utilities import equal
h = 0.2
n = 24
a = n * h
b = a / 2
H = Atoms('H', [(b - 0.1, b, b)], pbc=True, cell=(a, a, a))
calc = GPAW(nbands=1, gpts=(n, n, n), txt='ltt.txt')
H.set_calculator(calc)
e0 = H.get_potential_energy()
cmd = 'ps -eo comm,pmem | grep python'
mem0 = float(os.popen(cmd).readlines()[-1].split()[-1])
for i in range(50):
    e = H.get_potential_energy()
    H.positions += (0.09123456789, 0.0423456789, 0.03456789)
mem = float(os.popen(cmd).readlines()[-1].split()[-1])
equal(e, e0, 0.0005)
print e, e0, e-e0
print mem, mem0
assert mem < mem0 + 0.21
