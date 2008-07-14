import numpy as npy
from ase import *

from gpaw.cluster import Cluster
from gpaw.point_charges import PointCharges

# I/O, translate

# old Cmdft style
fCmdft='PC_info'
f = open(fCmdft, 'w')
print >> f, """0.4
-3.0  0 0 -4
1.5   0 0 4.
"""
f.close()

pc = PointCharges(fCmdft)
assert(len(pc) == 2)
assert(pc.charge() == -1.5)

fxyz='pc.xyz'
pc.write(fxyz)

shift = npy.array([0.2, 0.6, 2.8])
pc.translate(shift)

pc2 = PointCharges(fxyz)
for p1, p2 in zip(pc, pc2):
    assert(npy.sum(p1.position - shift - p2.position) < 1e-6)
