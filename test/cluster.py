from ase import *
#from gpaw.utilities.vector import Vector3d
from gpaw.cluster import Cluster
from gpaw.utilities import equal

R = 2.0
CO = Atoms([Atom('C', (1, 0, 0)), Atom('O', (1, 0, R))])

CO.rotate('y', pi/2)
equal(CO.positions[1, 0], R, 1e-10)

# translate
CO.translate(-CO.get_center_of_mass())
p = CO.positions.copy()
for i in range(2):
    equal(p[i, 1], 0, 1e-10)
    equal(p[i, 2], 0, 1e-10)

# rotate the nuclear axis to the direction (1,1,1)
CO.rotate(p[1] - p[0], (1, 1, 1))
q = CO.positions.copy()
for c in range(3):
    equal(q[0, c], p[0, 0] / sqrt(3), 1e-10)
    equal(q[1, c], p[1, 0] / sqrt(3), 1e-10)

# minimal box
b=4.0
CO = Cluster([Atom('C', (1, 0, 0)), Atom('O', (1, 0, R))])
CO.minimal_box(b)
cc = CO.get_cell() 
for c in range(3):
    width = 2*b
    if c==2:
        width += R
    equal(cc[c, c], width, 1e-10)

# I/O
fxyz='CO.xyz'
fpdb='CO.pdb'

CO.write(fxyz)
CO_b = Cluster(filename=fxyz)
assert(len(CO) == len(CO_b)) 
 
CO.write(fxyz, repeat=[1,1,1])
CO_b = Cluster(filename=fxyz)
assert(8*len(CO) == len(CO_b)) 
 
CO.write(fpdb)

os.remove(fpdb)
os.remove(fxyz)
