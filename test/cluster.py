from math import pi, sqrt
from ASE import Atom
from gpaw.cluster import Cluster
from gpaw.utilities.vector import Vector3d
from gpaw.utilities import equal

R=2.
CO = Cluster([Atom('C', (1,0,0)), Atom('O', (1,0,R))])

# rotation
x=Vector3d([1,0,0])
y=Vector3d([0,1,0])
z=Vector3d([0,0,1])
print 'x, y, z=', x, y, z

CO.rotate((pi/2) * y)
equal(CO[1].GetCartesianPosition()[0], R, 1e-10)

# translate
CO.translate(-CO.center_of_mass())
##print "CO=",CO
for i in range(2):
    pos = CO[i].GetCartesianPosition()
    equal(pos[1], 0, 1e-10)
    equal(pos[2], 0, 1e-10)

# rotate the nuclear axis to the direction (1,1,1)
xyz=Vector3d([1,1,1])
pC=Vector3d(CO[0].GetCartesianPosition())
pCx=pC.x()
pO=Vector3d(CO[1].GetCartesianPosition())
pOx=pO.x()
pCO = pO - pC
axis = pCO.rotation_axis(xyz)
## print CO, axis
CO.rotate(axis)
## print CO, axis
pC=CO[0].GetCartesianPosition()
pO=CO[1].GetCartesianPosition()
for c in range(3):
    equal(pC[c], pCx / sqrt(3), 1e-10)
    equal(pO[c], pOx / sqrt(3), 1e-10)
