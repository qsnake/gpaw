import os
import sys

from ase import Atoms, Atom, QuasiNewton, PickleTrajectory

from gpaw import *
from gpaw.cluster import Cluster
from gpaw.utilities.viewmol import ViewmolTrajectory, write_viewmol

s = Cluster([Atom('H'), Atom('H',(0,0,3))])
s.minimal_box(2)
c = Calculator(h=0.3, nbands=2)
s.set_calculator(c)

vfname='traj.vmol'
pfname='traj.pickle'
vmt = ViewmolTrajectory(s, vfname)
traj = PickleTrajectory(pfname, 'w', s)
#c.attach(vmt.add, 100000)
#sys.exit()

# Find the theoretical bond length:
dyn = QuasiNewton(s)
dyn.attach(traj.write)
dyn.attach(vmt.add)
dyn.run(fmax=0.05)

traj = PickleTrajectory(pfname, 'r')
vfname2='pickle.vmol'
write_viewmol(traj, vfname2)
