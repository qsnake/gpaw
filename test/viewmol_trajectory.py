import os

from ASE import Atom
from ASE.Dynamics.ConjugateGradient import ConjugateGradient

from gpaw import *
from gpaw.cluster import Cluster
from gpaw.utilities.viewmol import ViewmolTrajectory

s = Cluster([Atom('H'), Atom('H',(0,0,3))])
s.minimal_box(2)
c = Calculator(h=0.3, nbands=2)
s.SetCalculator(c)

fname='traj.vmol'
vmt = ViewmolTrajectory(s, fname)
c.attach(vmt.add,100000)
c.calculate()

# Find the theoretical bond length:
relax = ConjugateGradient(s, fmax=0.05)
relax.Converge()

os.remove(fname)
