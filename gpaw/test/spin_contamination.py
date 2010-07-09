import os
import sys

from gpaw.test import equal
from ase import Atom

from gpaw import GPAW, MixerDif
from gpaw.cluster import Cluster

h = .25
q = 0
box = 3.
spin=True

# setup H2 at large distance with differnt spins for the atoms
s = Cluster([Atom('H'), Atom('H',[0,0,3.0])])
s.minimal_box(box, h=h)
s.set_initial_magnetic_moments([-1,1])

c = GPAW(xc='LDA', nbands=-1, 
         charge=q, spinpol=spin, h=h,
         mixer=MixerDif(beta=0.05, nmaxold=5, weight=50.0),
         convergence={'eigenstates':1e-3, 'density':1e-2, 'energy':0.1},
         )
c.calculate(s)

scont_s = [c.density.get_spin_contamination(s), 
           c.density.get_spin_contamination(s, 1)]
assert(scont_s[0] == scont_s[1]) # symmetry
equal(scont_s[0], 0.968624, 1.e-5)
