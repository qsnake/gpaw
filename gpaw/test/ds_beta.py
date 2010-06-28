import os
import sys

import numpy as np

from ase import *
from ase.parallel import *

from gpaw import *
from gpaw.cluster import *
from gpaw.pes.state import BoundState, H1s
from gpaw.pes.ds_beta import CrossSectionBeta

xc = 'PBE'
ngauss=1
analytic=True
#analytic=False
form = 'L'

h=.3
box=3.

c = GPAW(xc='PBE', nbands=-1, h=h)
#s = Atoms('H')
#s.center(vacuum=box)
s = Cluster([Atom('H')])
s.minimal_box(box, h=h)
c.calculate(s)
cm = s.get_center_of_mass()

for analytic in [True, False]:
    if analytic:
        initial = H1s(c.density.gd, cm)
    else:
        initial=BoundState(c, 0, 0)
        initial.set_energy(-Ha/2.)

    csb = CrossSectionBeta(initial=initial,
                           final=None,
                           r0=cm, ngauss=ngauss, form=form)
    energy = 1.
    E = energy / Ha
    print '%5.3f' %(energy + Ha/2.),
    print '%7.4f %12.5f' %(csb.get_beta(E), 
                           csb.get_ds(E, units='Mb'))
