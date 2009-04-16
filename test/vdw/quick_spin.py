import os
from ase import *
from gpaw import GPAW

if 'GPAW_VDW' in os.environ:
    L = 2.5
    a = Atoms('H', cell=(L, L, L), pbc=True, calculator=GPAW(nbands=1,
                                                             xc='vdW-DF',
                                                             spinpol=True))
    e = a.get_potential_energy()

