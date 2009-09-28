import os
from ase import *
from gpaw import GPAW
from gpaw.vdw import FFTVDWFunctional

vdw = FFTVDWFunctional(verbose=1)
L = 2.5
a = Atoms('H', cell=(L, L, L), pbc=True, calculator=GPAW(nbands=1))
e = a.get_potential_energy()
e2 = a.calc.get_xc_difference(vdw)
a.calc.set(gpts=(12, 12, 12))
e = a.get_potential_energy()
e2 = a.calc.get_xc_difference(vdw)
assert (vdw.shape == (24, 24, 24)).all()
