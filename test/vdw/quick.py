import os
from ase import *
from gpaw import GPAW
from gpaw.vdw import FFTVDWFunctional

if 'GPAW_VDW' in os.environ:
    vdw = FFTVDWFunctional(verbose=1)
    L = 2.5
    a = Atoms('H', cell=(L, L, L), pbc=True, calculator=GPAW(nbands=1))
    e = a.get_potential_energy()
    e2 = a.calc.get_xc_difference(vdw)
    a.calc.set(gpts=(8, 8, 8))
    e = a.get_potential_energy()
    e2 = a.calc.get_xc_difference(vdw)
    assert (vdw.shape == (16, 16, 16)).all()
