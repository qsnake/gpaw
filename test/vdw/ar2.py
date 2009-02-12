import os
from ase import *
from gpaw import GPAW, setup_paths
from gpaw.vdw import FFTVDWFunctional
from ase.parallel import rank, barrier
from gpaw.atom.generator import Generator, parameters

if rank == 0:
    g = Generator('Ar', 'revPBE', scalarrel=True, nofiles=True)
    g.run(**parameters['Ar'])
barrier()
setup_paths.insert(0, '.')

def test():
    vdw = FFTVDWFunctional(verbose=1)
    d = 3.9
    x = d / sqrt(3)
    L = 3.0 + 2 * 4.0
    dimer = Atoms('Ar2', [(0, 0, 0), (x, x, x)], cell=(L, L, L))
    dimer.center()
    calc = GPAW(h=0.2, xc='revPBE')
    dimer.set_calculator(calc)
    e2 = dimer.get_potential_energy()
    calc.write('Ar2.gpw')
    e2vdw = calc.get_xc_difference(vdw)
    e2vdwb = GPAW('Ar2.gpw').get_xc_difference(vdw)
    print e2vdw - e2vdw
    assert abs(e2vdw - e2vdw) < 1e-9
    del dimer[1]
    e = dimer.get_potential_energy()
    evdw = calc.get_xc_difference(vdw)

    E = 2 * e - e2
    Evdw = E + 2 * evdw - e2vdw
    print E, Evdw
    assert abs(E - -0.0048) < 1e-4
    assert abs(Evdw - +0.0223) < 1e-4

if 'GPAW_VDW' in os.environ:
    test()
