import os
from ase import *
from gpaw import GPAW
from ase.parallel import rank, barrier
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

# Generate setup
if rank == 0:
    g = Generator('H', 'revPBE', scalarrel=True, nofiles=True)
    g.run(**parameters['H'])
barrier()
setup_paths.insert(0, '.')

if 'GPAW_VDW' in os.environ:
    L = 2.5
    a = Atoms('H', cell=(L, L, L), pbc=True)
    calc = GPAW(xc='vdW-DF', width=0.001,
                txt='H.vdw-DF.txt')
    a.set_calculator(calc)
    e1 = a.get_potential_energy()

    calc.set(txt='H.vdw-DF.spinpol.txt', spinpol=True)
    e2 = a.get_potential_energy()
    
    assert abs(calc.get_eigenvalues(spin=0)[0] -
               calc.get_eigenvalues(spin=1)[0]) < 1e-10

    assert abs(e1 - e2) < 1e-12

