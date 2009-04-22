import os
from ase import *
from gpaw import GPAW
from ase.parallel import rank, barrier
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

# Generate setup
symbol = 'H'
if rank == 0:
    g = Generator(symbol, 'revPBE', scalarrel=True, nofiles=True)
    g.run(**parameters[symbol])
barrier()
setup_paths.insert(0, '.')

if 'GPAW_VDW' in os.environ:
    L = 2.5
    a = Atoms('H', cell=(L, L, L), pbc=True, calculator=GPAW(nbands=1,
                                                             xc='vdW-DF',
                                                             spinpol=True))
    e = a.get_potential_energy()

