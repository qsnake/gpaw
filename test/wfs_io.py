"""Test automatically write out of restart files"""

import os
import sys
from gpaw import GPAW
from ase import *
from gpaw.utilities import equal
from ase.parallel import rank, barrier

endings = ['gpw']
try:
    import Scientific.IO.NetCDFXXXX
    endings.append('nc')
except ImportError:
    pass

for ending in endings:
    restart = 'gpaw-restart.' + ending
    restart_wf = 'gpaw-restart-wf.' + ending
    # H2
    H = Atoms([Atom('H', (0, 0, 0)), Atom('H', (0, 0, 1))])
    H.center(vacuum=2.0)

    wfdir = 'wfs_tmp'
    mode = ending+':' + wfdir + '/psit_s%dk%dn%d'

    if 1:
        calc = GPAW(nbands=2, convergence={'eigenstates': 1e-3})
        H.set_calculator(calc)
        H.get_potential_energy()
        calc.write(restart_wf, 'all')
        calc.write(restart, mode)

    barrier()
    # refine the restart file containing the wfs 
    E1 = GPAW(restart_wf,
              convergence=
              {'eigenstates': 1.e-5}).get_atoms().get_potential_energy()
        
    # refine the restart file and seperate wfs 
    calc = GPAW(restart, convergence={'eigenstates': 1.e-5})
    calc.read_wave_functions(mode)
    E2 = calc.get_atoms().get_potential_energy()

    print E1, E2
    equal(E1, E2, 1e-12)

    if rank == 0:
        os.remove(restart_wf)
        os.remove(restart)
        for f in os.listdir(wfdir):
            os.remove(wfdir + '/' + f)
        os.rmdir(wfdir)
