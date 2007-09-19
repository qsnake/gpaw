"""Test automatically write out of restart files"""

import os
import sys
from gpaw import Calculator
from ASE import Atom, ListOfAtoms
from gpaw.utilities import equal
from gpaw.cluster import Cluster

endings = ['nc']
##endings = ['gpw']
## try:
##     import Scientific.IO.NetCDF
##     endings.append('nc')
## except ImportError:
##     pass

for ending in endings:
    restart = 'gpaw-restart.'+ending
    restart_wf = 'gpaw-restart-wf.'+ending
    # H2
    H = Cluster([Atom('H', (0,0,0)),Atom('H', (0,0,1))])
    H.MinimalBox(2.)

    wfdir = 'wfs_tmp'
    mode = ending+':'+wfdir+'/psit_s%dk%dn%d'

    if 1:
        calc = Calculator(nbands=2,tolerance=1e-3)
        H.SetCalculator(calc)
        H.GetPotentialEnergy()
        calc.write(restart_wf,'all')
        calc.write(restart,mode)

    # refine the restart file containing the wfs 
    E1 = Calculator(restart_wf,tolerance=1.e-5).GetPotentialEnergy()
        
    # refine the restart file and seperate wfs 
    calc = Calculator(restart,tolerance=1.e-5)
    calc.read_wave_functions(mode)
    E2 = calc.GetPotentialEnergy()

    print E1, E2
    equal(E1, E2, 1e-12)

    os.remove(restart_wf)
    os.remove(restart)
    for f in os.listdir(wfdir):
        os.remove(wfdir+'/'+f)
    os.rmdir(wfdir)
