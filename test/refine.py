"""Test automatically write out of restart files"""

import os
from gpaw import Calculator
from ASE import Atom, ListOfAtoms
from gpaw.utilities import equal
from gpaw.cluster import Cluster

##endings = ['nc']
endings = ['gpw']
## try:
##     import Scientific.IO.NetCDF
##     endings.append('nc')
## except ImportError:
##     pass

for ending in endings:
    restart_wf = 'gpaw-restart-wf.'+ending
    # H2
    H = Cluster([Atom('H', (0,0,0)),Atom('H', (0,0,1))])
    H.MinimalBox(2.)

    if 1:
        calc = Calculator(nbands=2,tolerance=1e-3)
        H.SetCalculator(calc)
        H.GetPotentialEnergy()
        calc.write(restart_wf,'all')

        # refine the result directly
        calc.set(tolerance=1.e-9)
        Edirect = H.GetPotentialEnergy()

    # refine the result after reading from a file
    calc = Calculator(restart_wf)
    calc.set(tolerance=1.e-9)
    H = calc.GetListOfAtoms()
    H.SetCalculator(calc)
    Erestart = H.GetPotentialEnergy()

    print Edirect, Erestart
    # Note: the different density mixing introduces small differences 
    equal(Edirect, Erestart, 1e-5)

    os.remove(restart_wf)
