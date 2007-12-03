"""Test automatically write out of restart files"""

import os
from gpaw import Calculator
from ASE import Atom, ListOfAtoms
from gpaw.utilities import equal
from gpaw.cluster import Cluster

endings = ['gpw']
try:
    import Scientific.IO.NetCDF
    endings.append('nc')
except ImportError:
    pass

for ending in endings:
    restart = 'gpaw-restart.' + ending
    result  = 'gpaw-result.' + ending
    # H atom: 
    H = Cluster([Atom('H', (0,0,0))])
    H.MinimalBox(3.0)

    calc = Calculator(nbands=1)
    calc.attach(calc.write, 4, restart)
    H.SetCalculator(calc)
    H.GetPotentialEnergy()
    calc.write(result)

    # the two files should be equal
    if ending == 'nc':
        assert os.system('diff %s %s > /dev/null' % (restart, result)) == 0
    else:
        for f in ['gpaw-restart', 'gpaw-result']:
            os.system('rm -rf %s; mkdir %s; cd %s; tar xf ../%s.gpw' %
                      (f, f, f, f))
        assert os.system('diff -r gpaw-restart gpaw-result > /dev/null') == 0
        os.system('rm -rf gpaw-restart gpaw-result')
        

    os.remove(restart)
    os.remove(result)
