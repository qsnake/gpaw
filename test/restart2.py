"""Test automatically write out of restart files"""

import os
from gpaw import Calculator
from ase import *
from gpaw.utilities import equal

endings = ['gpw']
try:
    import Scientific.IO.NetCDFXXXX
    endings.append('nc')
except ImportError:
    pass

for ending in endings:
    restart = 'gpaw-restart.' + ending
    result  = 'gpaw-result.' + ending
    # H atom: 
    H = Atoms([Atom('H')])
    H.center(vacuum=3.0)

    calc = Calculator(nbands=1)
    calc.attach(calc.write, 4, restart)
    H.set_calculator(calc)
    H.get_potential_energy()
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
