"""Test automatically write out of restart files"""

import os
import filecmp
from gpaw import GPAW
from ase import *
from gpaw.utilities import equal
restart = 'gpaw-restart'
result  = 'gpaw-result'
# H atom:
H = Atoms([Atom('H')])
H.center(vacuum=3.0)

calc = GPAW(nbands=1)
calc.attach(calc.write, 4, restart)
H.set_calculator(calc)
H.get_potential_energy()
calc.write(result)

# the two files should be equal
from gpaw.mpi import rank
if rank == 0:
    for f in ['gpaw-restart', 'gpaw-result']:
        os.system('rm -rf %s; mkdir %s; cd %s; tar xf ../%s.gpw' %
                  (f, f, f, f))
    assert os.system('diff -r gpaw-restart gpaw-result > /dev/null') == 0
    os.system('rm -rf gpaw-restart gpaw-result')
