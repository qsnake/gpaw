"""Test automatically write out of restart files"""

import os
import filecmp
from gpaw import GPAW
from ase import *
from gpaw.test import equal
restart = 'gpaw-restart'
result  = 'gpaw-result'
# H atom:
H = Atoms([Atom('H')])
H.center(vacuum=3.0)

calc = GPAW(gpts=(32, 32, 32), nbands=1)
calc.attach(calc.write, 4, restart)
H.set_calculator(calc)
e = H.get_potential_energy()
niter = calc.get_number_of_iterations()
calc.write(result)

# the two files should be equal
from gpaw.mpi import rank
if rank == 0:
    for f in ['gpaw-restart', 'gpaw-result']:
        os.system('rm -rf %s; mkdir %s; cd %s; tar xf ../%s.gpw' %
                  (f, f, f, f))
    assert os.system('diff -r gpaw-restart gpaw-result > /dev/null') == 0
    os.system('rm -rf gpaw-restart gpaw-result')

energy_tolerance = 0.00006
niter_tolerance = 0
equal(e, 0.0451789, energy_tolerance)
equal(niter, 12, niter_tolerance)
