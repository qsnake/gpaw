import os
from ase import Atoms
from ase.units import Hartree
from ase.parallel import rank, barrier
from gpaw import GPAW, PoissonSolver
from gpaw.test import equal, gen

# Generate setup
gen('Li', xcname='revPBE')

a = 5.0
n = 24
li = Atoms('Li', magmoms=[1.0], cell=(a, a, a), pbc=True)

calc = GPAW(gpts=(n, n, n), nbands=1, xc='PBE',
            poissonsolver=PoissonSolver(nn='M'))
li.set_calculator(calc)
e = li.get_potential_energy() + calc.get_reference_energy()
niter_PBE = calc.get_number_of_iterations()
equal(e, -7.462 * Hartree, 1.4)

calc.set(xc='revPBE')
erev = li.get_potential_energy() + calc.get_reference_energy()
niter_revPBE = calc.get_number_of_iterations()

equal(erev, -7.487 * Hartree, 1.3)
equal(e - erev, 0.025 * Hartree, 0.002 * Hartree)

print e, erev
energy_tolerance = 0.00008
niter_tolerance = 0
equal(e, -204.381098849, energy_tolerance) # svnversion 5252
#equal(niter_PBE, 31, niter_tolerance) # svnversion 5252 # niter depends on the number of processes
assert 30 <= niter_PBE <= 31, niter_PBE
equal(erev, -205.012303379, energy_tolerance) # svnversion 5252
#equal(niter_revPBE, 19, 0) # svnversion 5252 # niter depends on the number of processes
assert 17 <= niter_revPBE <= 19, niter_revPBE
