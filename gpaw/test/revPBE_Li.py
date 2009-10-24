import os
from ase import *
from ase.parallel import rank, barrier
from gpaw import GPAW
from gpaw.test import equal
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths

# Generate setup
symbol = 'Li'
if rank == 0:
    g = Generator(symbol, 'revPBE', scalarrel=True, nofiles=True)
    g.run(exx=True, **parameters[symbol])
barrier()
setup_paths.insert(0, '.')

a = 5.0
n = 24
li = Atoms(symbol, magmoms=[1.0], cell=(a, a, a), pbc=True)

calc = GPAW(gpts=(n, n, n), nbands=1, xc='PBE')
li.set_calculator(calc)
e = li.get_potential_energy() + calc.get_reference_energy()
niter_PBE = calc.get_number_of_iterations()
equal(e, -7.462 * Hartree, 1.4)

calc.set(xc='revPBE')
erev = li.get_potential_energy() + calc.get_reference_energy()
niter_revPBE = calc.get_number_of_iterations()

del setup_paths[0]

equal(erev, -7.487 * Hartree, 1.3)
equal(e - erev, 0.025 * Hartree, 0.002 * Hartree)

print e, erev
energy_tolerance = 0.000001
niter_tolerance = 0
equal(e, -204.381098849, energy_tolerance) # svnversion 5252
equal(niter_PBE, 31, niter_tolerance) # svnversion 5252
equal(erev, -205.012303379, energy_tolerance) # svnversion 5252
equal(niter_revPBE, 19, niter_tolerance) # svnversion 5252
