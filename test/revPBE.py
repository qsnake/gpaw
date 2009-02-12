import os
from ase import *
from ase.parallel import rank, barrier
from gpaw.utilities import equal
from gpaw import GPAW
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths

# Generate setup
symbol = 'He'
if rank == 0:
    g = Generator(symbol, 'revPBE', scalarrel=True, nofiles=True)
    g.run(exx=True, **parameters[symbol])
barrier()
setup_paths.insert(0, '.')

a = 7.5 * Bohr
n = 16
atoms = Atoms([Atom(symbol, (0.0, 0.0, 0.0))], cell=(a, a, a), pbc=True)
calc = GPAW(gpts=(n, n, n), nbands=1, xc='PBE')
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy()
e1a = calc.get_reference_energy()
calc.set(xc='revPBE')
e2 = atoms.get_potential_energy()
e2a = calc.get_reference_energy()

del setup_paths[0]

equal(e1a, -2.893 * Hartree, 8e-3)
equal(e2a, -2.908 * Hartree, 9e-3)
equal(e1, e2, 4e-3)
