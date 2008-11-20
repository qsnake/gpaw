import os
from ase import *
from ase.parallel import rank, barrier
from gpaw.utilities import equal
from gpaw import Calculator
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
atoms = Atoms([Atom('He', (0.0, 0.0, 0.0))], cell=(a, a, a), pbc=True)
calc = Calculator(gpts=(n, n, n), nbands=1, xc='revPBE')
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy()
calc.write('He')
e2 = e1 + calc.get_xc_difference('vdWDF')
print e1, e2
