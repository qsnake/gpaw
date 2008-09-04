import os
from ase import *
from gpaw.utilities import equal
from gpaw import Calculator
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths

# Generate setup
symbol = 'He'
g = Generator(symbol, 'revPBE', scalarrel=True, nofiles=True)
g.run(exx=True, **parameters[symbol])
setup_paths.insert(0, '.')

a = 7.5 * Bohr
n = 16
atoms = Atoms([Atom('He', (0.0, 0.0, 0.0))], cell=(a, a, a), pbc=True)
calc = Calculator(gpts=(n, n, n), nbands=1, xc='PBE')
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
