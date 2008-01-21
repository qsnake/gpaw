import os
from ase import *
from gpaw.utilities import equal
from gpaw import Calculator
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths

# Generate setup
symbol = 'Li'
g = Generator(symbol, 'revPBE', scalarrel=True, nofiles=True)
g.run(exx=True, **parameters[symbol])
setup_paths.insert(0, '.')

a = 5.0
n = 24
li = Atoms([Atom('Li', (0.0, 0.0, 0.0), magmom=1.0)], cell=(a, a, a), pbc=True)

calc = Calculator(gpts=(n, n, n), nbands=1, xc='PBE')
li.set_calculator(calc)
e = li.get_potential_energy() + calc.get_reference_energy()
equal(e, -7.462 * Hartree, 1.4)

calc.set(xc='revPBE')
erev = li.get_potential_energy() + calc.get_reference_energy()

# Remove setup
os.remove(symbol + '.' + XCFunctional('revPBE').get_name())
del setup_paths[0]

equal(erev, -7.487 * Hartree, 1.3)
equal(e - erev, 0.025 * Hartree, 0.002 * Hartree)
