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
e1ref = calc.get_reference_energy()
de12 = calc.get_xc_difference('revPBE')
calc.set(xc='revPBE')
e2 = atoms.get_potential_energy()
e2ref = calc.get_reference_energy()
de21 = calc.get_xc_difference('PBE')
print e1ref + e1 + de12 - (e2ref + e2)
print e1ref + e1 - (e2ref + e2 + de21)
print de12, de21
equal(e1ref + e1 + de12, e2ref + e2, 8e-4)
equal(e1ref + e1, e2ref + e2 + de21, 3e-3)

calc.write('revPBE.gpw')

de21b = Calculator('revPBE.gpw').get_xc_difference('PBE')
equal(de21, de21b, 9e-8)

# Remove setup
del setup_paths[0]
