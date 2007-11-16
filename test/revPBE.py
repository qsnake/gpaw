import os
from ASE import Crystal, Atom
from ASE.Units import units
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


units.SetUnits('Bohr', 'Hartree')

a = 7.5
n = 16
atoms = Crystal([Atom('He', (0.0, 0.0, 0.0))], cell=(a, a, a))
calc = Calculator(gpts=(n, n, n), nbands=1, xc='PBE')
atoms.SetCalculator(calc)
e1 = atoms.GetPotentialEnergy()
e1a = calc.get_reference_energy()
calc.set(xc='revPBE')
e2 = atoms.GetPotentialEnergy()
e2a = calc.get_reference_energy()

# Remove setup
os.remove(symbol+'.'+XCFunctional('revPBE').get_name())
del setup_paths[0]

equal(e1a, -2.893, 27e-5)
equal(e2a, -2.908, 40e-5)
equal(e1, e2, 29e-5)
