import os
from ASE import Crystal, Atom
from ASE.Units import units
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


units.SetEnergyUnit('Hartree')
a = 5.0
n = 24
li = Crystal([Atom('Li', (0.0, 0.0, 0.0), magmom=1.0)], cell=(a, a, a))

calc = Calculator(gpts=(n, n, n), nbands=1, xc='PBE')
li.SetCalculator(calc)
e = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(e, -7.462, 0.056)

calc.set(xc='revPBE')
erev = li.GetPotentialEnergy() + calc.get_reference_energy()

# Remove setup
os.remove(symbol+'.'+XCFunctional('revPBE').get_name())
del setup_paths[0]

equal(erev, -7.487, 0.057)
equal(e - erev, 0.025, 0.002)
