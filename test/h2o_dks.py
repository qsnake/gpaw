from ase import *
from gpaw import Calculator
from gpaw.testing.g2 import get_g2
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

# Generate setup for oxygen with a core-hole:
g = Generator('O', xcname='PBE', scalarrel=True,
              corehole=(1, 0, 1.0), nofiles=True)
g.run(name='fch1s', **parameters['O'])
setup_paths.insert(0, '.')

atoms = get_g2('H2O')
atoms.center(vacuum=2.5)

calc = Calculator(xc='PBE')
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy() + calc.Eref * Hartree

atoms[0].magmom = 1
calc.set(charge=-1, setups={'O': 'fch1s'})
e2 = atoms.get_potential_energy() + calc.Eref * Hartree

print 'Energy difference %.3f eV' % (e2 - e1)
assert abs(e2 - e1 - 533.117) < 0.001
