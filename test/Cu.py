import os
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

# Generate non-scalar-relativistic setup for Cu:
g = Generator('Cu', scalarrel=False, nofiles=True)
g.run(logderiv=True, **parameters['Cu'])
setup_paths.insert(0, '.')

a = 8.0
c = a / 2
Cu = ListOfAtoms([Atom('Cu', (c, c, c), magmom=1)],
                 cell=(a, a, a))

calc = Calculator(h=0.2, lmax=0)
Cu.SetCalculator(calc)
Cu.GetPotentialEnergy()

e_4s_major = calc.kpt_u[0].eps_n[5]
e_3d_minor = calc.kpt_u[1].eps_n[4]

#
# The reference values are from:
#
#   http://physics.nist.gov/PhysRefData/DFTdata/Tables/29Cu.html
#

print e_4s_major - e_3d_minor, -0.184013 - -0.197109
assert abs(e_4s_major - e_3d_minor - (-0.184013 - -0.197109)) < 0.001

os.system('rm Cu.??.ld.?')
# remove Cu.* setup
os.remove(calc.setups[0].filename)
del setup_paths[0]
