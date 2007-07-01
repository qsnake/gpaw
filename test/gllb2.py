#!/usr/bin/env python
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import center, equal
from gpaw.atom.all_electron import AllElectron as AE
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths

# Generate non-scalar-relativistic setup for Cu:
g = Generator('Mg', 'GLLB', scalarrel=False, nofiles=True)
g.run(**parameters['Mg'])
setup_paths.insert(0, '.')

a = 8

SS = ListOfAtoms([Atom('Mg',[ 0, 0, 0 ] ) ],
                 cell=(a, a, a), periodic=False)

center(SS)

h = 0.25
calc = Calculator(h=h, verbosity=True, mix=(0.4, 3, 1), xc='GLLB', eigensolver='rmm-diis', tolerance=1e-10,
                  softgauss=False)

# Setup-generator eigenvalue -0.247637

SS.SetCalculator(calc)
SS.GetPotentialEnergy()
SSe = calc.GetEigenvalues() 
print SSe
diff = SSe[0] + 0.247637*27.211
print "Eigenvalue difference", diff
# Assure that 3D-GPAW wont change the 1D Result more than 0.1eV
assert (abs(diff)<0.10)

del setup_paths[0]
