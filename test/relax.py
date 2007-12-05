import os
from ASE import Atom, ListOfAtoms
from ASE.Dynamics.ConjugateGradient import ConjugateGradient
from gpaw import Calculator

a = 4.    # Size of unit cell (Angstrom)
c = a / 2
d = 0.74  # Experimental bond length
molecule = ListOfAtoms([Atom('H', (c - d / 2, c, c)),
                        Atom('H', (c + d / 2, c, c))],
                       cell=(a, a, a), periodic=False)
calc = Calculator(h=0.2, nbands=1, xc='PBE', txt=None)
molecule.SetCalculator(calc)
e2 = molecule.GetPotentialEnergy()
calc.write('H2.gpw')
calc.write('H2a.gpw', mode='all')
molecule.GetCartesianForces()
calc.write('H2f.gpw')
calc.write('H2fa.gpw', mode='all')

from time import time
t0 = time()
molecule = Calculator('H2.gpw', txt=None).get_atoms()
f1 = molecule.GetCartesianForces()
t1 = time() - t0
molecule = Calculator('H2a.gpw', txt=None).get_atoms()
f2 = molecule.GetCartesianForces()
t2 = time() - t0 - t1
molecule = Calculator('H2f.gpw', txt=None).get_atoms()
f3 = molecule.GetCartesianForces()
t3 = time() - t0 - t1 - t2
molecule = Calculator('H2fa.gpw', txt=None).get_atoms()
f4 = molecule.GetCartesianForces()
t4 = time() - t0 - t1 - t2 - t3
print t1, t2, t3, t4
assert t2 < t1 / 2
assert t3 < 0.5
assert t4 < 0.5
print f1
print f2
print f3
print f4
assert sum((f1 - f4).flat**2) < 1e-6
assert sum((f2 - f4).flat**2) < 1e-6
assert sum((f3 - f4).flat**2) < 1e-6

positions = molecule.GetCartesianPositions()
#                 x-coordinate      x-coordinate
#                 v                 v
d0 = positions[1, 0] - positions[0, 0]
#              ^                 ^
#              second atom       first atom

print 'experimental bond length:'
print 'hydrogen molecule energy: %7.3f eV' % e2
print 'bondlength              : %7.3f Ang' % d0

# Find the theoretical bond length:
relax = ConjugateGradient(molecule, fmax=0.05)
relax.Converge()

e2 = molecule.GetPotentialEnergy()

positions = molecule.GetCartesianPositions()
#                 x-coordinate      x-coordinate
#                 v                 v
d0 = positions[1, 0] - positions[0, 0]
#              ^                 ^
#              second atom       first atom

print 'PBE energy minimum:'
print 'hydrogen molecule energy: %7.3f eV' % e2
print 'bondlength              : %7.3f Ang' % d0

os.system('rm H2.gpw H2a.gpw H2f.gpw H2fa.gpw')
