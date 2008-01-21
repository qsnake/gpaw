import os
from ase import *
from gpaw import Calculator

a = 4.    # Size of unit cell (Angstrom)
c = a / 2
d = 0.74  # Experimental bond length
molecule = Atoms([Atom('H', (c - d / 2, c, c)),
                        Atom('H', (c + d / 2, c, c))],
                       cell=(a, a, a), pbc=False)
calc = Calculator(h=0.2, nbands=1, xc='PBE', txt=None)
molecule.set_calculator(calc)
e2 = molecule.get_potential_energy()
calc.write('H2.gpw')
calc.write('H2a.gpw', mode='all')
molecule.get_forces()
calc.write('H2f.gpw')
calc.write('H2fa.gpw', mode='all')

from time import time
t0 = time()
molecule = Calculator('H2.gpw', txt=None).get_atoms()
f1 = molecule.get_forces()
t1 = time() - t0
molecule = Calculator('H2a.gpw', txt=None).get_atoms()
f2 = molecule.get_forces()
t2 = time() - t0 - t1
molecule = Calculator('H2f.gpw', txt=None).get_atoms()
f3 = molecule.get_forces()
t3 = time() - t0 - t1 - t2
molecule = Calculator('H2fa.gpw', txt=None).get_atoms()
f4 = molecule.get_forces()
t4 = time() - t0 - t1 - t2 - t3
print t1, t2, t3, t4
assert t2 < t1 / 2
assert t3 < 0.5
assert t4 < 0.5
print f1
print f2
print f3
print f4
assert sum((f1 - f4).ravel()**2) < 1e-6
assert sum((f2 - f4).ravel()**2) < 1e-6
assert sum((f3 - f4).ravel()**2) < 1e-6

positions = molecule.get_positions()
#                 x-coordinate      x-coordinate
#                 v                 v
d0 = positions[1, 0] - positions[0, 0]
#              ^                 ^
#              second atom       first atom

print 'experimental bond length:'
print 'hydrogen molecule energy: %7.3f eV' % e2
print 'bondlength              : %7.3f Ang' % d0

# Find the theoretical bond length:
relax = QuasiNewton(molecule)
relax.run(fmax=0.05)

e2 = molecule.get_potential_energy()

positions = molecule.get_positions()
#                 x-coordinate      x-coordinate
#                 v                 v
d0 = positions[1, 0] - positions[0, 0]
#              ^                 ^
#              second atom       first atom

print 'PBE energy minimum:'
print 'hydrogen molecule energy: %7.3f eV' % e2
print 'bondlength              : %7.3f Ang' % d0


molecule = Calculator('H2fa.gpw', txt=None).get_atoms()
relax = QuasiNewton(molecule)
relax.run(fmax=0.05)
e2q = molecule.get_potential_energy()
positions = molecule.get_positions()
d0q = positions[1, 0] - positions[0, 0]
assert abs(e2 - e2q) < 2e-6
assert abs(d0q - d0) < 4e-4

os.system('rm H2.gpw H2a.gpw H2f.gpw H2fa.gpw')
