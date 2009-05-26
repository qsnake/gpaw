from ase import *
from gpaw import GPAW

# Read in the 5-layer slab:
calc = GPAW('slab-5.gpw', txt=None)
slab = calc.get_atoms()

# Get the height of the unit cell:
L = slab.get_cell()[2, 2]

# Get the effective potential on a 3D grid:
v = calc.get_effective_potential()

nx, ny, nz = v.shape
z = linspace(0, L, nz, endpoint=False)

efermi = calc.get_fermi_level()

# Calculate xy averaged potential:
vz = v.mean(axis=0).mean(axis=0)
print 'Work function: %.2f eV' % (vz.max() - efermi)

import pylab as p
p.plot(z, vz, label='xy averaged effective potential')
p.plot([0, L], [efermi, efermi], label='Fermi level')
p.ylabel('Potential / V')
p.xlabel('z / Angstrom')
p.legend(loc=0)
p.show()

