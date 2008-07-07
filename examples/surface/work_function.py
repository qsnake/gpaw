from ase import *
from gpaw import Calculator

# Read in the 4-layer slab:
calc = Calculator('slab-4.gpw', txt=None)
slab = calc.get_atoms()

# Get the height of the unit cell:
L = slab.get_cell()[2, 2]

# Get the effective potential on a 3D grid:
v = calc.hamiltonian.vt_sG[0] * Hartree

nx, ny, nz = v.shape
z = linspace(0, L - L / nz, nz)

efermi = calc.get_fermi_level()

# Calculate xy averaged potential:
vz = v.reshape((nx * ny, nz)).mean(axis=0)

import pylab as p
p.plot(z, vz)
p.plot([0, L], [efermi, efermi])
p.show()

