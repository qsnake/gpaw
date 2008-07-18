from ase import *
from gpaw import *
calc = Calculator('anti.gpw')
atoms = calc.get_atoms()
up = calc.get_pseudo_density(0)
down = calc.get_pseudo_density(1)
zeta = (up - down) / (up + down)
write('magnetization.cube', atoms, data=zeta)

import os
os.system('vmd -e isosurfaces.vmd magnetization.cube')
