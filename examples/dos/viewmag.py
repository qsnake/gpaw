from ase import *
from gpaw import *
calc = Calculator('anti.gpw')
atoms = calc.get_atoms()
up, down = calc.get_pseudo_valence_density(pad=True)
zeta = (up - down) / (up + down)
write('magnetization.cube', atoms, data=zeta)

import os
os.system('vmd -e isosurfaces.vmd magnetization.cube')
