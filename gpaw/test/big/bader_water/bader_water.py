from ase import *
from gpaw import *

atoms = molecule('H2O', cell=[7.5, 9, 9], calculator=GPAW(h=.17, xc='PBE'))
atoms.center()
atoms.get_potential_energy()

rho = atoms.calc.get_all_electron_density(gridrefinement=4) * Bohr**3
write('water_density.cube', atoms, data=rho)

rho = atoms.calc.get_pseudo_density(gridrefinement=2) * Bohr**3
write('water_pseudo_density.cube', atoms, data=rho)

import os
os.system('bader -p all_atom -p atom_index water_density.cube')
