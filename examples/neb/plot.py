"""Make a plot"""

from ase import *
from ase.data.colors import jmol_colors

a = 4.0614
b = a / sqrt(2)
h = b / 2
atoms = Atoms('Al2', pbc=(1, 1, 0), cell=(a, b, 2 * h),
              positions=((0, 0, 0),
                         (a / 2, b / 2, -h)))
atoms *= (2, 2, 2)
atoms.append(Atom('Al', (a / 2, b / 2, 3 * h)))
atoms.center(vacuum=4., axis=2)

atoms *= (2, 3, 1)
atoms.cell /= [2, 3, 1]
rotation = '-60x, 10y'
radii = covalent_radii[atoms.numbers] * 1.2
colors = jmol_colors[atoms.numbers]
colors[16::17] = [1, 0, 0]
pov_options = {'display': False,
               'transparent': False,
               'canvas_width': 500}

write('slab.pov', atoms,
      rotation=rotation,
      colors=colors,
      radii=radii,
      show_unit_cell=2,
      **pov_options)

import os
os.system('povray slab.ini')
