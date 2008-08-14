# creates: pt_h2.png

from ase import *
import os

a = 2.41 # Pt binding lenght
b = 0.90 # H2 binding lenght
c = 1.70 # Pt-H binding lenght
L = 7.00 # width of unit cell
N = 10 # Total number of Pt atoms on each side

H2 = Atoms('H2', positions=[(-c - b, 0, 0), (-c, 0, 0)])
atoms = Atoms('Pt', cell=(a, L, L)) * (N, 1, 1) + H2
atoms.set_cell([(N - 1) * a + b + 2 * c, L, L])
atoms.translate(-atoms.get_center_of_mass()) # center H2
atoms.set_scaled_positions(atoms.get_scaled_positions() % 1) #wrap to unit cell
atoms.translate([-(N / 2. - 2) * a, L / 2, L / 2])
atoms.set_cell([3 * a + 2 * c + b, L, L])

pov_options = {'display': True, 'transparent': False}
write('pt_h2.pov', atoms, show_unit_cell=2, **pov_options)
os.system('povray pt_h2.ini')



