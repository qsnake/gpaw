from ase import *
from gpaw import Calculator

calc = Calculator('ontop.gpw', txt=None)
HAl_density = calc.get_pseudo_valence_density()

atoms = calc.get_atoms()
HAl = atoms.copy()

# Save the position of the hydrogen atom for later:
H = atoms[2]

# Remove hydrogen and do a clean slab calculation:
del atoms[2]
atoms.get_potential_energy()
Al_density = calc.get_pseudo_valence_density()

# Add the hydrogen atom again and remove the slab:
atoms += H
del atoms[:2]

# Find the ground state for hydrogen only:
atoms.get_potential_energy()
H_density = calc.get_pseudo_valence_density()

diff = HAl_density - H_density - Al_density
write('diff.cube', HAl, data=diff)
write('diff.plt', HAl, data=diff)
