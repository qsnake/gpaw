from ase import *
from ase.lattice.surface import fcc100, add_adsorbate
from gpaw import GPAW

# Initial state:
# 2x2-Al(001) surface with 1 layer and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 1))
add_adsorbate(slab, 'Au', 1.6, 'hollow')
slab.center(axis=2, vacuum=3.0)

# Make sure the structure is correct:
view(slab)

# Fix the Al atoms:
mask = [atom.symbol == 'Al' for atom in slab]
print mask
fixlayer = FixAtoms(mask=mask)
slab.set_constraint(fixlayer)

# Use GPAW:
calc = GPAW(h=0.25, kpts=(2, 2, 1), xc='PBE', txt='hollow.txt')
slab.set_calculator(calc)


qn = QuasiNewton(slab, trajectory='ontop.traj')

# Find optimal height.  The stopping criteria is: the force on the
# Au atom should be less than 0.05 eV/Ang
dyn.run(fmax=0.05)

calc.write('ontop.gpw') # Write gpw output after the minimization

print 'energy:', fcc.get_potential_energy()
print 'height:', fcc.positions[-1, 2] - fcc.positions[0, 2]

pseudo_density = calc.get_pseudo_density()
ae_density = calc.get_all_electron_density()

for format in ['cube', 'plt']:
    write('pseudo.' + format, fcc, data=pseudo_density)
    write('ae.' + format, fcc, data=ae_density)
