from ase import *
from ase.lattice.surface import fcc100, add_adsorbate
from gpaw import *

use_emt = False
# Initial state:
# 2x2-Al(001) surface with 1 layer and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 1))
add_adsorbate(slab, 'Au', 1.6, 'hollow')
x, y, z = slab[-1].position
slab.translate((-x, -y, 0))
slab.center(axis=2, vacuum=3.0)

# Make sure the structure is correct:
#view(slab)

# Fix first layer:
mask = [atom.tag > 0 for atom in slab]
#print mask
fixlayers = FixAtoms(mask=mask)
slab.set_constraint(fixlayers)

if use_emt:
    calc = EMT()
else:
    calc = GPAW(h=0.25, kpts=(2, 2, 1), xc='PBE', txt='initial.txt')
slab.set_calculator(calc)
qn = QuasiNewton(slab, trajectory='initial.traj')
qn.run(fmax=0.05)
if not use_emt:
    assert len(calc.get_ibz_k_points()) == 1

del slab[-1]
slab.translate((x, y, 0))
add_adsorbate(slab, 'Au', 2.0, 'bridge')
if use_emt:
    calc = EMT()
else:
    calc = GPAW(h=0.25, kpts=(2, 2, 1), xc='PBE', txt='final.txt')
slab.set_calculator(calc)
qn = QuasiNewton(slab, trajectory='final.traj')
qn.run(fmax=0.05)
