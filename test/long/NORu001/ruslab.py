from sys import argv
from gpaw import *
from ase import *
from ase.lattice.surface import *

tag = 'Ru001'

adsorbate_heights = {'N': 1.108, 'O': 1.257}

slab = hcp0001('Ru', size=(2, 2, 4), a=2.72, c=1.58*2.72, vacuum=7.0,
               orthogonal=True)
slab.center(axis=2)

if len(argv) > 1:
    adsorbate = argv[1]
    tag = adsorbate + tag
    add_adsorbate(slab, adsorbate, adsorbate_heights[adsorbate], 'hcp')

slab.set_constraint(FixAtoms(mask=slab.get_tags() >= 3))

calc = GPAW(xc='PBE',
            h=0.2,
            mixer=Mixer(0.1, 5, metric='new', weight=100.0),
            stencils=(3, 3),
            width=0.1,
            kpts=[4, 4, 1],
            txt=tag + '.txt')
slab.set_calculator(calc)
  
opt = HessLBFGS(slab, logfile=tag + '.log', trajectory=tag + '.traj')
opt.run(fmax=0.05)
