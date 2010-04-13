#!/usr/bin/env python

#PBS -m ae
#PBS -q verylong
#PBS -l nodes=1:ppn=8
#!/usr/bin/env python

from ase import *
from gpaw import *
from ase.optimize.bfgslinesearch import *

slab = read('CH4Au532.xyz')
slab.set_cell([[7.309254, 0., 0.], [4.872836, 7.509545, 0.], [0., 0., 20.]],scale_atoms=False)
slab.set_pbc((1,1,1))

calc = GPAW(h=0.18, kpts=(4, 4, 1))
slab.set_calculator(calc)

dyn = BFGSLineSearch(slab,trajectory='relax.traj')
dyn.run(fmax=0.02)
