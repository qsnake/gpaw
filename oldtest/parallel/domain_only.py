import os

from ase import *
from gpaw import GPAW
from gpaw.mpi import world 

O = Atoms([Atom('O')])
O.center(vacuum=2.)
calc = GPAW(nbands=6,
            h=.25,
            convergence={'eigenstates':1.e-2, 'energy':.1, 'density':.1},
            hund=True,
            parallel={'domain': world.size})
O.set_calculator(calc)
O.get_potential_energy()

print "calc.wfs.gd.comm.size, world.size=", calc.wfs.gd.comm.size, world.size
assert(calc.wfs.gd.comm.size == world.size)
