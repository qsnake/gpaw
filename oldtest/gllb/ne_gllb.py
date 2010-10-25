import os
from ase import *
from gpaw import GPAW, Mixer
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from gpaw import setup_paths
from gpaw.test import equal
from gpaw.mpi import world

atom = 'Ne'
setup_paths.insert(0, '.')

for xcname in ['GLLBSC','GLLB']:
    if world.rank == 0:
        g = Generator(atom, xcname =xcname, scalarrel=False,nofiles=True)
        g.run(**parameters[atom])
        eps = g.e_j[-1]
    world.barrier()

    a = 15
    Ne = Atoms([Atom(atom, (0, 0, 0))],
               cell=(a, a, a), pbc=False)
    Ne.center()
    calc = GPAW(nbands=7, h=0.15, xc=xcname)
    Ne.set_calculator(calc)
    e = Ne.get_potential_energy()
    calc.write('Ne_'+xcname+'.gpw')
    eps3d = calc.wfs.kpt_u[0].eps_n[3]
    if world.rank == 0:
        equal(eps, eps3d, 1e-3)
        equal(e, 0, 5e-2)
