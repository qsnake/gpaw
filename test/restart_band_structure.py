from gpaw import GPAW, restart
from ase import *
from ase.calculators import numeric_force
from gpaw.utilities import equal
import os
from gpaw import setup_paths
from gpaw.mpi import world
setup_paths.insert(0, '.')

for xc in ['LDA','GLLBSC']:
    a = 4.23
    bulk = Atoms('Si2', cell=(a, a, a), pbc=True,
              scaled_positions=[[0, 0, 0], [.5, .5, .5]])
    calc = GPAW(h=0.25,
                nbands=8,
                width=0.01,
                kpts=(3, 3, 3), convergence={'eigenstates':1e-12, 'bands':8}, xc=xc, eigensolver='cg')

    bulk.set_calculator(calc)
    bulk.get_potential_energy()
    print calc.get_ibz_k_points()
    old_eigs = calc.get_eigenvalues(kpt=3)
    calc.write('Si_gs.gpw')
    del bulk
    del calc
    bulk, calc = restart('Si_gs.gpw', fixdensity=True, kpts=[[0,0,0],[1./3,1./3,1./3]])
    bulk.get_potential_energy()

    if world.rank == 0:
        os.remove('Si_gs.gpw')
    diff = calc.get_eigenvalues(kpt=1)[:6]-old_eigs[:6]
    if world.rank == 0:
        print "occ. eig. diff.", diff
        error = max(abs(diff))
        assert error < 5e-6
