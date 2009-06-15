from gpaw import GPAW, restart
from ase import *
from ase.calculators import numeric_force
from gpaw.utilities import equal
import os

for xc in ['GLLBSC','LDA']:
    a = 5.404
    bulk = Atoms(symbols='Si8',
                 positions=[(0, 0, 0),
                            (0, 0.5, 0.5),
                            (0.5, 0, 0.5),
                            (0.5, 0.5, 0),
                            (0.25, 0.25, 0.25),
                            (0.25, 0.75, 0.75),
                            (0.75, 0.25, 0.75),
                            (0.75, 0.75, 0.25)],
                 pbc=True)
    bulk.set_cell((a, a, a), scale_atoms=True)
    n = 20
    calc = GPAW(gpts=(n, n, n),
                nbands=24,
                width=0.01,
                kpts=(3, 3, 3), convergence={'eigenvalue':1e-12}, xc=xc)

    bulk.set_calculator(calc)
    bulk.get_potential_energy()
    print calc.get_ibz_k_points()
    old_eigs = calc.get_eigenvalues(kpt=1)
    calc.write('Si_gs.gpw')
    del bulk
    del calc
    bulk, calc = restart('Si_gs.gpw', fixdensity=True, kpts=[[0,0,0],[1./3,0,0]])
    bulk.get_potential_energy()

    os.remove('Si_gs.gpw')
    diff = calc.get_eigenvalues(kpt=1)[1:16]-old_eigs[1:16]
    print "occ. eig. diff.", diff
    error = max(abs(diff))
    assert error < 2e-5
