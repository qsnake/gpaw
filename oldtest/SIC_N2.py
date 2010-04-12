#!/usr/bin/env python
from ase import *
from gpaw import GPAW
from gpaw import restart
from gpaw.wannier import Wannier
from gpaw.eigensolvers.nuimin import NUIMin
from gpaw.sic import SIC

#N = Atoms(symbols='C6H6',
#                pbc=False,
#                positions=[
#    ( 0.000000,  1.395248, 0.000000),
#    ( 1.208320,  0.697624, 0.000000),
#    ( 1.208320, -0.697624, 0.000000),
#    ( 0.000000, -1.395248, 0.000000),
#    (-1.208320, -0.697624, 0.000000),
#    (-1.208320,  0.697624, 0.000000),
#    ( 0.000000,  2.482360, 0.000000),
#    ( 2.149787,  1.241180, 0.000000),
#    ( 2.149787, -1.241180, 0.000000),
#    ( 0.000000, -2.482360, 0.000000),
#    (-2.149787, -1.241180, 0.000000),
#    (-2.149787,  1.241180, 0.000000)])

#N.center(vacuum=2.5)
    
#N = Atoms('N2', positions=[(0, 0, 0), (0, 0, 1.14)],
#                magmoms  =[-3.0, 3.0])
N = Atoms('H2', positions=[(0, 0, 0), (0, 0, 0.74)],
                magmoms  =[-1.0, 1.0])
#N.set_initial_magnetic_moments([1.0,-1.0])
N.center(vacuum=2.0)

if 0:
    calc = GPAW(#xc=SIC(1, 'LDA'),
                #eigensolver=NUIMin(),
                maxiter=500,
                convergence={'eigenstates': 1e-3},
                nbands=0)
    N.set_calculator(calc)
    e = N.get_potential_energy()
    calc.write('N2.gpw', 'all')
else:
    atoms, calc = restart('N2.gpw')
    nbands=calc.wfs.nbands
    nspins=calc.get_number_of_spins()
    # Initialize the Wannier class
    if 1:
        w = Wannier(calc,spin=1)
        w.localize()
        centers = w.get_centers()
        #view(atoms + Atoms(symbols='X5', positions=centers))
        psit_nG = calc.wfs.kpt_u[0].psit_nG
        shape = psit_nG.shape
        psit_nG.shape = (nbands, -1)
        calc.wfs.kpt_u[0].psit_nG = np.dot(w.U_nn, psit_nG)
        calc.wfs.kpt_u[0].psit_nG.shape = shape

    calc.wfs.eigensolver=None
    calc.set(xc=SIC(2, 'LDA'),
             maxiter=750,
             eigensolver=NUIMin(),
             convergence={'eigenstates': 1e-3})
    calc.get_potential_energy()
    
    for s in range(nspins):
        for n in range(nbands):
            write('N2-%d-%d.cube' % (n,s), atoms,
                  data=calc.get_pseudo_wave_function(n,s)**2)
