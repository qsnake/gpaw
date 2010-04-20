# Symmetry can only be used in EELS spectra calculations for GPAW svn 6305 above.
# Refer to A. Rubio and V. Olevano, et.al, Physical Review B 69, 245419 (2004)
# for comparision of results

import numpy as np
import sys

from math import sqrt
from ase import Atoms
from ase.units import Bohr
from ase.parallel import paropen
from gpaw.atom.basis import BasisMaker
from gpaw import GPAW
from gpaw.response.grid_chi import CHI
from gpaw.mpi import serial_comm, rank
from gpaw.utilities import devnull


if rank != 0:
    sys.stdout = devnull 

GS = 1
EELS = 1

nband = 60

if GS:

    basis = BasisMaker('C').generate(2, 1) # dzp

    kpts = (20,20,7)
    a=1.42
    c=3.355

    # AB stack
    atoms = Atoms('C4',[
                  (1/3.0,1/3.0,0),
                  (2/3.0,2/3.0,0),
                  (0.   ,0.   ,0.5),
                  (1/3.0,1/3.0,0.5)
                  ],
                  pbc=(1,1,1))
    atoms.set_cell([(sqrt(3)*a/2.0,3/2.0*a,0),
                    (-sqrt(3)*a/2.0,3/2.0*a,0),
                    (0.,0.,2*c)],
                   scale_atoms=True)
    
    calc = GPAW(xc='LDA',
                kpts=kpts,
                h=0.2,
                basis='dzp',
                nbands=nband+10,
                convergence={'bands':nband},
                eigensolver='cg',
                width=0.05, txt='out.txt')
    
    atoms.set_calculator(calc)
#    view(atoms)

    atoms.get_potential_energy()
    calc.write('graphite.gpw','all')


if EELS:

    calc = GPAW('graphite.gpw',communicator=serial_comm)
                
    dw = 0.1  
    wmax = 40.

    f = paropen('graphite_q_list', 'w')
    for i in range(1,8):
       
        q = np.array([i/20., 0., 0.]) # Gamma-M excitation
        #q = np.array([i/20., -i/20., 0.]) # Gamma-K excitation
        chi = CHI()
        chi.nband = nband
        chi.initialize((calc,), q, wmax, dw, eta=0.2, Ecut = 40 + (i-1)*10)
        chi.periodic()
        
        chi.get_EELS_spectrum('graphite_EELS_' + str(i))
        chi.check_sum_rule()
        
        print >> f, sqrt(np.inner(chi.qq / Bohr, chi.qq / Bohr))
    


