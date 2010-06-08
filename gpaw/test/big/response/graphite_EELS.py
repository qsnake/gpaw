# Symmetry can only be used in EELS spectra calculations for GPAW svn 6305 above.
# Refer to A. Rubio and V. Olevano, et.al, Physical Review B 69, 245419 (2004)
# for comparision of results

import os
import sys
from math import sqrt

import numpy as np
from ase import Atoms
from ase.units import Bohr
from ase.parallel import paropen

from gpaw import GPAW
from gpaw.mpi import rank
from gpaw.response.df import DF
from gpaw.utilities import devnull


if rank != 0:
    sys.stdout = devnull 

GS = 1
EELS = 1

nband = 60

if GS:

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
                
    w = np.linspace(0, 40, 401)
    f = paropen('graphite_q_list', 'w')

    for i in range(1,8):
       
        q = np.array([i/20., 0., 0.]) # Gamma-M excitation
        #q = np.array([i/20., -i/20., 0.]) # Gamma-K excitation

        ecut = 40 + (i-1)*10
        df = DF(calc='graphite.gpw', nbands=nband, q=q, w=w,
                eta=0.2,ecut=ecut)

        df1, df2 = df.get_dielectric_function()

        df.get_EELS_spectrum(df1, df2,filename='graphite_EELS_' + str(i))
        df.check_sum_rule(df1, df2)

        print >> f, sqrt(np.inner(df.qq / Bohr, df.qq / Bohr)), ecut

    if rank == 0:
        os.remove('graphite.gpw')



