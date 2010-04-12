#!/usr/bin/env python
#PBS -l nodes=1:ppn=8
#PBS -q small

import numpy as np
import sys

from math import sqrt
from ase import Atoms, Atom
from ase.units import Bohr
from ase.parallel import paropen
from gpaw.atom.basis import BasisMaker
from gpaw import GPAW, Mixer
from gpaw.response.grid_chi import CHI
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
    sys.stdout = devnull 

GS = 1
GS2 = 1
EELS = 1
OpticalLimit = 1

nband = 40

if GS:

    # Gs calculation one
    basis = BasisMaker('C').generate(2, 1) # dzp

    kpts = (20,20,10)
    a=1.42
    c=3.355

    # AB stack
    atoms = Atoms([
         Atom('C',(1/3.0,1/3.0,0)),
         Atom('C',(2/3.0,2/3.0,0)),
         Atom('C',(0.   ,0.   ,0.5)),
         Atom('C',(1/3.0,1/3.0,0.5))
                   ],
                   pbc=(1,1,1))
    atoms.set_cell([(sqrt(3)*a/2.0,3/2.0*a,0),
                    (-sqrt(3)*a/2.0,3/2.0*a,0),
                    (0.,0.,2*c)],
                   scale_atoms=True)
    
#    atoms.center(axis=2)
    calc = GPAW(xc='LDA',
                kpts=kpts,
                h = 0.2,
                basis='dzp',
                usesymm=None,
                nbands=nband+10,
                convergence={'bands':nband},
                eigensolver = 'cg',
                width=0.05, txt = 'out.txt')
    
    atoms.set_calculator(calc)
#    view(atoms)

    atoms.get_potential_energy()
    calc.write('graphite1.gpw','all')

if GS2:

    # Gs calculation shifted kpoint
    calc = GPAW('graphite1.gpw')
    q = np.array([0.00001, 0., 0.])
    kpts = q + calc.get_ibz_k_points()

    atoms = calc.atoms
    calc = GPAW(xc='LDA',
                kpts=kpts,
                h = 0.2,
                basis='dzp',
                usesymm=None,
                nbands=nband+10,
                convergence={'bands':nband},
                eigensolver = 'cg',
                mixer=Mixer(0.05, 3, weight=100.0),
                width=0.05, txt = 'out2.txt')
    
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('graphite2.gpw','all')    

if EELS:

    calc = GPAW('graphite1.gpw',communicator=serial_comm)
                
    dw = 0.1  
    wmax = 40.

    f = paropen('graphite_q_list', 'w')
    for i in range(1,11):
       
        q = np.array([i/20., 0., 0.]) # Gamma-M excitation
        #q = np.array([i/20., -i/20., 0.]) # Gamma-K excitation
        chi = CHI()
        chi.nband = nband
        chi.initialize((calc,), q, wmax, dw, eta=0.2, Ecut = 40 + (i-1)*10)
        chi.periodic()
        
        chi.get_EELS_spectrum('graphite_EELS_' + str(i))
        chi.check_sum_rule()
        
        print >> f, sqrt(np.inner(chi.qq / Bohr, chi.qq / Bohr))
    
if OpticalLimit:
    
    calc1 = GPAW('graphite1.gpw',communicator=serial_comm)
    calc2 = GPAW('graphite2.gpw',communicator=serial_comm)
                
    dw = 0.1  
    wmax = 40.

    q = np.array([0.00001, 0., 0.])
    chi = CHI()
    chi.nband = nband
    chi.initialize((calc1,calc2), q, wmax, dw, eta=0.2, Ecut = 40)
    chi.periodic()
    
    chi.get_absorption_spectrum('graphite_absorption')
    chi.check_sum_rule()

