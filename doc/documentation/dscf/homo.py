#!/usr/bin/env python
from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw import dscf

filename='homo'

#-------------------------------------------

c_mol = GPAW(nbands=9, h=0.2, xc='RPBE', kpts=(4,6,1),
             spinpol=True,
             convergence={'energy': 100,
                          'density': 100,
                          'eigenstates': 1.0e-9,
                          'bands': 'occupied'}, txt='CO_homo.txt')

calc = GPAW(nbands=80, h=0.2, xc='RPBE', kpts=(4,6,1),
            eigensolver='cg',
            spinpol=True,
            mixer=MixerSum(nmaxold=5, beta=0.1, weight=100),
            convergence={'energy': 100,
                         'density': 100,
                         'eigenstates': 1.0e-9,
                         'bands': -10}, txt=filename+'.txt')

#----------------------------------------

#Import Slab with relaxed CO
slab = Calculator('gs.gpw').get_atoms()
E_gs = slab.get_potential_energy()

molecule = slab.copy()

del molecule [:-2]

#   Molecule
#----------------
molecule.set_calculator(c_mol)
molecule.get_potential_energy()

#Homo wavefunction
wf_u = [kpt.psit_nG[4] for kpt in c_mol.kpt_u]

#Homo projector overlaps
P_aui = [a.P_uni[:,4,:] for a in c_mol.nuclei]

#   Slab with adsorbed molecule
#-----------------------------------
slab.set_calculator(calc)
orbital = dscf.WaveFunction(calc, wf_u, P_aui, molecule=range(len(slab))[-2:],
                            Estart=-100.0, Eend=0.0)
dscf.dscf_calculation(calc, [[-1.0, orbital, 1]], slab)
E_es = slab.get_potential_energy()

print 'Excitations energy: ', E_es-E_gs
