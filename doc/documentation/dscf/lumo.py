#!/usr/bin/env python
from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw import dscf
from numpy import *

filename='lumo'

#-------------------------------------------

c_mol = GPAW(nbands=9, h=0.2, xc='RPBE', kpts=(4,6,1),
             spinpol=True,
             convergence={'energy': 100,
                          'density': 100,
                          'eigenstates': 1.0e-9,
                          'bands': -2}, txt='CO_lumo.txt')

calc = GPAW(nbands=130, h=0.2, xc='RPBE', kpts=(4,6,1),
            eigensolver='cg',
            spinpol=True,
            mixer=MixerSum(nmaxold=5, beta=0.1, weight=100),
            convergence={'energy': 100,
                         'density': 100,
                         'eigenstates': 1.0e-7,
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

#Find band corresponding to lumo
lumo = c_mol.get_pseudo_wave_function(band=6, kpt=0, spin=0)
lumo = reshape(lumo, -1)

wf1_k = [c_mol.get_pseudo_wave_function(band=5, kpt=k, spin=0)
         for k in range(c_mol.nkpts)]
wf2_k = [c_mol.get_pseudo_wave_function(band=6, kpt=k, spin=0)
         for k in range(c_mol.nkpts)]

band_k = []
for k in range(c_mol.nkpts):
    wf1 = reshape(wf1_k[k], -1)
    wf2 = reshape(wf2_k[k], -1)
    p1 = abs(dot(wf1, lumo))
    p2 = abs(dot(wf2, lumo))
    if p1 > p2:
        band_k.append(5)
    else:
        band_k.append(6)

print band_k

#Lumo wavefunction
wf_u = [kpt.psit_nG[band_k[kpt.k]] for kpt in c_mol.kpt_u]

#Lumo projector overlaps
P_aui = []
for atom in c_mol.nuclei:
    P_aui.append([atom.P_uni[kpt.u][band_k[kpt.k]]
                  for kpt in c_mol.kpt_u])

#   Slab with adsorbed molecule
#-----------------------------------
slab.set_calculator(calc)
orbital = dscf.WaveFunction(calc, wf_u, P_aui, molecule=range(len(slab))[-2:])
dscf.dscf_calculation(calc, [[1.0, orbital, 1]], slab)
E_es = slab.get_potential_energy()

print 'Excitations energy: ', E_es-E_gs

