#!/usr/bin/env python
from ase import *
from gpaw import GPAW
import tab

def get_eigs(calc):
    # returns eigs_skn
    return np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                      for k in range(calc.wfs.nibzkpts)]
                     for s in range(calc.wfs.nspins)])

noconv = dict(fixdensity=True, convergence={'eigenstates': 1e9, 'energy': 1e9})

# Do the selfconsistent calculation and dump to file
calc = GPAW(nbands=2, xc='PBE', txt=None)
atoms = Atoms('H', [(.5, .5, .5)], magmoms=[1], pbc=False, calculator=calc)
atoms.center(vacuum=3)
atoms.get_potential_energy()
calc.write('nontest', mode='all')


# Do the non-selfconsistent calculations

calc = GPAW('nontest')
E_nPBE, eigs_nPBE = calc.get_nonselfconsistent_eigenvalues('PBE')
E_nPBE += calc.get_reference_energy()
#calc.set(xc='PBE', **noconv)
E_PBE = calc.get_potential_energy() + calc.get_reference_energy()
eigs_PBE = get_eigs(calc)

calc = GPAW('nontest')
E_nLDA, eigs_nLDA = calc.get_nonselfconsistent_eigenvalues('LDA')
E_nLDA += calc.get_reference_energy()
print calc.get_reference_energy()
calc.set(xc='LDA', **noconv)
print calc.get_reference_energy()
E_LDA = atoms.get_potential_energy() + calc.get_reference_energy()
eigs_LDA = get_eigs(calc)

calc = GPAW('nontest')
E_nEXX, eigs_nEXX = calc.get_nonselfconsistent_eigenvalues('EXX')
E_nEXX += calc.get_reference_energy()
print calc.get_reference_energy()
calc.set(xc='EXX', **noconv)
print calc.get_reference_energy()
E_EXX = atoms.get_potential_energy() + calc.get_reference_energy()
eigs_EXX = get_eigs(calc)

print 'LDA'
print E_LDA, E_nLDA
print eigs_LDA - eigs_nLDA

print 'PBE'
print E_PBE, E_nPBE
print eigs_PBE - eigs_nPBE

print 'EXX'
print E_EXX, E_nEXX
print eigs_EXX - eigs_nEXX


# selfconsistent LDA
## -0.883282451388
## -12.1289625969
## [[[-7.24822465  3.13856831]]
##  [[-2.43790837  4.22731969]]]

# selfconsistent PBE
## -1.0996835981
## -12.4901666629
## [[[-7.52740492  3.04583427]]
##  [[ 5.14286664  7.01543167]]]

# selfconsistent EXX starting fom selfcon. LDA orbitals
## -13.6018033066
## 0.0
## [[[-13.61181534   5.70652396]]
##  [[  1.44180798   6.18816302]]]

# selfconsistent EXX starting fom selfcon. PBE orbitals
## -13.6818871849
## 0.0
## [[[-13.65755584   5.71060805]]
##  [[  1.45560588   6.18935012]]]
