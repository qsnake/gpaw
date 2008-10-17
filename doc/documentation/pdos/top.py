#!/usr/bin/env python
from ase import *
from ase.lattice.surface import fcc111, add_adsorbate
from gpaw import *
from gpaw import dscf

filename='top'

#-------------------------------------------

c_mol = GPAW(nbands=9, h=0.2, xc='RPBE', kpts=(8,6,1),
             spinpol=True,
             convergence={'energy': 100,
                          'density': 100,
                          'eigenstates': 1.0e-9,
                          'bands': 'occupied'})#, txt='CO.txt')

calc = GPAW(nbands=45, h=0.2, xc='RPBE', kpts=(8,6,1),
            eigensolver='cg',
            spinpol=True,
            mixer=MixerSum(nmaxold=5, beta=0.1, weight=100),
            convergence={'energy': 100,
                         'density': 100,
                         'eigenstates': 1.0e-7,
                         'bands': -10})#, txt=filename+'.txt')

#----------------------------------------

#  Import Slab with relaxed CO
slab = fcc111('Pt', size=(1, 2, 3), orthogonal=True)
add_adsorbate(slab, 'C', 2.0, 'ontop')
add_adsorbate(slab, 'O', 3.15, 'ontop')
slab.center(axis=2, vacuum=4.0)

#view(slab)

molecule = slab.copy()

del molecule [:-2]

#   Molecule
#----------------
molecule.set_calculator(c_mol)
molecule.get_potential_energy()
c_mol.write('CO.gpw', mode='all')

slab.set_calculator(calc)
slab.get_potential_energy()
calc.write('top.gpw', mode='all')
