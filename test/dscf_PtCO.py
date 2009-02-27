from ase import *
from ase.lattice.surface import fcc100, add_adsorbate
from gpaw import *
from gpaw.dscf import dscf_calculation, MolecularOrbital, AEOrbital
from gpaw.utilities import equal

atoms = fcc100('Pt', (1,1,2))
add_adsorbate(atoms, 'C', 2.00, 'ontop')
add_adsorbate(atoms, 'O', 3.15, 'ontop')
atoms.center(axis=2, vacuum=3)

calc_1 = GPAW(h=0.24,
              nbands=20,
              xc='PBE',
              kpts=[4,4,1],
              convergence={'energy': 0.001,
                           'density': 0.001,
                           'eigenstates': 1.0e-7,
                           'bands': -3})

atoms.set_calculator(calc_1)
E_gs = atoms.get_potential_energy()

# Dscf calculation based on the MolecularOrbital class
calc_1.set(nbands=25, spinpol=True)
lumo = MolecularOrbital(calc_1, molecule=[2,3],
                        w=[[0.,0.,0.,1.],[0.,0.,0.,-1.]])
dscf_calculation(calc_1, [[1.0, lumo, 1]], atoms)
E_es1 = atoms.get_potential_energy()
equal(E_es1, E_gs + 4.55, 0.1)

# Dscf calculation based on the AEOrbital class. Two new calculators
# are needed. One for the molecule and one for the dscf calcultion
calc_mol = GPAW(h=0.24,
                nbands=8,
                xc='PBE',
                spinpol=True,
                kpts=[4,4,1],
                convergence={'bands': -1})
CO = atoms.copy()
del CO[:2]
CO.set_calculator(calc_mol)
CO.get_potential_energy()
wf_u = [kpt.psit_nG[1] for kpt in calc_mol.wfs.kpt_u]
P_aui = [[kpt.P_ani[a][1] for kpt in calc_mol.wfs.kpt_u]
         for a in range(len(CO))]

calc_2 = GPAW(h=0.24,
              nbands=25,
              xc='PBE',
              spinpol = True,
              kpts=[4,4,1],
              convergence={'energy': 0.001,
                           'density': 0.001,
                           'eigenstates': 1.0e-7,
                           'bands': -3})
atoms.set_calculator(calc_2)
lumo = AEOrbital(calc_2, wf_u, P_aui, molecule=[2,3])
dscf_calculation(calc_2, [[1.0, lumo, 1]], atoms)
E_es2 = atoms.get_potential_energy()
equal(E_es2, E_gs + 3.55, 0.1)
