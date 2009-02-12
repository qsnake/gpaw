from ase import Atoms
from gpaw import GPAW
from gpaw.dscf import dscf_calculation, MolecularOrbital, AEOrbital
from gpaw.utilities import equal

atoms = Atoms(positions=[[0.,0.,5.],
                         [0.,0.,6.1],
                         [0.,0.,3.]],
              symbols='H2Al',
              cell=[3.,3.,9.1],
              pbc=[True,True,False])

calc = GPAW(h=0.24,
            nbands=6,
            xc='PBE',
            spinpol = True,
            kpts=[8,8,1],
            convergence={'energy': 0.001,
                         'density': 0.001,
                         'eigenstates': 1.0e-8,
                         'bands': -3})

atoms.set_calculator(calc)
E_gs = atoms.get_potential_energy()

# Dscf calculation based on the MolecularOrbital class
calc.set(nbands=12)
lumo = MolecularOrbital(calc, molecule=[0,1],
                        w=[[1.,0.,0.,0.],[-1.,0.,0.,0.]])
dscf_calculation(calc, [[1.0, lumo, 1]], atoms)
E_es1 = atoms.get_potential_energy()
equal(E_es1, E_gs + 6.8, 0.1)

# Dscf calculation based on the AEOrbital class
H2 = atoms.copy()
del H2[-1]
calc_mol = GPAW(h=0.24, xc='PBE', spinpol=True, kpts=[8, 8, 1])
H2.set_calculator(calc_mol)
H2.get_potential_energy()
wf_u = [kpt.psit_nG[1] for kpt in calc_mol.wfs.kpt_u]
P_aui = [[kpt.P_ani[a][1] for kpt in calc_mol.wfs.kpt_u]
         for a in range(len(H2))]
calc.set(nbands=12)
lumo = AEOrbital(calc, wf_u, P_aui, molecule=[0,1])
dscf_calculation(calc, [[1.0, lumo, 1]], atoms)
E_es2 = atoms.get_potential_energy()
equal(E_es2, E_gs + 3.9, 0.1)
