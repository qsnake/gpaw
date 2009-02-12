from ase import *
from gpaw import *
from gpaw import dscf
from gpaw.utilities import equal

# Ground state calculation
#------------------------------------------------------------------

calc_gs = GPAW(nbands=8, h=0.2, xc='PBE', spinpol=True,
               convergence={'energy': 100,
                            'density': 100,
                            'eigenstates': 1.0e-9,
                            'bands': -1})

CO = molecule('CO')
CO.center(vacuum=3)
CO.set_calculator(calc_gs)
E_gs = CO.get_potential_energy()

'''Obtain the pseudowavefunctions and projector overlaps of the
   state which is to be occupied. n=5,6 is the 2pix and 2piy orbitals'''
wf_u = [kpt.psit_nG[5] for kpt in calc_gs.wfs.kpt_u]
P_aui = [[kpt.P_ani[a][5] for kpt in calc_gs.wfs.kpt_u]
          for a in range(len(CO))]

# Excited state calculation
#--------------------------------------------

calc_es = GPAW(nbands=8, h=0.2, xc='PBE', spinpol=True,
               convergence={'energy': 100,
                            'density': 100,
                            'eigenstates': 1.0e-9,
                            'bands': -1})

CO.set_calculator(calc_es)

lumo = dscf.MolecularOrbital(calc_es, molecule=[0,1], w=[[0,0,0,1],[0,0,0,-1]])
dscf.dscf_calculation(calc_es, [[1.0, lumo, 1]], CO)
E_es1 = CO.get_potential_energy()

lumo = dscf.AEOrbital(calc_es, wf_u, P_aui, molecule=[0,1])
dscf.dscf_calculation(calc_es, [[1.0, lumo, 1]], CO)
E_es2 = CO.get_potential_energy()

equal(E_es1, E_gs+5.8, 0.1)
equal(E_es1, E_es2, 0.001)

