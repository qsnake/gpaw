import Numeric as num
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize


class LCAO:
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self, paw):
        self.gd = paw.gd
        self.nuclei = paw.nuclei
        self.initialized = False
        self.error = 0.0
    
    def iterate(self, hamiltonian, kpt_u):
        if not self.initialized:
            hamiltonian.initialize()
            self.initialized = True
            
        kpt = kpt_u[0]
        
        vt_G = hamiltonian.vt_sG[0]

        nao = hamiltonian.nao
        nbands = kpt.nbands
        H_mm = num.zeros((nao, nao), num.Float)

        #phi_mG = hamiltonian.phi_mG
        #r2k(0.5 * self.gd.dv, phi_mG, vt_G * phi_mG, 0.0, H_mm)

        if 1:
            #print H_mm
            hamiltonian.calculate_effective_potential_matrix(H_mm)
            
        for nucleus in self.nuclei:
            dH_ii = unpack(nucleus.H_sp[0])
            H_mm += num.dot(num.dot(nucleus.P_mi, dH_ii),
                            num.transpose(nucleus.P_mi))

        H_mm += hamiltonian.T_mm
        eps_n = num.zeros(nao, num.Float)
        diagonalize(H_mm, eps_n, hamiltonian.S_mm.copy())
        kpt.C_nm = H_mm[0:nbands].copy()
        kpt.eps_n[:] = eps_n[0:nbands]
        
        for nucleus in self.nuclei:
            nucleus.P_uni[0] = num.dot(kpt.C_nm, nucleus.P_mi)
