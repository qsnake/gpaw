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
        self.iter = 0

    
    def iterate(self, hamiltonian, kpt_u):
        if not self.initialized:
            hamiltonian.initialize()

        kpt = kpt_u[0]
        
        vt_G = hamiltonian.vt_sG[0]
        phi_mG = hamiltonian.phi_mG

        nao = hamiltonian.nao
        nbands = kpt.nbands
        H_mm = num.zeros((nao, nao), num.Float)
        r2k(0.5 * self.gd.dv, phi_mG, vt_G * phi_mG, 0.0, H_mm)
        
        for nucleus in self.nuclei:
            dH_ii = unpack(nucleus.H_sp[0])
            H_mm += num.dot(num.dot(nucleus.P_mi, dH_ii),
                            num.transpose(nucleus.P_mi))

        H_mm += hamiltonian.T_mm
        eps_n = num.zeros(nao, num.Float)
        diagonalize(H_mm, eps_n, hamiltonian.S_mm)
        kpt.C_nm = H_mm[0:nbands].copy()

        for nucleus in self.nuclei:
            nucleus.P_uni[0] = num.dot(kpt.C_nm, nucleus.P_mi)

        self.error = 100.0
        self.iter += 1
        if self.iter == 15:
            self.error = 1e-13
