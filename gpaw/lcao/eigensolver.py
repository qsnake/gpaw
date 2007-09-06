import Numeric as num
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize


class LCAO:
    def __init__(self, paw):
        self.gd = paw.gd
        self.nuclei = paw.nuclei
        self.initialized = False

    
    def iterate(self, hamiltonian, kpt_u):
        if not self.initialized:
            hamiltonian.initialize()

        vt_G = hamiltonian.vt_sG[0]
        nao = hamiltonian.nao
        phi_mG = hamiltonian.phi_mG
        H_mm = num.zeros((nao, nao), num.Float)
        r2k(0.5 * self.gd.dv, phi_mG, vt_G * phi_mG, 0.0, H_mm)
        
        for nucleus in self.nuclei:
            dH_ii = unpack(nucleus.H_sp[0])
            H_mm += num.dot(num.dot(nucleus.P_mi, dH_ii),
                            num.transpose(nucleus.P_mi))

        H_mm += hamiltonian.T_mm
        eps_n = num.zeros(nao, num.Float)
        diagonalize(H_mm, eps_n, hamiltonian.S_mm.copy())

        print H_mm
        print eps_n
        print ('Identity check: ',
               num.dot(H_mm, num.dot(hamiltonian.S_mm, num.transpose(H_mm))))
        self.error = 1e-20
