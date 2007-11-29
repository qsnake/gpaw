import Numeric as num
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize


class LCAO:
    """Eigensolver for LCAO-basis calculation"""
	
    def __init__(self):		
    	pass	
    
    def initialize(self, paw):
        self.gd = paw.gd
        self.nuclei = paw.nuclei
        self.initialized = False
        self.error = 0.0
        
    def iterate(self, hamiltonian, kpt_u):
        if not self.initialized:
            hamiltonian.initialize(self.gd.domain.cell_c)
            self.initialized = True

        for kpt in kpt_u:
            self.iterate_one_k_point(hamiltonian, kpt)

    def iterate_one_k_point(self, hamiltonian, kpt):
        k = kpt.k
        s = kpt.s
        u = kpt.u
        
        vt_G = hamiltonian.vt_sG[s]

        nao = hamiltonian.nao
        nbands = kpt.nbands
        H_mm = num.zeros((nao, nao), num.Complex)   #Changed to complex!
        
        V_mm = num.zeros((nao, nao), num.Float)
        hamiltonian.calculate_effective_potential_matrix(V_mm)
        H_mm += V_mm

        for nucleus in self.nuclei:
            dH_ii = unpack(nucleus.H_sp[s])
            H_mm += num.dot(num.dot(nucleus.P_kmi[k], dH_ii),
                            num.transpose(nucleus.P_kmi[k]))

        H_mm += hamiltonian.T_kmm[k]
        eps_n = num.zeros(nao, num.Float)
        diagonalize(H_mm, eps_n, hamiltonian.S_kmm[k].copy())
        kpt.C_nm = H_mm[0:nbands].copy()
        #print kpt.C_nm
        kpt.eps_n[:] = eps_n[0:nbands]
        
        for nucleus in self.nuclei:
            nucleus.P_uni[u] = num.dot(kpt.C_nm, nucleus.P_kmi[k]).real # XXX
 
