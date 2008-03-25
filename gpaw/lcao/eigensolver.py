import numpy as npy
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize


class LCAO:
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self):
        self.lcao = True # paw object wants to know this
        self.initialized = False

    def initialize(self, paw):
        self.gd = paw.gd
        self.nuclei = paw.nuclei
        self.error = 0.0
        self.nspins = paw.nspins
        self.nkpts = paw.nkpts
        self.nbands = paw.nbands
        self.dtype = paw.dtype
        self.initialized = True

    def iterate(self, hamiltonian, kpt_u):
        if not hamiltonian.initialized:
            hamiltonian.initialize()
            nao = hamiltonian.nao
            self.eps_m = npy.empty(nao)
            self.S_mm = npy.empty((nao, nao), self.dtype)
            self.Vt_skmm = npy.empty((self.nspins, self.nkpts, nao, nao),
                                     self.dtype)

        hamiltonian.calculate_effective_potential_matrix(self.Vt_skmm)
        for kpt in kpt_u:
            self.iterate_one_k_point(hamiltonian, kpt)

    def iterate_one_k_point(self, hamiltonian, kpt):
        k = kpt.k
        s = kpt.s
        u = kpt.u
        
        H_mm = self.Vt_skmm[s, k]
        for nucleus in self.nuclei:
            dH_ii = unpack(nucleus.H_sp[s])
            P_mi = nucleus.P_kmi[k]
            H_mm += npy.dot(P_mi, npy.inner(dH_ii, P_mi).conj())

        H_mm += hamiltonian.T_kmm[k]

        self.S_mm[:] = hamiltonian.S_kmm[k]
        #error = diagonalize(self.S_mm, self.eps_m)
        #print self.eps_m, error
        self.eps_m[0] = 42
        error = diagonalize(H_mm, self.eps_m, self.S_mm)
        assert self.eps_m[0] != 42
        if error != 0:
            raise RuntimeError('Error code from dsyevd/zheevd: %d.' % error)
        kpt.C_nm = H_mm[0:self.nbands].copy()  # XXX
        kpt.eps_n[:] = self.eps_m[0:self.nbands]
        
        for nucleus in self.nuclei:
            nucleus.P_uni[u] = npy.dot(kpt.C_nm, nucleus.P_kmi[k])
 
