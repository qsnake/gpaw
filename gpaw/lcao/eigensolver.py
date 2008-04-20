import numpy as npy
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize


class LCAO:
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self):
        self.lcao = True
        self.initialized = False

    def initialize(self, paw):
        self.nuclei = paw.nuclei
        self.my_nuclei = paw.my_nuclei
        self.comm = paw.gd.comm
        self.error = 0.0
        self.nspins = paw.nspins
        self.nkpts = paw.nkpts
        self.nbands = paw.nbands
        self.dtype = paw.dtype
        self.initialized = True

    def iterate(self, hamiltonian, kpt_u):
        if not hamiltonian.lcao_initialized:
            hamiltonian.initialize_lcao()
            self.nao = hamiltonian.nao
            self.eps_m = npy.empty(self.nao)
            self.S_mm = npy.empty((self.nao, self.nao), self.dtype)
            self.Vt_skmm = npy.empty((self.nspins, self.nkpts,
                                      self.nao, self.nao), self.dtype)
            for kpt in kpt_u:
                kpt.C_nm = npy.empty((self.nbands, self.nao), self.dtype)

        hamiltonian.calculate_effective_potential_matrix(self.Vt_skmm)
        for kpt in kpt_u:
            self.iterate_one_k_point(hamiltonian, kpt)

    def iterate_one_k_point(self, hamiltonian, kpt):
        k = kpt.k
        s = kpt.s
        u = kpt.u

        H_mm = self.Vt_skmm[s, k]
        for nucleus in self.my_nuclei:
            dH_ii = unpack(nucleus.H_sp[s])
            P_mi = nucleus.P_kmi[k]
            H_mm += npy.dot(P_mi, npy.inner(dH_ii, P_mi).conj())

        self.comm.sum(H_mm)
        
        H_mm += hamiltonian.T_kmm[k]

        self.S_mm[:] = hamiltonian.S_kmm[k]
        
        # Near-linear dependence check
        p_m = npy.empty(self.nao)
        P_mm = self.S_mm.copy()
        diagonalize(P_mm, p_m)
        thres = 1e-6
        if (p_m <= thres).any():
            print '*Warning*: near linear dependence detected for k=%s' %k
            eps_q, C_nm = self.remove_linear_dependence(P_mm, p_m, H_mm, thres)
            kpt.C_nm[:] = C_nm[0:self.nbands]
            kpt.eps_n[:] = eps_q[0:self.nbands]
        else:
            self.eps_m[0] = 42
            errorcode = diagonalize(H_mm, self.eps_m, self.S_mm)
            assert self.eps_m[0] != 42
            if errorcode != 0:
                raise RuntimeError('Error code from dsyevd/zheevd: %d.' %
                                   errorcode)

            kpt.C_nm[:] = H_mm[0:self.nbands]
            kpt.eps_n[:] = self.eps_m[0:self.nbands]

        for nucleus in self.my_nuclei:
            nucleus.P_uni[u] = npy.dot(kpt.C_nm, nucleus.P_kmi[k])

    def remove_linear_dependence(self, P_mm, p_m, H_mm, thres):
        """Diagonalize H_mm with a reduced overlap matrix from which the
        linear dependent eigenvectors have been removed.

        The eigenvectors P_mm of the overlap matrix S_mm which correspond
        to eigenvalues p_m < thres are removed, thus producing a
        q-dimensional subspace. The hamiltonian H_mm is also transformed into
        H_qq and diagonalized. The transformation operator P_mq looks like

                ------------m--------- ...
                ---p---  ------q------ ...
               +---------------------------
               |
           |   |
           |   |
           m   |
           |   |   
           |   |   
             . |
             .        

        """

        s_q = npy.extract(p_m > thres, p_m)
        S_qq = npy.diag(s_q)
        S_qq = npy.array(S_qq, self.dtype)
        q = len(s_q)
        p = self.nao - q
        P_mq = P_mm[p:,:].T.conj()
        
        # Filling up the upper triangle
        for m in range(self.nao - 1):
            H_mm[m, m:] = H_mm[m:, m].conj()
        
        H_qq = npy.dot(P_mq.T.conj(), npy.dot(H_mm, P_mq))
        
        eps_q = npy.zeros(q)
        eps_q[0] = 42
        errorcode = diagonalize(H_qq, eps_q, S_qq)
        C_nq = H_qq
        assert eps_q[0] != 42
        if errorcode != 0:
            raise RuntimeError('Error code from dsyevd/zheevd: %d.' %
                               errorcode)
        C_nm = npy.dot(C_nq, P_mq.T.conj())
        return eps_q, C_nm 
