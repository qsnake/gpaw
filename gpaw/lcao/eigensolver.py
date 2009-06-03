import numpy as np
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import gemm
from gpaw.utilities import scalapack
from gpaw import sl_diagonalize, extra_parameters
import gpaw.mpi as mpi


class LCAO:
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self):
        self.error = 0.0
        self.linear_kpts = None
        self.eps_n = None
        self.S_MM = None
        self.H_MM = None
        self.timer = None
        self.mynbands = None
        self.band_comm = None
        self.world = None
        self.has_initialized = False # XXX

    def initialize(self, gd, band_comm, dtype, nao, mynbands, world):
        self.gd = gd
        self.band_comm = band_comm
        self.dtype = dtype
        self.nao = nao
        self.mynbands = mynbands
        self.world = world
        self.has_initialized = True # XXX
        assert self.H_MM is None # Right now we're not sure whether
        # this will work when reusing

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt):
        assert self.has_initialized
        s = kpt.s
        q = kpt.q
        if self.H_MM is None:
            nao = self.nao
            mynao = wfs.S_qMM.shape[1]
            self.eps_n = np.empty(nao)
            self.S_MM = np.empty((mynao, nao), self.dtype)
            self.H_MM = np.empty((mynao, nao), self.dtype)
            self.timer = wfs.timer
            #self.linear_dependence_check(wfs)

        self.timer.start('LCAO: potential matrix')

        wfs.basis_functions.calculate_potential_matrix(hamiltonian.vt_sG[s],
                                                       self.H_MM, q)

        self.timer.stop('LCAO: potential matrix')

        # Add atomic contribution
        #
        #           --   a     a  a*
        # H      += >   P    dH  P
        #  mu nu    --   mu i  ij nu j
        #           aij
        #
        Mstart = wfs.basis_functions.Mstart
        Mstop = wfs.basis_functions.Mstop
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(hamiltonian.dH_asp[a][s]), P_Mi.dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), P_Mi.dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            if Mstart != -1:
                P_Mi = P_Mi[Mstart:Mstop]
            gemm(1.0, dHP_iM, P_Mi, 1.0, self.H_MM)
        self.gd.comm.sum(self.H_MM)
        self.H_MM += wfs.T_qMM[q]

    def iterate(self, hamiltonian, wfs):
        for kpt in wfs.kpt_u:
            self.iterate_one_k_point(hamiltonian, wfs, kpt)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        self.calculate_hamiltonian_matrix(hamiltonian, wfs, kpt)
        self.S_MM[:] = wfs.S_qMM[kpt.q]

        b = self.band_comm.rank
        B = self.band_comm.size
        mynbands = self.mynbands
        n1 = b * mynbands
        n2 = n1 + mynbands

        if kpt.eps_n is None:
            kpt.eps_n = np.empty(mynbands)
            
        # Check and remove linear dependence for the current k-point
        #if 0:#k in self.linear_kpts:
        #    print '*Warning*: near linear dependence detected for k=%s' % k
        #    P_MM, p_M = wfs.lcao_hamiltonian.linear_kpts[k]
        #    eps_q, C2_nM = self.remove_linear_dependence(P_MM, p_M, H_MM)
        #    kpt.C_nM[:] = C2_nM[n1:n2]
        #    kpt.eps_n[:] = eps_q[n1:n2]
        #else:
        if sl_diagonalize:
            assert mpi.parallel
            assert scalapack()
            dsyev_zheev_string = 'LCAO: '+'pdsyevx/pzhegvx'
        else:
            dsyev_zheev_string = 'LCAO: '+'dsygv/zhegv'

        self.eps_n[0] = 42

        self.timer.start(dsyev_zheev_string)
        if extra_parameters.get('blacs'):
            import _gpaw
            nao = self.H_MM.shape[1]
            band_comm = self.band_comm
            B = band_comm.size
            c1 = self.world.new_communicator(np.arange(B) * self.gd.comm.size)
            d1 = _gpaw.blacs_create(c1, nao, nao, 1, band_comm.size,
                                    nao, -((-nao) // band_comm.size))
            n, m, nb = sl_diagonalize[:3]
            c2 = self.world.new_communicator(np.arange(n * m))
            d2 = _gpaw.blacs_create(c2, nao, nao, n, m, nb, nb)
            S_MM = _gpaw.scalapack_redist(self.S_MM, d1, d2)
            H_MM = _gpaw.scalapack_redist(self.H_MM, d1, d2)
            if self.world.rank < n * m:
                assert S_MM is not None
                self.eps_n[:], H_MM = _gpaw.scalapack_general_diagonalize(H_MM,
                                                                          S_MM,
                                                                          d2)
            else:
                assert S_MM is None
            d1b = _gpaw.blacs_create(c1, nao, nao, 1, band_comm.size,
                                     nao, mynbands)
            
            H_MM = _gpaw.scalapack_redist(H_MM, d2, d1b)
            if H_MM is not None:
                assert self.gd.comm.rank == 0
                kpt.C_nM[:] = H_MM[:, :mynbands].T
                self.band_comm.scatter(self.eps_n[:wfs.nbands], kpt.eps_n, 0)
            else:
                assert self.gd.comm.rank != 0
                
            self.gd.comm.broadcast(kpt.C_nM, 0)
            self.gd.comm.broadcast(kpt.eps_n, 0)

        elif sl_diagonalize:
            info = diagonalize(self.H_MM, self.eps_n, self.S_MM, root=0)
            if info != 0:
                raise RuntimeError('Failed to diagonalize: info=%d' % info)
        else:
            if self.gd.comm.rank == 0:
                info = diagonalize(self.H_MM, self.eps_n, self.S_MM)
                if info != 0:
                    raise RuntimeError('Failed to diagonalize: info=%d' %
                                       info)
        self.timer.stop(dsyev_zheev_string)

        if not extra_parameters.get('blacs'):
            if b == 0:
                self.gd.comm.broadcast(self.H_MM[:wfs.nbands], 0)
                self.gd.comm.broadcast(self.eps_n[:wfs.nbands], 0)
            self.band_comm.scatter(self.H_MM[:wfs.nbands], kpt.C_nM, 0)
            self.band_comm.scatter(self.eps_n[:wfs.nbands], kpt.eps_n, 0)

        assert kpt.eps_n[0] != 42

        for a, P_ni in kpt.P_ani.items():
            # ATLAS can't handle uninitialized output array:
            P_ni.fill(117)
            gemm(1.0, kpt.P_aMi[a], kpt.C_nM, 0.0, P_ni, 'n')

    def remove_linear_dependence(self, P_MM, p_M, H_MM):
        """Diagonalize H_MM with a reduced overlap matrix from which the
        linear dependent eigenvectors have been removed.

        The eigenvectors P_MM of the overlap matrix S_mm which correspond
        to eigenvalues p_M < thres are removed, thus producing a
        q-dimensional subspace. The hamiltonian H_MM is also transformed into
        H_qq and diagonalized. The transformation operator P_Mq looks like::

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

        s_q = np.extract(p_M > self.thres, p_M)
        S_qq = np.diag(s_q)
        S_qq = np.array(S_qq, self.dtype)
        q = len(s_q)
        p = self.nao - q
        P_Mq = P_MM[p:, :].T.conj()

        # Filling up the upper triangle
        for M in range(self.nao - 1):
            H_MM[M, m:] = H_MM[M:, M].conj()

        H_qq = np.dot(P_Mq.T.conj(), np.dot(H_MM, P_Mq))

        eps_q = np.zeros(q)

        if sl_diagonalize:
            assert mpi.parallel
            dsyev_zheev_string = 'LCAO: '+'pdsyevx/pzhegvx remove'
        else:
            dsyev_zheev_string = 'LCAO: '+'dsygv/zhegv remove'

        self.timer.start(dsyev_zheev_string)

        if sl_diagonalize:
            eps_q[0] = 42
            info = diagonalize(H_qq, eps_q, S_qq, root=0)
            assert eps_q[0] != 42
            if info != 0:
                raise RuntimeError('Failed to diagonalize: info=%d' % info)
        else:
            if self.comm.rank == 0:
                eps_q[0] = 42
                info = diagonalize(H_qq, eps_q, S_qq)
                assert eps_q[0] != 42
                if info != 0:
                    raise RuntimeError('Failed to diagonalize: info=%d' % info)

        self.timer.stop(dsyev_zheev_string)

        self.comm.broadcast(eps_q, 0)
        self.comm.broadcast(H_qq, 0)

        C_nq = H_qq
        C_nM = np.dot(C_nq, P_Mq.T.conj())
        return eps_q, C_nM

    def linear_dependence_check(self, wfs):
        # Near-linear dependence check. This is done by checking the
        # eigenvalues of the overlap matrix S_kmm. Eigenvalues close
        # to zero mean near-linear dependence in the basis-set.

        assert not sl_diagonalize
        self.linear_kpts = {}
        for k, S_MM in enumerate(wfs.S_kMM):
            P_MM = S_MM.copy()
            #P_mm = wfs.S_kMM[k].copy()
            p_M = np.empty(self.nao)

            dsyev_zheev_string = 'LCAO: '+'diagonalize-test'

            self.timer.start(dsyev_zheev_string)

            if self.comm.rank == 0:
                p_M[0] = 42
                info = diagonalize(P_MM, p_M)
                assert p_M[0] != 42
                if info != 0:
                    raise RuntimeError('Failed to diagonalize: info=%d' % info)

            self.timer.stop(dsyev_zheev_string)

            self.comm.broadcast(P_MM, 0)
            self.comm.broadcast(p_M, 0)

            self.thres = 1e-6
            if (p_M <= self.thres).any():
                self.linear_kpts[k] = (P_MM, p_M)

    def estimate_memory(self, mem):
        itemsize = np.array(1, self.dtype).itemsize
        mem.subnode('H, work [2*MM]', self.nao * self.nao * itemsize)

