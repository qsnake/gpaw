# Copyright (C) 2008 CAMd
# Please see the accompanying LICENSE file for further information.

import numpy as np
from gpaw.utilities.blas import rk, r2k, gemm


class Operator:
    nblocks = 1
    """Base class for overlap and hamiltonian operators."""
    def __init__(self, bd, gd, nblocks=None):
        self.bd = bd
        self.gd = gd
        self.work1_xG = None
        self.work2_xG = None
        self.A_qnn = None
        self.A_nn = None
        if nblocks is not None:
            self.nblocks = nblocks

    def allocate_work_arrays(self, mynbands, dtype):
        ngroups = self.bd.comm.size
        if ngroups == 1 and self.nblocks == 1:
            self.work1_xG = self.gd.zeros(mynbands, dtype)
        else:
            assert mynbands % self.nblocks == 0
            X = mynbands // self.nblocks
            if self.gd.n_c.prod() % self.nblocks != 0:
                X += self.nblocks
            self.work1_xG = self.gd.zeros(X, dtype)
            self.work2_xG = self.gd.zeros(X, dtype)
            if ngroups > 1:
                self.A_qnn = np.zeros((ngroups // 2 + 1, mynbands, mynbands),
                                      dtype)
        nbands = ngroups * mynbands
        self.A_nn = np.zeros((nbands, nbands), dtype)

    def estimate_memory(self, mem, mynbands, dtype):
        ngroups = self.bd.comm.size
        gdbytes = self.gd.bytecount(dtype)
        # Code semipasted from allocate_work_arrays
        if ngroups == 1 and self.nblocks == 1:
            mem.subnode('work_xG', mynbands * gdbytes)
        else:
            X = mynbands // self.nblocks
            if self.gd.n_c.prod() % self.nblocks != 0:
                X += 1
            mem.subnode('2 work_xG', 2 * X * gdbytes)
            if ngroups > 1:
                count = (ngroups // 2 + 1) * mynbands**2
                mem.subnode('A_qnn', count * mem.itemsize[dtype])

    def calculate_matrix_elements(self, psit_nG, P_ani, A, dA):
        """Calculate matrix elements for A-operator.

        Results will be put in the *A_nn* array::

                                  ___
                    ~   ^  ~     \     ~   ~a    a   ~a  ~
           A    = <psi |A|psi > + )  <psi |p > dA   <p |psi >
            nn'       n      n'  /___    n  i    ii'  i'   n'
                                  aii'

        Fills in the lower part of *A_nn*, but only on domain and band masters.


        Parameters:

        psit_nG: ndarray
            Set of vectors in which the matrix elements are evaulated.
        P_ani: dict
            Dictionary of projector overlap integrals P_ni = <p_i | psit_nG>.
        A: function
            Functional form of the operator A which works on psit_nG.
            Must accept and return a ndarray of the same shape as psit_nG.
        dA: dict or function
            Dictionary of atomic matrix elements dA_ii = <phi_i | A | phi_i >
            or functional form of the operator which works on | phi_i >.
            Must accept atomic index a and P_ni and return a ndarray with the
            same shape as P_ni, thus representing P_ni multiplied by dA_ii.

        """

        #TODO eliminate need for A_nn.conj() by extended gemm(..., 'c')?

        band_comm = self.bd.comm
        domain_comm = self.gd.comm
        B = band_comm.size
        J = self.nblocks
        N = len(psit_nG)  # mynbands
        dv = self.gd.dv
        
        if self.work1_xG is None:
            self.allocate_work_arrays(N, psit_nG.dtype)

        A_NN = self.A_nn

        dAP_ani = {}
        for a, P_ni in P_ani.items():
            if callable(dA):
                dAP_ani[a] = dA(a, P_ni)
            else:
                # dA denotes dA_aii as usual
                dAP_ani[a] = np.dot(P_ni, dA[a])
        
        if B == 1 and J == 1:
            # Simple case:
            Apsit_nG = A(psit_nG)
            if Apsit_nG is psit_nG:
                rk(dv, psit_nG, 0.0, A_NN)
            else:
                r2k(0.5 * dv, psit_nG, Apsit_nG, 0.0, A_NN)
            for a, P_ni in P_ani.items():
                gemm(1.0, P_ni, dAP_ani[a], 1.0, A_NN, 'c')
            domain_comm.sum(A_NN, 0)
            return A_NN
        
        # Now it gets nasty!  We parallelize over B groups of bands
        # and each group is blocked in J blocks.

        Q = B // 2 + 1
        rank = band_comm.rank
        rankm = (rank - 1) % B
        rankp = (rank + 1) % B
        M = N // J
        
        if B == 1:
            A_qnn = A_NN.reshape((1, N, N))
        else:
            A_qnn = self.A_qnn

        for j in range(J):
            n1 = j * M
            n2 = n1 + M
            psit_mG = psit_nG[n1:n2]
            sbuf_mG = A(psit_mG)
            rbuf_mG = self.work2_xG[:M]
            for q in range(Q):
                A_nn = A_qnn[q]
                A_mn = A_nn[n1:n2]
                if q < Q - 1:
                    sreq = band_comm.send(sbuf_mG, rankm, 11, False)
                    rreq = band_comm.receive(rbuf_mG, rankp, 11, False)
                if j == J - 1 and P_ani:
                    if q == 0:
                        sbuf_In = np.concatenate([dAP_ani[a].T
                                                  for a, P_ni in P_ani.items()])
                        if B > 1:
                            rbuf_In = np.empty_like(sbuf_In)
                    if q < Q - 1:
                        sreq2 = band_comm.send(sbuf_In, rankm, 31, False)
                        rreq2 = band_comm.receive(rbuf_In, rankp, 31, False)

                if q == 0 and not self.bd.strided:
                    # We only need the lower part:
                    if j == 0:
                        # Important special-cases:
                        if sbuf_mG is psit_mG:
                            rk(dv, psit_mG, 0.0, A_mn[:, :M])
                        else:
                            r2k(0.5 * dv, psit_mG, sbuf_mG, 0.0, A_mn[:, :M])
                    else:
                        gemm(dv, psit_nG[:n2], sbuf_mG, 0.0, A_mn[:, :n2], 'c')
                else:
                    gemm(dv, psit_nG, sbuf_mG, 0.0, A_mn, 'c')

                if j == J - 1 and P_ani:
                    I1 = 0
                    for P_ni in P_ani.values():
                        I2 = I1 + P_ni.shape[1]
                        gemm(1.0, P_ni, sbuf_In[I1:I2].T.copy(), 1.0, A_nn, 'c')
                        I1 = I2

                if q == Q - 1:
                    break

                if j == J - 1 and P_ani:
                    band_comm.wait(sreq2)
                    band_comm.wait(rreq2)
                    sbuf_In, rbuf_In = rbuf_In, sbuf_In

                band_comm.wait(sreq)
                band_comm.wait(rreq)

                if q == 0:
                    sbuf_mG = self.work1_xG[:M]
                sbuf_mG, rbuf_mG = rbuf_mG, sbuf_mG

        domain_comm.sum(A_qnn, 0)

        if B == 1:
            return A_NN

        if domain_comm.rank == 0:
            self.bd.matrix_assembly(A_qnn, A_NN)

        return A_NN
        
    def matrix_multiply(self, C_NN, psit_nG, P_ani=None):
        """Calculate new linear combinations of wave functions.

        ::

                     __                                __
            ~       \       ~           ~a  ~         \       ~a  ~
           psi  <--  ) C   psi    and  <p |psi >  <--  ) C   <p |psi >
              n     /__ nn'   n'         i    n       /__ nn'  i    n'
                     n'                                n'

        """

        band_comm = self.bd.comm
        B = band_comm.size
        J = self.nblocks

        if B == 1 and J == 1:
            # Simple case:
            newpsit_nG = self.work1_xG
            gemm(1.0, psit_nG, C_NN, 0.0, newpsit_nG)
            self.work1_xG = psit_nG
            if P_ani:
                for P_ni in P_ani.values():
                    gemm(1.0, P_ni.copy(), C_NN, 0.0, P_ni)
            return newpsit_nG
        
        # Now it gets nasty!  We parallelize over B groups of bands
        # and each group is blocked in J blocks.

        rank = band_comm.rank
        rankm = (rank - 1) % B
        rankp = (rank + 1) % B
        N = len(psit_nG)       # mynbands
        shape = psit_nG.shape
        psit_nG = psit_nG.reshape(N, -1)
        G = psit_nG.shape[1]   # number of grid-points
        g = G // J
        if g * J < G:
            g += 1

        for j in range(J):
            G1 = j * g
            G2 = G1 + g
            if G2 > G:
                G2 = G
                g = G2 - G1
            sbuf_ng = self.work1_xG.reshape(-1)[:N * g].reshape(N, g)
            rbuf_ng = self.work2_xG.reshape(-1)[:N * g].reshape(N, g)
            sbuf_ng[:] = psit_nG[:, G1:G2]
            if P_ani:
                sbuf_In = np.concatenate([P_ni.T for P_ni in P_ani.values()])
            beta = 0.0
            for q in range(B):
                if j == 0 and P_ani:
                    if B > 1:
                        rbuf_In = np.empty_like(sbuf_In)
                    if q < B - 1:
                        sreq2 = band_comm.send(sbuf_In, rankm, 31, False)
                        rreq2 = band_comm.receive(rbuf_In, rankp, 31, False)
                if q < B - 1:
                    sreq = band_comm.send(sbuf_ng, rankm, 61, False)
                    rreq = band_comm.receive(rbuf_ng, rankp, 61, False)
                C_mm = self.bd.extract_block(C_NN, rank, (rank + q) % B)
                gemm(1.0, sbuf_ng, C_mm, beta, psit_nG[:, G1:G2])
                if j == 0 and P_ani:
                    I1 = 0
                    for P_ni in P_ani.values():
                        I2 = I1 + P_ni.shape[1]
                        gemm(1.0, sbuf_In[I1:I2].T.copy(), C_mm, beta, P_ni)
                        I1 = I2

                if q == B - 1:
                    break
                
                if j == 0 and P_ani:
                    band_comm.wait(sreq2)
                    band_comm.wait(rreq2)
                    sbuf_In, rbuf_In = rbuf_In, sbuf_In

                beta = 1.0
                band_comm.wait(rreq)
                band_comm.wait(sreq)
                sbuf_ng, rbuf_ng = rbuf_ng, sbuf_ng

        psit_nG.shape = shape
        return psit_nG
