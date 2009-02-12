# Copyright (C) 2008 CAMd
# Please see the accompanying LICENSE file for further information.

import numpy as np
from gpaw.utilities.blas import rk, r2k, gemm


class Operator:
    nblocks = 1
    """Base class for overlap and hamiltonian operators."""
    def __init__(self, band_comm, gd, nblocks=None):
        self.band_comm = band_comm
        self.domain_comm = gd.comm
        self.gd = gd
        self.work1_xG = None
        self.work2_xG = None
        self.A_qnn = None
        self.A_nn = None
        if nblocks is not None:
            self.nblocks = nblocks

    def allocate_work_arrays(self, mynbands, dtype):
        ngroups = self.band_comm.size
        if ngroups == 1 and self.nblocks == 1:
            self.work1_xG = self.gd.empty(mynbands, dtype)
        else:
            assert mynbands % self.nblocks == 0
            X = mynbands // self.nblocks
            if self.gd.n_c.prod() % self.nblocks != 0:
                X += 1
            self.work1_xG = self.gd.empty(X, dtype)
            self.work2_xG = self.gd.empty(X, dtype)
            if ngroups > 1:
                self.A_qnn = np.empty((ngroups // 2 + 1, mynbands, mynbands),
                                      dtype)
        nbands = ngroups * mynbands
        self.A_nn = np.empty((nbands, nbands), dtype)

    def calculate_matrix_elements(self, psit_nG, P_ani, A, dA_aii):
        """Calculate matrix elements for A-operator.

        Results will be put in the *A_nn* array::

                                  ___
                    ~   ^  ~     \     ~   ~a    a   ~a  ~
           A    = <psi |A|psi > + )  <psi |p > dA   <p |psi >
            nn'       n      n'  /___    n  i    ii'  i'   n'
                                  aii'

        """

        bcomm = self.band_comm
        B = bcomm.size
        J = self.nblocks
        N = len(psit_nG)  # mynbands
        dv = self.gd.dv
        
        if self.work1_xG is None:
            self.allocate_work_arrays(N, psit_nG.dtype)

        A_NN = self.A_nn
        
        if B == 1 and J == 1:
            # Simple case:
            Apsit_nG = A(psit_nG)
            A_NN.fill(0.0)  # Some BLAS libraries needs this!
            if Apsit_nG is psit_nG:
                rk(dv, psit_nG, 0.0, A_NN)
            else:
                r2k(0.5 * dv, psit_nG, Apsit_nG, 0.0, A_NN)
            for a, P_ni in P_ani.items():
                gemm(1.0, P_ni, np.dot(P_ni, dA_aii[a]), 1.0, A_NN, 'c')
            self.domain_comm.sum(A_NN, 0)
            return A_NN
        
        # Now is gets nasty!  We parallelize over B groups of bands
        # and each group is blocked in J blocks.

        Q = B // 2 + 1
        rank = bcomm.rank
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
                    sreq = bcomm.send(sbuf_mG, rankm, 11, False)
                    rreq = bcomm.receive(rbuf_mG, rankp, 11, False)
                if j == J - 1 and P_ani:
                    if q == 0:
                        sbuf_In = np.concatenate([np.dot(P_ni, dA_aii[a]).T
                                                  for a, P_ni in P_ani.items()])
                        if B > 1:
                            rbuf_In = np.empty_like(sbuf_In)
                    if q < Q - 1:
                        sreq2 = bcomm.send(sbuf_In, rankm, 31, False)
                        rreq2 = bcomm.receive(rbuf_In, rankp, 31, False)

                if q > 0:
                    gemm(dv, psit_nG, sbuf_mG, 0.0, A_mn, 'c')
                else:
                    # We only need the lower part:
                    if j == 0:
                        # Important special-cases:
                        if sbuf_mG is psit_mG:
                            rk(dv, psit_mG, 0.0, A_mn[:, :M])
                        else:
                            r2k(0.5 * dv, psit_mG, sbuf_mG, 0.0, A_mn[:, :M])
                    else:
                        gemm(dv, psit_nG[:n2], sbuf_mG, 0.0, A_mn[:, :n2], 'c')
                        
                if j == J - 1 and P_ani:
                    I1 = 0
                    for P_ni in P_ani.values():
                        I2 = I1 + P_ni.shape[1]
                        gemm(1.0, P_ni, sbuf_In[I1:I2].T.copy(), 1.0, A_nn, 'c')
                        I1 = I2

                if q == Q - 1:
                    break

                if j == J - 1 and P_ani:
                    bcomm.wait(sreq2)
                    bcomm.wait(rreq2)
                    sbuf_In, rbuf_In = rbuf_In, sbuf_In

                bcomm.wait(sreq)
                bcomm.wait(rreq)

                if q == 0:
                    sbuf_mG = self.work1_xG[:M]
                sbuf_mG, rbuf_mG = rbuf_mG, sbuf_mG

        self.domain_comm.sum(A_qnn, 0)

        if B == 1:
            return A_NN

        A_bnbn = A_NN.reshape((B, N, B, N))
        if self.domain_comm.rank == 0:
            if rank == 0:
                A_bnbn[:Q, :, 0] = A_qnn
                for q1 in range(1, B):
                    bcomm.receive(A_qnn, q1, 13)
                    for q2 in range(Q):
                        if q1 + q2 < B:
                            A_bnbn[q1 + q2, :, q1] = A_qnn[q2]
                        else:
                            A_bnbn[q1, :, q1 + q2 - B] = A_qnn[q2].T
            else:
                bcomm.send(A_qnn, 0, 13)
        return A_NN
        
    def matrix_multiply(self, C_nn, psit_nG, P_ani=None):
        """Calculate new linear combinations of wave functions.

        ::

                     __                                __
            ~       \       ~           ~a  ~         \       ~a  ~
           psi  <--  ) C   psi    and  <p |psi >  <--  ) C   <p |psi >
              n     /__ nn'   n'         i    n       /__ nn'  i    n'
                     n'                                n'

        """

        bcomm = self.band_comm
        B = bcomm.size
        J = self.nblocks

        if B == 1 and J == 1:
            # Simple case:
            newpsit_nG = self.work1_xG
            gemm(1.0, psit_nG, C_nn, 0.0, newpsit_nG)
            self.work1_xG = psit_nG
            if P_ani:
                for P_ni in P_ani.values():
                    gemm(1.0, P_ni.copy(), C_nn, 0.0, P_ni)
            return newpsit_nG
        
        # Now is gets nasty!  We parallelize over B groups of bands
        # and each group is blocked in J blocks.

        rank = bcomm.rank
        rankm = (rank - 1) % B
        rankp = (rank + 1) % B
        N = len(psit_nG)       # mynbands
        shape = psit_nG.shape
        psit_nG = psit_nG.reshape(N, -1)
        G = psit_nG.shape[1]   # number of grid-points
        g = G // J
        if g * J < G:
            g += 1

        C_bnbn = C_nn.reshape((B, N, B, N))

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
                        sreq2 = bcomm.send(sbuf_In, rankm, 31, False)
                        rreq2 = bcomm.receive(rbuf_In, rankp, 31, False)
                if q < B - 1:
                    sreq = bcomm.send(sbuf_ng, rankm, 61, False)
                    rreq = bcomm.receive(rbuf_ng, rankp, 61, False)
                C_mm = C_bnbn[rank, :, (rank + q) % B]
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
                    bcomm.wait(sreq2)
                    bcomm.wait(rreq2)
                    sbuf_In, rbuf_In = rbuf_In, sbuf_In

                beta = 1.0
                bcomm.wait(rreq)
                bcomm.wait(sreq)
                sbuf_ng, rbuf_ng = rbuf_ng, sbuf_ng

        psit_nG.shape = shape
        return psit_nG
