# -*- coding: utf-8 -*-
# Copyright (C) 2008  CSC Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.

"""This module defines an Overlap operator.

The module defines an overlap operator and implements overlap-related
functions.

"""
import sys

import numpy as npy
from gpaw.mpi import run, parallel
from gpaw.utilities.complex import cc
from gpaw.utilities.blas import rk, gemm
from gpaw.utilities.lapack import inverse_cholesky
from gpaw.utilities import swap
from gpaw.eigensolvers.eigensolver import blocked_matrix_multiply
from gpaw.utilities import scalapack
from gpaw import sl_inverse_cholesky


class Overlap:
    """Overlap operator class.

    Attributes
    ==========
    nuclei: list of Nucleus objects
        All nuclei.
    my_nuclei: list of Nucleus objects
        Nuclei with centers in the current domain.
    pt_nuclei: list of Nucleus objects
        Nuclei with projector functions overlapping the current domain.
    dtype: type object
        Numerical type of operator (float/complex)
    """

    def __init__(self, paw):
        """Create the Overlap operator."""

        self.my_nuclei = paw.my_nuclei
        self.pt_nuclei = paw.pt_nuclei
        self.nuclei = paw.nuclei
        self.gd = paw.gd
        self.dtype = paw.dtype
        self.timer = paw.timer
        self.comm = paw.gd.comm
        self.band_comm = paw.band_comm
        self.work_nn = None
        self.S_pnn = None
        self.work_In = None
        self.work2_In = None
        self.big_work_arrays = paw.big_work_arrays

    def apply(self, a_nG, b_nG, kpt, calculate_P_uni=True):
        """Apply the overlap operator to a set of vectors.

        Parameters
        ==========
        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting S times a_nG vectors.
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_uni: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_uni are used

        """

        self.timer.start('Apply overlap')
        b_nG[:] = a_nG

        run([nucleus.apply_overlap(a_nG, b_nG, kpt, calculate_P_uni)
             for nucleus in self.pt_nuclei])

        self.timer.stop('Apply overlap')

    def apply_inverse(self, a_nG, b_nG, kpt):
        """Apply approximative inverse overlap operator to wave functions."""

        b_nG[:] = a_nG

        for nucleus in self.pt_nuclei:
            # Apply the non-local part:
            nucleus.apply_inverse_overlap(a_nG, b_nG, kpt.k)


    def orthonormalize(self, kpt, psit_nG=None, work_nG=None,
                       calculate_projections=True):
        """Orthonormalizes the vectors a_nG with respect to the overlap.

        First, a Cholesky factorization C is done for the overlap
        matrix S_nn = <a_nG | S | a_nG> = C*_nn C_nn Cholesky matrix C
        is inverted and orthonormal vectors a_nG' are obtained as::

          psit_nG' = inv(C_nn) psit_nG
                    __
           ~   _   \    -1   ~   _
          psi (r) = )  C    psi (r)
             n     /__  nm     m
                    m

        Parameters
        ----------
        psit_nG: ndarray, input/output
            On input the set of vectors to orthonormalize,
            on output the overlap-orthonormalized vectors.
        kpt: KPoint object:
            k-point object from kpoint.py.
        work_nG: ndarray
            Optional work array for overlap matrix times psit_nG.
        work_nn: ndarray
            Optional work array for overlap matrix.

        """

        self.timer.start('Orthonormalize')

        if calculate_projections:
            assert psit_nG is None
            run([nucleus.calculate_projections(kpt)
                 for nucleus in self.pt_nuclei])

        if psit_nG is None:
            psit_nG = kpt.psit_nG

        nmybands = len(psit_nG)
        nbands = nmybands * self.band_comm.size

        # Allocate work arrays if necessary:
        if self.work_nn is None:
            self.work_nn = npy.zeros((nbands, nbands), self.dtype)

        S_nn = self.work_nn

        if work_nG is None:
            if 'work_nG' in self.big_work_arrays:
                work_nG = self.big_work_arrays['work_nG']
            else:
                work_nG = npy.zeros_like(psit_nG)

        # Construct the overlap matrix:
        self.calculate_overlap_matrix(psit_nG, work_nG, kpt, S_nn)

        self.timer.start('Orthonormalize: inverse_cholesky')
        if sl_inverse_cholesky:
            assert parallel and scalapack()
            info = inverse_cholesky(S_nn, kpt.root)
        else:
            if self.comm.rank == kpt.root and self.band_comm.rank == 0:
                info = inverse_cholesky(S_nn)
        if sl_inverse_cholesky:
            if info != 0:
                raise RuntimeError('Orthogonalization failed!')
        else:
            if self.comm.rank == kpt.root and self.band_comm.rank == 0:
                if info != 0:
                    raise RuntimeError('Orthogonalization failed!')
        self.timer.stop('Orthonormalize: inverse_cholesky')

        # S_nn has been overwriten - let's call it something different:
        C_nn = S_nn
        del S_nn

        if self.band_comm.rank == 0:
            self.comm.broadcast(C_nn, kpt.root)
        self.band_comm.broadcast(C_nn, 0)

        self.matrix_multiplication(kpt, C_nn)

        self.timer.stop('Orthonormalize')

    def matrix_multiplication(self, kpt, C_nn):
        band_comm = self.band_comm
        size = band_comm.size
        psit_nG =  kpt.psit_nG
        work_nG = self.big_work_arrays['work_nG']
        nmybands = len(psit_nG)
        if size == 1:
            if psit_nG.shape != work_nG.shape:
                blocked_matrix_multiply(psit_nG, C_nn, work_nG)
            else:
                gemm(1.0, psit_nG, C_nn, 0.0, work_nG)

                kpt.psit_nG = work_nG

                if work_nG is self.big_work_arrays.get('work_nG'):
                    self.big_work_arrays['work_nG'] = psit_nG

            for nucleus in self.my_nuclei:
                P_ni = nucleus.P_uni[kpt.u]
                gemm(1.0, P_ni.copy(), C_nn, 0.0, P_ni)

            return

        # Parallelize over bands:
        C_bnbn = C_nn.reshape((size, nmybands, size, nmybands))
        work2_nG = self.big_work_arrays['work2_nG']

        rank = band_comm.rank

        beta = 0.0
        for p in range(size - 1):
            sreq = band_comm.send(psit_nG, (rank - 1) % size, 61, False)
            rreq = band_comm.receive(work_nG, (rank + 1) % size, 61, False)
            gemm(1.0, psit_nG, C_bnbn[rank, :, (rank + p) % size],
                 beta, work2_nG)
            beta = 1.0
            band_comm.wait(rreq)
            band_comm.wait(sreq)
            psit_nG, work_nG = work_nG, psit_nG

        gemm(1.0, psit_nG, C_bnbn[rank, :, rank - 1], 1.0, work2_nG)

        if size % 2 == 0:
            psit_nG, work_nG = work_nG, psit_nG

        kpt.psit_nG = work2_nG
        self.big_work_arrays['work2_nG'] = psit_nG

        run([nucleus.calculate_projections(kpt)
             for nucleus in self.pt_nuclei])

    def calculate_overlap_matrix(self, psit_nG, work_nG, kpt, S_nn):
        band_comm = self.band_comm
        size = band_comm.size
        #S_nn.fill(10.0)
        if size == 1:
            rk(self.gd.dv, psit_nG, 0.0, S_nn)
            for nucleus in self.my_nuclei:
                P_ni = nucleus.P_uni[kpt.u]
                dO_ii = nucleus.setup.O_ii
                #???????gemm(1.0, P_ni, npy.dot(P_ni, dO_ii), 1.0, S_nn, 't')
                S_nn += npy.dot(P_ni, cc(npy.inner(nucleus.setup.O_ii, P_ni)))

            self.comm.sum(S_nn, kpt.root)
            return

        #assert size % 2 == 1
        np = size // 2 + 1
        rank = band_comm.rank
        nmybands = len(work_nG)

        nI = 0
        for nucleus in self.my_nuclei:
            nI += nucleus.get_number_of_partial_waves()

        if self.work_In is None or len(self.work_In) != nI:
            self.work_In = npy.empty((nI, nmybands), psit_nG.dtype)
            self.work2_In = npy.empty((nI, nmybands), psit_nG.dtype)
        work_In = self.work_In
        work2_In = self.work2_In

        if 'work2_nG' not in self.big_work_arrays:
            self.big_work_arrays['work2_nG'] = npy.empty_like(psit_nG)
        work2_nG = self.big_work_arrays['work2_nG']

        if self.S_pnn is None:
            self.S_pnn = npy.empty((np, nmybands, nmybands), psit_nG.dtype)
        S_pnn = self.S_pnn

        I1 = 0
        for nucleus in self.my_nuclei:
            ni = nucleus.get_number_of_partial_waves()
            I2 = I1 + ni
            P_ni = nucleus.P_uni[kpt.u]
            dO_ii = nucleus.setup.O_ii
            work_In[I1:I2] = npy.inner(dO_ii, P_ni).conj()
            I1 = I2

        for p in range(np - 1):
            if p == 0:
                sreq = band_comm.send(psit_nG, (rank - 1) % size, 11, False)
            else:
                sreq = band_comm.send(work_nG, (rank - 1) % size, 11, False)
            sreq2 = band_comm.send(work_In, (rank - 1) % size, 31, False)
            rreq = band_comm.receive(work2_nG, (rank + 1) % size, 11, False)
            rreq2 = band_comm.receive(work2_In, (rank + 1) % size, 31, False)

            if p == 0:
                rk(self.gd.dv, psit_nG, 0.0, S_pnn[0])
            else:
                gemm(self.gd.dv, psit_nG, work_nG, 0.0, S_pnn[p], 'c')

            I1 = 0
            for nucleus in self.my_nuclei:
                ni = nucleus.get_number_of_partial_waves()
                I2 = I1 + ni
                P_ni = nucleus.P_uni[kpt.u]
                S_pnn[p] += npy.dot(P_ni, work_In[I1:I2]).T
                I1 = I2

            band_comm.wait(sreq)
            band_comm.wait(sreq2)
            band_comm.wait(rreq)
            band_comm.wait(rreq2)

            work_nG, work2_nG = work2_nG, work_nG
            work_In, work2_In = work2_In, work_In

        gemm(self.gd.dv, psit_nG, work_nG, 0.0, S_pnn[-1], 'c')

        I1 = 0
        for nucleus in self.my_nuclei:
            ni = nucleus.get_number_of_partial_waves()
            I2 = I1 + ni
            P_ni = nucleus.P_uni[kpt.u]
            S_pnn[-1] += npy.dot(P_ni, work_In[I1:I2]).T
            I1 = I2

        self.comm.sum(S_pnn, kpt.root)

        S_bnbn = S_nn.reshape((size, nmybands, size, nmybands))
        if self.comm.rank == kpt.root:
            if rank == 0:
                S_bnbn[:np, :, 0] = S_pnn
                for p1 in range(1, size):
                    band_comm.receive(S_pnn, p1, 13)
                    for p2 in range(np):
                        if p1 + p2 < size:
                            S_bnbn[p1 + p2, :, p1] = S_pnn[p2]
                        else:
                            S_bnbn[p1, :, p1 + p2 - size] = S_pnn[p2].T
            else:
                band_comm.send(S_pnn, 0, 13)
