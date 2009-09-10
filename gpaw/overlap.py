# -*- coding: utf-8 -*-
# Copyright (C) 2008  CSC Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.

"""This module defines an Overlap operator.

The module defines an overlap operator and implements overlap-related
functions.

"""
import sys

import numpy as np

from gpaw.mpi import parallel
from gpaw.utilities.lapack import inverse_cholesky
from gpaw.utilities import scalapack
from gpaw import sl_inverse_cholesky
from gpaw.hs_operators import Operator


class Overlap:
    """Overlap operator class.

    Attributes
    ==========

    dtype: type object
        Numerical type of operator (float/complex)

    """

    def __init__(self, wfs):
        """Create the Overlap operator."""

        self.operator = Operator(wfs.bd, wfs.gd)
        self.timer = wfs.timer
        self.domain_comm = wfs.gd.comm
        self.band_comm = wfs.bd.comm
        self.setups = wfs.setups

        self.S_nn = None
        
    def orthonormalize(self, wfs, kpt):
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

        psit_nG = kpt.psit_nG
        P_ani = kpt.P_ani
        wfs.pt.integrate(psit_nG, P_ani, kpt.q)
        mynbands = len(psit_nG)
        nbands = mynbands * self.band_comm.size

        # Construct the overlap matrix:
        S = lambda x: x
        dS_aii = dict([(a, self.setups[a].O_ii) for a in P_ani])
        self.timer.start('Orthonormalize: calc_matrix')
        S_nn = self.operator.calculate_matrix_elements(psit_nG, P_ani,
                                                       S, dS_aii)
        self.timer.stop('Orthonormalize: calc_matrix')
        self.timer.start('Orthonormalize: inverse_cholesky')

        if sl_inverse_cholesky:
            assert parallel and scalapack()
            info = inverse_cholesky(S_nn, 0)
        else:
            if self.domain_comm.rank == 0 and self.band_comm.rank == 0:
                info = inverse_cholesky(S_nn)
        if sl_inverse_cholesky:
            if info != 0:
                raise RuntimeError('Orthogonalization failed!')
        else:
            if self.domain_comm.rank == 0 and self.band_comm.rank == 0:
                if info != 0:
                    raise RuntimeError('Orthogonalization failed!')
        self.timer.stop('Orthonormalize: inverse_cholesky')

        # S_nn now contains the inverse of the Cholesky factorization.
        # Let's call it something different:
        C_nn = S_nn
        del S_nn

        if self.band_comm.rank == 0:
            self.domain_comm.broadcast(C_nn, 0)
        self.band_comm.broadcast(C_nn, 0)

        self.timer.start('Orthonormalize: rotate_psi')
        kpt.psit_nG = self.operator.matrix_multiply(C_nn, psit_nG, P_ani)
        self.timer.stop('Orthonormalize: rotate_psi')
        self.timer.stop('Orthonormalize')

    def apply(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True):
        """Apply the overlap operator to a set of vectors.

        Parameters
        ==========
        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting S times a_nG vectors.
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_ani are used

        """

        self.timer.start('Apply overlap')
        b_xG[:] = a_xG
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani:
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a, P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            P_axi[a] = np.dot(P_xi, self.setups[a].O_ii)
            # gemm(1.0, self.setups[a].O_ii, P_xi, 0.0, P_xi, 'n')
        wfs.pt.add(b_xG, P_axi, kpt.q) # b_xG += sum_ai pt^a_i P_axi
        self.timer.stop('Apply overlap')

    def apply_inverse(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True):
        """Apply approximative inverse overlap operator to wave functions."""

        b_xG[:] = a_xG
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani:
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a,P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            P_axi[a] = np.dot(P_xi, self.setups[a].C_ii)
        wfs.pt.add(b_xG, P_axi, kpt.q)

    def estimate_memory(self, mem, dtype):
        self.operator.estimate_memory(mem, dtype)
