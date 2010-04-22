# -*- coding: utf-8 -*-
# Copyright (C) 2008  CSC Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.

"""This module defines an Overlap operator.

The module defines an overlap operator and implements overlap-related
functions.

"""
import sys
import numpy as np

from gpaw.hs_operators import MatrixOperator

class Overlap:
    """Overlap operator class.

    Attributes
    ==========

    dtype: type object
        Numerical type of operator (float/complex)

    """

    def __init__(self, wfs):
        """Create the Overlap operator."""
        self.ksl = wfs.orthoksl
        self.operator = MatrixOperator(wfs.bd, wfs.gd, self.ksl)
        self.timer = wfs.timer
        self.setups = wfs.setups
        
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

        # Construct the overlap matrix:
        S = lambda x: x
        dS_aii = dict([(a, self.setups[a].dO_ii) for a in P_ani])
        self.timer.start('calc_matrix')
        S_nn = self.operator.calculate_matrix_elements(psit_nG, P_ani,
                                                       S, dS_aii)
        self.timer.stop('calc_matrix')

        orthonormalization_string = repr(self.ksl)
        self.timer.start(orthonormalization_string)
        self.ksl.inverse_cholesky(S_nn)
        # S_nn now contains the inverse of the Cholesky factorization.
        # Let's call it something different:
        C_nn = S_nn
        del S_nn
        self.timer.stop(orthonormalization_string)

        self.timer.start('rotate_psi')
        kpt.psit_nG = self.operator.matrix_multiply(C_nn, psit_nG, P_ani)
        self.timer.stop('rotate_psi')
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
            P_axi[a] = np.dot(P_xi, self.setups[a].dO_ii)
            # gemm(1.0, self.setups[a].dO_ii, P_xi, 0.0, P_xi, 'n')
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
            P_axi[a] = np.dot(P_xi, self.setups[a].dC_ii)
        wfs.pt.add(b_xG, P_axi, kpt.q)

    def estimate_memory(self, mem, dtype):
        self.operator.estimate_memory(mem, dtype)

