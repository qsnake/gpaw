# -*- coding: utf-8 -*-
# Copyright (C) 2008  CSC Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.

"""This module defines an Overlap operator.

The module defines an overlap operator and implements overlap-related
functions.

"""
import sys

import numpy as npy
from gpaw.mpi import run
from gpaw.utilities.complex import cc
from gpaw.utilities.blas import rk, gemm
from gpaw.utilities.lapack import inverse_cholesky
from gpaw.utilities import swap

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
        self.work_nn = None
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
            
        nbands = len(psit_nG)

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
        rk(self.gd.dv, psit_nG, 0.0, S_nn)

        for nucleus in self.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            dO_ii = nucleus.setup.O_ii
            #???????gemm(1.0, P_ni, npy.dot(P_ni, dO_ii), 1.0, S_nn, 't')
            S_nn += npy.dot(P_ni, cc(npy.inner(nucleus.setup.O_ii, P_ni)))

        self.comm.sum(S_nn, kpt.root)

        if self.comm.rank == kpt.root:
            info = inverse_cholesky(S_nn)
            if info != 0:
                raise RuntimeError('Orthogonalization failed!')

        # S_nn has been overwriten - let's call it something different:
        C_nn = S_nn
        del S_nn

        self.comm.broadcast(C_nn, kpt.root)

        if len(work_nG) == nbands:
            # We have enough work space to do this operation in one step:
            gemm(1.0, psit_nG, C_nn, 0.0, work_nG)

            kpt.psit_nG = work_nG

            if work_nG is self.big_work_arrays.get('work_nG'):
                self.big_work_arrays['work_nG'] = psit_nG

        else:
            blocked_matrix_multiplication(psit_nG, C_nn, work_nG)
            
        for nucleus in self.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            gemm(1.0, P_ni.copy(), C_nn, 0.0, P_ni)

        self.timer.stop('Orthonormalize')
