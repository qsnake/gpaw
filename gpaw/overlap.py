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


    def orthonormalize(self, a_nG, kpt, work_nG=None, work_nn=None):
        """Orthonormalizes the vectors a_nG with respect to the overlap.

        First, a Cholesky factorization C is done for the overlap
        matrix S_nn = <a_nG | S | a_nG> = C*_nn C_nn Cholesky matrix C
        is inverted and orthonormal vectors a_nG' are obtained as::

          a_nG' = inv(C_nn) a_nG
        
        Parameters
        ----------
        a_nG: ndarray, input/output
            On input the set of vectors to orthonormalize,
            on output the overlap-orthonormalized vectors.
        kpt: KPoint object:
            k-point object from kpoint.py.
        work_nG: ndarray
            Optional work array for overlap matrix times a_nG.
        work_nn: ndarray
            Optional work array for overlap matrix.

        """

        self.timer.start('Orthonormalize')

        # Allocate work arrays if necessary
        nbands = len(a_nG)
        if work_nn is None:
            work_nn = npy.zeros((nbands, nbands), self.dtype)
        elif len(work_nn) != nbands:
                raise RuntimeError('Incompatible dimensions: %d != %d' % (len(S_nn), nbands))
        
        if work_nG is None:
            work_nG = npy.zeros(a_nG.shape, self.dtype)
        elif work_nG.shape != a_nG.shape:
                raise RuntimeError('Incompatible dimensions')

        S_nn = work_nn

        # Construct the overlap matrix
        rk(self.gd.dv, a_nG, 0.0, S_nn)

        for nucleus in self.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            S_nn += npy.dot(P_ni, cc(npy.inner(nucleus.setup.O_ii, P_ni)))

        self.comm.sum(S_nn, kpt.root)

        if self.comm.rank == kpt.root:
            info = inverse_cholesky(S_nn)
            if info != 0:
                raise RuntimeError('Orthogonalization failed!')

        self.comm.broadcast(S_nn, kpt.root)

        gemm(1.0, a_nG, S_nn, 0.0, work_nG)

        if sys.getrefcount(a_nG) > 4:
            raise RuntimeError("Can't swap pointers.   " +
                               'A view of a_nG exists somewhere!')
        swap(a_nG, work_nG)  # swap the pointers

        for nucleus in self.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            gemm(1.0, P_ni.copy(), S_nn, 0.0, P_ni)

        self.timer.stop('Orthonormalize')
