"""Module defining  ``Eigensolver`` classes."""

import numpy as npy

from gpaw.utilities.blas import axpy, rk, gemm
from gpaw.utilities.lapack import inverse_cholesky
from gpaw.utilities import elementwise_multiply_add, utilities_vdot, utilities_vdot_self
from gpaw.utilities import unpack
from gpaw.utilities.complex import cc, real
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.mpi import run


class RMM_DIIS2(Eigensolver):
    """RMM-DIIS eigensolver


    This is a simpler (and more inefficient) implementation of RMM-DIIS
    algorithm which utilizes `hamiltonian.apply` and `overlap.apply`
    functions. Main purpose of this implementation is testing and optimization
    of the above `apply` functions.

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated

    Solution steps are:

    * Subspace diagonalization
    * Calculation of residuals
    * Improvement of wave functions:  psi' = psi + lambda PR + lambda PR'
    * Orthonormalization"""

    def __init__(self, rotate=True):
        Eigensolver.__init__(self)
        self.rotate = rotate
        self.nbands = None

    def initialize(self, paw):
        Eigensolver.initialize(self, paw)

        self.S_nn = npy.empty((self.nbands, self.nbands), self.dtype)
        self.S_nn[:] = 0.0  # rk fails the first time without this!

        self.overlap = paw.overlap
        
    def iterate_one_k_point(self, hamiltonian, kpt):      
        """Do a single RMM-DIIS iteration for the kpoint"""

        overlap = self.overlap

        self.diagonalize(hamiltonian, kpt, self.rotate)

        R_nG = self.Htpsit_nG
        # Calculate all the residuals
        self.timer.start('Residuals')

        hamiltonian.apply(kpt.psit_nG, R_nG, kpt, calculate_P_uni=False)
        overlap.apply(kpt.psit_nG, self.work, kpt, calculate_P_uni=False)
        for R_G, eps, Spsit_G in zip(R_nG, kpt.eps_n, self.work):
            # R_G -= eps * Spsit_G
            axpy(-eps, Spsit_G, R_G)

        self.timer.stop('Residuals')

        self.timer.start('RMM-DIIS')
        vt_G = hamiltonian.vt_sG[kpt.s]
        dR_G = self.work[0]
        error = 0.0
        for n in range(kpt.nbands):
            R_G = R_nG[n]

            weight = kpt.f_n[n]
            if self.nbands_converge != 'occupied':
                weight = kpt.weight * float(n < self.nbands_converge)
            error += weight * real(npy.vdot(R_G, R_G))

            # Precondition the residual
            pR_G = self.preconditioner(R_G, kpt.phase_cd, kpt.psit_nG[n],
                                  kpt.k_c)

            # Calculate the residual of pR_G, dR_G = (H - e S) pR_G
            hamiltonian.apply(pR_G, dR_G, kpt)
            overlap.apply(pR_G, self.work[1], kpt)
            axpy(-kpt.eps_n[n], self.work[1], dR_G)

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            RdR = self.gd.comm.sum(real(npy.vdot(R_G, dR_G)))
            dRdR = self.gd.comm.sum(real(npy.vdot(dR_G, dR_G)))
            lam = -RdR / dRdR

            # Calculate new psi'_G = psi_G + lam pR_G + lam pR'_G
            #                      = psi_G + p(2 lam R_G + lam**2 dR_G)
            R_G *= 2.0 * lam
            axpy(lam**2, dR_G, R_G)  # R_G += lam**2 * dR_G
            kpt.psit_nG[n] += self.preconditioner(R_G, kpt.phase_cd,
                                                 kpt.psit_nG[n], kpt.k_c)
            
        self.timer.stop('RMM-DIIS')

        # Orthonormalize the wave functions
        self.timer.start('Calculate projections')
        run([nucleus.calculate_projections(kpt)
             for nucleus in hamiltonian.pt_nuclei])
        self.timer.stop('Calculate projections')

        overlap.orthonormalize(kpt.psit_nG, kpt, self.work, self.S_nn)
     
     
        error = self.gd.comm.sum(error)
        return error
    
