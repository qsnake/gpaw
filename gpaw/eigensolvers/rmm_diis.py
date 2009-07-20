"""Module defining  ``Eigensolver`` classes."""

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.utilities import unpack
from gpaw.mpi import run


class RMM_DIIS(Eigensolver):
    """RMM-DIIS eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated

    Solution steps are:

    * Subspace diagonalization
    * Calculation of residuals
    * Improvement of wave functions:  psi' = psi + lambda PR + lambda PR'
    * Orthonormalization"""

    def __init__(self, keep_htpsit=True):
        Eigensolver.__init__(self, keep_htpsit)

    def initialize(self, wfs):
        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do a single RMM-DIIS iteration for the kpoint"""

        self.subspace_diagonalize(hamiltonian, wfs, kpt)

        self.timer.start('Residuals')
        if self.keep_htpsit:
            R_nG = self.Htpsit_nG
            self.calculate_residuals2(wfs, hamiltonian, kpt, R_nG)
        self.timer.stop('Residuals')

        self.timer.start('RMM-DIIS')
        vt_G = hamiltonian.vt_sG[kpt.s]
        dR_G = wfs.overlap.operator.work1_xG[0] # XXX presumptuous, but works
        error = 0.0
        n0 = self.band_comm.rank * self.mynbands
        #B = max(self.mynbands // 8, self.mynbands)
        B = 1
        for n1 in range(0, self.mynbands, B):
            n2 = min(n1 + B, self.mynbands)
            shape = (1,) + dR_G.shape
            if self.keep_htpsit:
                R_G = R_nG[n1]
            else:
                R_G = wfs.overlap.operator.work1_xG[1] # XXX hello IndexError
                self.calculate_residuals(wfs, hamiltonian, kpt,
                                         kpt.eps_n[n1:n2],
                                         R_G.reshape(shape),
                                         kpt.psit_nG[n1:n2])

            for n in range(n1, n2):
                if kpt.f_n is None:
                    weight = kpt.weight
                else:
                    weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    if n0 + n < self.nbands_converge:
                        weight = kpt.weight
                    else:
                        weight = 0.0
                error += weight * np.vdot(R_G, R_G).real

            # Precondition the residual:
            pR_G = self.preconditioner(R_G, kpt.phase_cd, kpt.psit_nG[n1])

            # Calculate the residual of pR_G, dR_G = (H - e S) pR_G:
            self.calculate_residuals(wfs, hamiltonian, kpt, kpt.eps_n[n1:n2],
                                     dR_G.reshape(shape),
                                     pR_G.reshape(shape), n1)

            hamiltonian.xc.xcfunc.adjust_non_local_residual(
                pR_G.reshape(shape), dR_G.reshape(shape), kpt, n1)
            
            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            RdR = self.gd.comm.sum(np.vdot(R_G, dR_G).real)
            dRdR = self.gd.comm.sum(np.vdot(dR_G, dR_G).real)

            lam = -RdR / dRdR
            # Calculate new psi'_G = psi_G + lam pR_G + lam pR'_G
            #                      = psi_G + p(2 lam R_G + lam**2 dR_G)
            R_G *= 2.0 * lam
            axpy(lam**2, dR_G, R_G)  # R_G += lam**2 * dR_G
            kpt.psit_nG[n1:n2] += self.preconditioner(R_G, kpt.phase_cd,
                                                   kpt.psit_nG[n1])
            
        self.timer.stop('RMM-DIIS')
        error = self.gd.comm.sum(error)
        return error
    
