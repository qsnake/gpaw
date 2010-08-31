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

    def __init__(self, keep_htpsit=True, block=1):
        Eigensolver.__init__(self, keep_htpsit, block)
        
    def initialize(self, wfs):
        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap
        self.RdR = np.empty(self.block, float)
        self.dRdR = np.empty(self.block, float)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do a single RMM-DIIS iteration for the kpoint"""

        self.subspace_diagonalize(hamiltonian, wfs, kpt)

        self.timer.start('RMM-DIIS')
        if self.keep_htpsit:
            R_nG = self.Htpsit_nG
            self.calculate_residuals2(wfs, hamiltonian, kpt, R_nG)

        vt_G = hamiltonian.vt_sG[kpt.s]
        # XXX Smarter and safer way to use temporary buffers is needed!
        if self.keep_htpsit:
            # The full work1_xG is available for dR_xG
            dR_xG = wfs.matrixoperator.work1_xG
        else:
            # Space is needed both for R_xG and dR_xG
            # XXX we are assuming now that there is either band
            # XXX parallelization or blocking for matrix multiplies
            # XXX so that both work1_xG and work2_xG exist
            B = self.mynbands // wfs.matrixoperator.nblocks
            dR_xG = wfs.matrixoperator.work1_xG
            R_xG = wfs.matrixoperator.work2_xG
                        
        B = self.block
        RdR = self.RdR
        dRdR = self.dRdR
        error = 0.0
        n0 = self.band_comm.rank * self.mynbands
        for n1 in range(0, self.mynbands, B):
            n2 = min(n1 + B, self.mynbands)
            nb = n2 - n1 # Number of bands in this block
            if self.keep_htpsit:
                R_xG = R_nG[n1:n2]
            else:
                self.calculate_residuals(wfs, hamiltonian, kpt,
                                         kpt.eps_n[n1:n2],
                                         R_xG[:nb],
                                         kpt.psit_nG[n1:n2])

            for i in range(nb):
                n = n1 + i
                if kpt.f_n is None:
                    weight = kpt.weight
                else:
                    weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    if n0 + n < self.nbands_converge:
                        weight = kpt.weight
                    else:
                        weight = 0.0
                error += weight * np.vdot(R_xG[i], R_xG[i]).real

            # Precondition the residual:
            self.timer.start('precondition')
            pR_xG = self.preconditioner(R_xG[:nb], kpt)
            self.timer.stop('precondition')

            # Calculate the residual of pR_G, dR_G = (H - e S) pR_G:
            self.calculate_residuals(wfs, hamiltonian, kpt, kpt.eps_n[n1:n2],
                                     dR_xG[:nb], pR_xG[:nb], (n1,n2))

            
            # Apply non-local corrections and find lam that minimizes
            # the norm of R'_G = R_G + lam dR_G
            for i in range(nb):
                n = n1 + i
                hamiltonian.xc.xcfunc.adjust_non_local_residual(
                    pR_xG[i], dR_xG[i], kpt, n)

                RdR[i] = np.vdot(R_xG[i], dR_xG[i]).real
                dRdR[i] = np.vdot(dR_xG[i], dR_xG[i]).real
            self.gd.comm.sum(RdR)
            self.gd.comm.sum(dRdR)

            lam = -RdR / dRdR
            # Calculate new psi'_G = psi_G + lam pR_G + lam pR'_G
            #                      = psi_G + p(2 lam R_G + lam**2 dR_G)
            for i in range(nb):
                R_xG[i] *= 2.0 * lam[i]
                axpy(lam[i]**2, dR_xG[i], R_xG[i])  # R_G += lam**2 * dR_G

            self.timer.start('precondition')
            kpt.psit_nG[n1:n2] += self.preconditioner(R_xG[:nb], kpt)
            self.timer.stop('precondition')
            
        self.timer.stop('RMM-DIIS')
        error = self.gd.comm.sum(error)
        return error
    
