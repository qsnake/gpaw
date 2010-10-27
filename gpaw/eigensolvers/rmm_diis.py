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

        self.timer.start('RMM-DIIS')
        if self.keep_htpsit:
            R_nG = self.Htpsit_nG
            self.calculate_residuals(kpt, wfs, hamiltonian, kpt.psit_nG,
                                     kpt.P_ani, kpt.eps_n, R_nG)

        B = 1
        dR_xG = self.gd.empty(B, wfs.dtype)
        P_axi = wfs.pt.dict(B)
        error = 0.0
        assert B == 1
        for n1 in range(0, wfs.bd.mynbands, B):
            n2 = min(n1 + B, wfs.bd.mynbands)
            n_x = range(n1, n2)
            if self.keep_htpsit:
                R_xG = R_nG[n_x]
            else:
                R_xG = self.gd.empty(n2 - n1, wfs.dtype)
                psit_xG = kpt.psit_nG[n_x]
                wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_xG, R_xG)
                wfs.pt.integrate(psit_xG, P_axi)
                self.calculate_residuals(kpt, wfs, hamiltonian, psit_xG,
                                         P_axi, kpt.eps_n[n_x], R_xG, n_x)

            for n in n_x:
                if kpt.f_n is None:
                    weight = kpt.weight
                else:
                    weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    if wfs.bd.global_index(n) < self.nbands_converge:
                        weight = kpt.weight
                    else:
                        weight = 0.0
                error += weight * np.vdot(R_xG[n - n1], R_xG[n - n1]).real

            # Precondition the residual:
            self.timer.start('precondition')
            dpsit_xG = self.preconditioner(R_xG, kpt)
            self.timer.stop('precondition')

            # Calculate the residual of dpsit_G, dR_G = (H - e S) dpsit_G:
            wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, dpsit_xG, dR_xG)
            wfs.pt.integrate(dpsit_xG, P_axi, kpt.q)
            self.calculate_residuals(kpt, wfs, hamiltonian, dpsit_xG,
                                     P_axi, kpt.eps_n[n_x], dR_xG, n_x,
                                     calculate_change=True)
            
            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            RdR = self.gd.comm.sum(np.vdot(R_xG, dR_xG).real)
            dRdR = self.gd.comm.sum(np.vdot(dR_xG, dR_xG).real)

            lam = -RdR / dRdR
            # Calculate new psi'_G = psi_G + lam pR_G + lam pR'_G
            #                      = psi_G + p(2 lam R_G + lam**2 dR_G)
            R_xG *= 2.0 * lam
            axpy(lam**2, dR_xG, R_xG)  # R_G += lam**2 * dR_G
            self.timer.start('precondition')
            kpt.psit_nG[n1:n2] += self.preconditioner(R_xG, kpt)
            self.timer.stop('precondition')
            
        self.timer.stop('RMM-DIIS')
        error = self.gd.comm.sum(error)
        return error
