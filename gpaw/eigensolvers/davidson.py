"""Module defining  ``Eigensolver`` classes."""

import numpy as np

from gpaw.utilities.blas import axpy, rk, r2k, gemm
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import unpack
from gpaw.eigensolvers.eigensolver import Eigensolver


class Davidson(Eigensolver):
    """Simple Davidson eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated.

    Solution steps are:

    * Subspace diagonalization
    * Calculate all residuals
    * Add preconditioned residuals to the subspace and diagonalize 
    """

    def __init__(self, niter=2):
        Eigensolver.__init__(self)
        self.niter = niter

    def initialize(self, wfs):
        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap
        # Allocate arrays
        self.S_nn = np.zeros((self.nbands, self.nbands), self.dtype)
        self.H_2n2n = np.empty((2 * self.nbands, 2 * self.nbands),
                                self.dtype)
        self.S_2n2n = np.empty((2 * self.nbands, 2 * self.nbands),
                                self.dtype)        
        self.eps_2n = np.empty(2 * self.nbands)        

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do Davidson iterations for the kpoint"""
        niter = self.niter
        nbands = self.nbands

        self.subspace_diagonalize(hamiltonian, wfs, kpt)
                    
        H_2n2n = self.H_2n2n
        S_2n2n = self.S_2n2n
        eps_2n = self.eps_2n
        psit2_nG = self.overlap.operator.suggest_temporary_buffer(wfs.dtype)

        self.timer.start('Davidson')
        R_nG = self.Htpsit_nG 
        self.calculate_residuals2(wfs, hamiltonian, kpt, R_nG)

        for nit in range(niter):
            H_2n2n[:] = 0.0
            S_2n2n[:] = 0.0

            error = 0.0
            for n in range(nbands):
                if kpt.f_n is None:
                    weight = kpt.weight
                else:
                    weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    if n < self.nbands_converge:
                        weight = kpt.weight
                    else:
                        weight = 0.0
                error += weight * np.vdot(R_nG[n], R_nG[n]).real

                H_2n2n[n,n] = kpt.eps_n[n]
                S_2n2n[n,n] = 1.0
                psit2_nG[n] = self.preconditioner(R_nG[n], kpt.phase_cd)
            
            # Calculate projections
            P2_ani = wfs.pt.dict(nbands)
            wfs.pt.integrate(psit2_nG, P2_ani, kpt.q)
            
            # Hamiltonian matrix
            # <psi2 | H | psi>
            wfs.kin.apply(psit2_nG, self.Htpsit_nG, kpt.phase_cd)
            hamiltonian.apply_local_potential(psit2_nG, self.Htpsit_nG, kpt.s)
            gemm(self.gd.dv, kpt.psit_nG, self.Htpsit_nG, 0.0, self.H_nn, 'c')

            for a, P_ni in kpt.P_ani.items():
                P2_ni = P2_ani[a]
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                self.H_nn += np.dot(P2_ni, np.dot(dH_ii, P_ni.T.conj()))

            self.gd.comm.sum(self.H_nn, 0)
            H_2n2n[nbands:, :nbands] = self.H_nn

            # <psi2 | H | psi2>
            r2k(0.5 * self.gd.dv, psit2_nG, self.Htpsit_nG, 0.0, self.H_nn)
            for a, P2_ni in P2_ani.items():
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                self.H_nn += np.dot(P2_ni, np.dot(dH_ii, P2_ni.T.conj()))

            self.gd.comm.sum(self.H_nn, 0)
            H_2n2n[nbands:, nbands:] = self.H_nn

            # Overlap matrix
            # <psi2 | S | psi>
            gemm(self.gd.dv, kpt.psit_nG, psit2_nG, 0.0, self.S_nn, "c")
        
            for a, P_ni in kpt.P_ani.items():
                P2_ni = P2_ani[a]
                dO_ii = wfs.setups[a].O_ii
                self.S_nn += np.dot(P2_ni, np.inner(dO_ii, P_ni.conj()))

            self.gd.comm.sum(self.S_nn, 0)
            S_2n2n[nbands:, :nbands] = self.S_nn

            # <psi2 | S | psi2>
            rk(self.gd.dv, psit2_nG, 0.0, self.S_nn)
            for a, P2_ni in P2_ani.items():
                dO_ii = wfs.setups[a].O_ii
                self.S_nn += np.dot(P2_ni, np.dot(dO_ii, P2_ni.T.conj()))

            self.gd.comm.sum(self.S_nn, 0)
            S_2n2n[nbands:, nbands:] = self.S_nn

            if self.gd.comm.rank == 0:
                info = diagonalize(H_2n2n, eps_2n, S_2n2n)
                if info != 0:
                    raise RuntimeError, 'Very Bad!!'

            self.gd.comm.broadcast(H_2n2n, 0)
            self.gd.comm.broadcast(eps_2n, 0)

            kpt.eps_n[:] = eps_2n[:nbands]

            # Rotate psit_nG
            gemm(1.0, kpt.psit_nG, H_2n2n[:nbands, :nbands],
                 0.0, self.Htpsit_nG)
            gemm(1.0, psit2_nG, H_2n2n[:nbands, nbands:],
                 1.0, self.Htpsit_nG)
            kpt.psit_nG, self.Htpsit_nG = self.Htpsit_nG, kpt.psit_nG

            # Rotate P_uni:
            for a, P_ni in kpt.P_ani.items():
                P2_ni = P2_ani[a]
                gemm(1.0, P_ni.copy(), H_2n2n[:nbands, :nbands], 0.0, P_ni)
                gemm(1.0, P2_ni, H_2n2n[:nbands, nbands:], 1.0, P_ni)

            if nit < niter - 1 :
                wfs.kin.apply(kpt.psit_nG, self.Htpsit_nG, kpt.phase_cd)
                hamiltonian.apply_local_potential(kpt.psit_nG, self.Htpsit_nG,
                                                  kpt.s)
                R_nG = self.Htpsit_nG
                self.calculate_residuals2(wfs, hamiltonian, kpt, R_nG)

        self.timer.stop('Davidson')
        error = self.gd.comm.sum(error)
        return error

    
