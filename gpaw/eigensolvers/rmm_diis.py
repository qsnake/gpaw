"""Module defining  ``Eigensolver`` classes."""

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!
import LinearAlgebra as linalg

from gpaw.utilities.blas import axpy, rk, gemm
from gpaw.utilities import elementwise_multiply_add, utilities_vdot, utilities_vdot_self
from gpaw.utilities.complex import cc, real
from gpaw.eigensolvers.eigensolver import Eigensolver
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

    def __init__(self, paw):

        Eigensolver.__init__(self, paw)

        self.S_nn = num.empty((self.nbands, self.nbands), self.typecode)
        self.S_nn[:] = 0.0  # rk fails the first time without this!

    def iterate_one_k_point(self, hamiltonian, kpt):      
        """Do a single RMM-DIIS iteration for the kpoint"""

        self.diagonalize(hamiltonian, kpt)

        self.timer.start('Residuals')
        R_nG = self.Htpsit_nG
        # optimize XXX 
        for R_G, eps, psit_G in zip(R_nG, kpt.eps_n, kpt.psit_nG):
            axpy(-eps, psit_G, R_G)  # R_G -= eps * psit_G

        run([nucleus.adjust_residual(R_nG, kpt.eps_n, kpt.s, kpt.u, kpt.k)
             for nucleus in hamiltonian.pt_nuclei])

        self.timer.stop('Residuals')

        self.timer.start('RMM-DIIS')
        vt_G = hamiltonian.vt_sG[kpt.s]
        dR_G = self.work[0]
        error = 0.0
        for n in range(kpt.nbands):
            R_G = R_nG[n]

            weight = kpt.f_n[n]
            if self.convergeall:
                weight = 1.0
            error += weight * real(num.vdot(R_G, R_G))

            pR_G = self.preconditioner(R_G, kpt.phase_cd, kpt.psit_nG[n],
                                  kpt.k_c)

            hamiltonian.kin.apply(pR_G, dR_G, kpt.phase_cd)
                
            if (dR_G.typecode() == num.Float):
                elementwise_multiply_add(vt_G, pR_G, dR_G)
            else:
                dR_G += vt_G * pR_G
            
            axpy(-kpt.eps_n[n], pR_G, dR_G)  # dR_G -= kpt.eps_n[n] * pR_G

            run([nucleus.adjust_residual2(pR_G, dR_G, kpt.eps_n[n],
                                          kpt.u, kpt.s, kpt.k, n)
                 for nucleus in hamiltonian.pt_nuclei])

            hamiltonian.xc.xcfunc.adjust_non_local_residual(
                pR_G, dR_G, kpt.eps_n[n], kpt.u, kpt.s, kpt.k, n)
            
            if (dR_G.typecode() == num.Float):
                RdR = self.comm.sum(utilities_vdot(R_G, dR_G))
                dRdR = self.comm.sum(utilities_vdot_self(dR_G))
            else:
                RdR = self.comm.sum(real(num.vdot(R_G, dR_G)))
                dRdR = self.comm.sum(real(num.vdot(dR_G, dR_G)))
            lam = -RdR / dRdR

            R_G *= 2.0 * lam
            axpy(lam**2, dR_G, R_G)  # R_G += lam**2 * dR_G
            kpt.psit_nG[n] += self.preconditioner(R_G, kpt.phase_cd,
                                                 kpt.psit_nG[n], kpt.k_c)
            
        self.timer.stop('RMM-DIIS')

        self.timer.start('Orthogonalize')
        run([nucleus.calculate_projections(kpt)
             for nucleus in hamiltonian.pt_nuclei])

        S_nn = self.S_nn

        # Fill in the lower triangle:
        rk(self.gd.dv, kpt.psit_nG, 0.0, S_nn)
        
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            S_nn += num.dot(P_ni, cc(inner(nucleus.setup.O_ii, P_ni)))
        
        self.comm.sum(S_nn, kpt.root)

        if self.comm.rank == kpt.root:
            # inverse returns a non-contigous matrix - grrrr!  That is
            # why there is a copy.  Should be optimized with a
            # different lapack call to invert a triangular matrix XXXXX
            S_nn[:] = linalg.inverse(
                linalg.cholesky_decomposition(S_nn)).copy()

        self.comm.broadcast(S_nn, kpt.root)
        
        gemm(1.0, kpt.psit_nG, S_nn, 0.0, self.work)
        kpt.psit_nG, self.work = self.work, kpt.psit_nG  # swap

        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            gemm(1.0, P_ni.copy(), S_nn, 0.0, P_ni)
        self.timer.stop('Orthogonalize')

        return error
    
