"""Module defining  ``Eigensolver`` classes."""

from math import pi, sqrt, sin, cos, atan2

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!
import LinearAlgebra as linalg

from gpaw.utilities.blas import axpy, rk, r2k, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import unpack
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.mpi import run


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

    def __init__(self):
        Eigensolver.__init__(self)

    def initialize(self, paw):
        Eigensolver.initialize(self, paw)
        # Allocate arrays
        self.S_nn = num.zeros((self.nbands, self.nbands), self.typecode)
        self.H_2n2n = num.empty((2 * self.nbands, 2 * self.nbands),
                                self.typecode)
        self.S_2n2n = num.empty((2 * self.nbands, 2 * self.nbands),
                                self.typecode)        
        self.eps_2n = num.empty(2 * self.nbands, num.Float)        

    def iterate_one_k_point(self, hamiltonian, kpt, niter=2):
        """Do Davidson iterations for the kpoint"""

        nbands = self.nbands

        self.diagonalize(hamiltonian, kpt)
                    
        H_2n2n = self.H_2n2n
        S_2n2n = self.S_2n2n
        eps_2n = self.eps_2n
        psit2_nG = self.work

        self.timer.start('Davidson')
        R_nG = self.Htpsit_nG
        # optimize XXX 
        for R_G, eps, psit_G in zip(R_nG, kpt.eps_n, kpt.psit_nG):
            axpy(-eps, psit_G, R_G)  # R_G -= eps * psit_G
                
        run([nucleus.adjust_residual(R_nG, kpt.eps_n, kpt.s, kpt.u, kpt.k)
             for nucleus in hamiltonian.pt_nuclei])

        for nit in range(niter):
            H_2n2n[:] = 0.0
            S_2n2n[:] = 0.0
            error = 0.0
            for n in range(nbands):
                weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    weight = kpt.weight * float(n < self.nbands_converge)
                error += weight * real(num.vdot(R_nG[n], R_nG[n]))

                H_2n2n[n,n] = kpt.eps_n[n]
                S_2n2n[n,n] = 1.0
                psit2_nG[n] = self.preconditioner(R_nG[n], kpt.phase_cd, None, kpt.k_c)
            
            # Calculate projections
            for nucleus in hamiltonian.pt_nuclei:
                ni = nucleus.get_number_of_partial_waves()
                nucleus.P2_ni = num.zeros((nbands, ni), self.typecode)
                if nucleus.in_this_domain:
                    nucleus.pt_i.integrate(psit2_nG, nucleus.P2_ni, kpt.k)
                else:
                    nucleus.pt_i.integrate(psit2_nG, None, kpt.k)
            
            # Hamiltonian matrix
            # <psi2 | H | psi>
            hamiltonian.kin.apply(psit2_nG, self.Htpsit_nG, kpt.phase_cd)
            self.Htpsit_nG += psit2_nG * hamiltonian.vt_sG[kpt.s]
            gemm(self.gd.dv, kpt.psit_nG, self.Htpsit_nG, 0.0, self.H_nn,
"c")

            for nucleus in hamiltonian.my_nuclei:
                P_ni = nucleus.P_uni[kpt.u]
                P2_ni = nucleus.P2_ni
                self.H_nn += num.dot(P2_ni, num.dot(unpack(nucleus.H_sp[kpt.s]),
                                                    cc(num.transpose(P_ni))))

            self.comm.sum(self.H_nn, kpt.root)
            H_2n2n[nbands:, :nbands] = self.H_nn

            # <psi2 | H | psi2>
            r2k(0.5 * self.gd.dv, psit2_nG, self.Htpsit_nG, 0.0, self.H_nn)
            for nucleus in hamiltonian.my_nuclei:
                P2_ni = nucleus.P2_ni
                self.H_nn += num.dot(P2_ni, num.dot(unpack(nucleus.H_sp[kpt.s]),
                                                    cc(num.transpose(P2_ni))))

            self.comm.sum(self.H_nn, kpt.root)
            H_2n2n[nbands:, nbands:] = self.H_nn

            # Overlap matrix
            # <psi2 | S | psi>
            gemm(self.gd.dv, kpt.psit_nG, psit2_nG, 0.0, self.S_nn, "c")
        
            for nucleus in hamiltonian.my_nuclei:
                P_ni = nucleus.P_uni[kpt.u]
                P2_ni = nucleus.P2_ni
                self.S_nn += num.dot(P2_ni,
                                     cc(inner(nucleus.setup.O_ii, P_ni)))

            self.comm.sum(self.S_nn, kpt.root)
            S_2n2n[nbands:, :nbands] = self.S_nn

            # <psi2 | S | psi2>
            rk(self.gd.dv, psit2_nG, 0.0, self.S_nn)
            for nucleus in hamiltonian.my_nuclei:
                P2_ni = nucleus.P2_ni
                self.S_nn += num.dot(P2_ni,
                                     cc(inner(nucleus.setup.O_ii, P2_ni)))

            self.comm.sum(self.S_nn, kpt.root)
            S_2n2n[nbands:, nbands:] = self.S_nn

            if self.comm.rank == kpt.root:
                info = diagonalize(H_2n2n, eps_2n, S_2n2n)
                if info != 0:
                    raise RuntimeError, 'Very Bad!!'

            self.comm.broadcast(H_2n2n, kpt.root)
            self.comm.broadcast(eps_2n, kpt.root)

            kpt.eps_n[:] = eps_2n[:nbands]

            # Rotate psit_nG
            gemm(1.0, kpt.psit_nG, H_2n2n[:nbands, :nbands].copy(),
                 0.0, self.Htpsit_nG)
            gemm(1.0, psit2_nG, H_2n2n[:nbands, nbands:].copy(),
                 1.0, self.Htpsit_nG)
            kpt.psit_nG, self.Htpsit_nG = self.Htpsit_nG, kpt.psit_nG

            # Rotate P_uni:
            for nucleus in hamiltonian.my_nuclei:
                P_ni = nucleus.P_uni[kpt.u]
                P2_ni = nucleus.P2_ni
                gemm(1.0, P_ni.copy(), H_2n2n[:nbands, :nbands].copy(), 0.0, P_ni)
                gemm(1.0, P2_ni, H_2n2n[:nbands, nbands:].copy(), 1.0, P_ni)

            if nit < niter - 1 :
                hamiltonian.kin.apply(kpt.psit_nG, self.Htpsit_nG, kpt.phase_cd)
                self.Htpsit_nG += kpt.psit_nG * hamiltonian.vt_sG[kpt.s]
                R_nG = self.Htpsit_nG
                # optimize XXX 
                for R_G, eps, psit_G in zip(R_nG, kpt.eps_n, kpt.psit_nG):
                    axpy(-eps, psit_G, R_G)  # R_G -= eps * psit_G
                
                run([nucleus.adjust_residual(R_nG, kpt.eps_n,
                                             kpt.s, kpt.u, kpt.k)
                     for nucleus in hamiltonian.pt_nuclei])
        self.timer.stop('Davidson')
        error = self.comm.sum(error)
        return error

    
