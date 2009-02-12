"""Module defining  ``Eigensolver`` classes."""

from math import pi, sqrt, sin, cos, atan2

import numpy as np
from numpy import dot # avoid the dotblas bug!

from gpaw.utilities.blas import axpy, rk, r2k, gemm
from gpaw.utilities import unpack
from gpaw.eigensolvers.eigensolver import Eigensolver


class CG(Eigensolver):
    """Conjugate gardient eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated.

    Solution steps are:

    * Subspace diagonalization
    * Calculate all residuals
    * Conjugate gradient steps
    """

    def __init__(self, niter=4):
        Eigensolver.__init__(self)
        self.niter = niter

    def initialize(self, wfs):
        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap
        # Allocate arrays
        self.phi_G = self.gd.empty(dtype=self.dtype)
        self.phi_old_G = self.gd.empty(dtype=self.dtype)

        # self.f = open('CG_debug','w')

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):      
        """Do a conjugate gradient iterations for the kpoint"""
        
        niter = self.niter
        phi_G = self.phi_G
        phi_old_G = self.phi_old_G
        
        self.subspace_diagonalize(hamiltonian, wfs, kpt)
        
        R_nG = self.overlap.operator.work1_xG
        Htphi_G = R_nG[0]
        
        R_nG[:] = self.Htpsit_nG
        self.timer.start('Residuals')        
        self.calculate_residuals2(wfs, hamiltonian, kpt, R_nG)
        self.timer.stop('Residuals')        

        self.timer.start('CG')
        vt_G = hamiltonian.vt_sG[kpt.s]

        total_error = 0.0
        for n in range(self.nbands):
            R_G = R_nG[n]
            Htpsit_G = self.Htpsit_nG[n]
            gamma_old = 1.0
            phi_old_G[:] = 0.0
            error = self.comm.sum(np.vdot(R_G, R_G).real)
            for nit in range(niter):
                if error < self.tolerance / self.nbands:
                    # print >> self.f, "cg:iters", n, nit
                    break

                pR_G = self.preconditioner(R_G, kpt.phase_cd)

                # New search direction
                gamma = self.comm.sum(np.vdot(pR_G, R_G).real)
                phi_G[:] = -pR_G - gamma/gamma_old * phi_old_G
                gamma_old = gamma
                phi_old_G[:] = phi_G[:]
                
                # Calculate projections
                P2_ai = wfs.pt.dict()
                wfs.pt.integrate(phi_G, P2_ai, kpt.q)

                # Orthonormalize phi_G to all bands
                self.timer.start('CG: orthonormalize')
                for nn in range(self.nbands):
                    overlap = np.vdot(kpt.psit_nG[nn], phi_G) * self.gd.dv
                    for a, P2_i in P2_ai.items():
                        P_i = kpt.P_ani[a][nn]
                        dO_ii = wfs.setups[a].O_ii
                        overlap += np.vdot(P_i, np.inner(dO_ii, P2_i))
                    overlap = self.comm.sum(overlap)
                    # phi_G -= overlap * kpt.psit_nG[nn]
                    axpy(-overlap, kpt.psit_nG[nn], phi_G)
                    for a, P2_i in P2_ai.items():
                        P_i = kpt.P_ani[a][nn]
                        P2_i -= P_i * overlap

                norm = np.vdot(phi_G, phi_G) * self.gd.dv
                for a, P2_i in P2_ai.items():
                    dO_ii = wfs.setups[a].O_ii
                    norm += np.vdot(P2_i, np.inner(dO_ii, P2_i))
                norm = self.comm.sum(norm)
                phi_G /= sqrt(norm.real)
                for P2_i in P2_ai.values():
                    P2_i /= sqrt(norm.real)
                self.timer.stop('CG: orthonormalize')
                    
                #find optimum linear combination of psit_G and phi_G
                an = kpt.eps_n[n]
                wfs.kin.apply(phi_G, Htphi_G, kpt.phase_cd)
                Htphi_G += phi_G * vt_G
                b = np.vdot(phi_G, Htpsit_G) * self.gd.dv
                c = np.vdot(phi_G, Htphi_G) * self.gd.dv
                for a, P2_i in P2_ai.items():
                    P_i = kpt.P_ani[a][n]
                    dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                    b += dot(P2_i, dot(dH_ii, P_i.conj()))
                    c += dot(P2_i, dot(dH_ii, P2_i.conj()))
                b = self.comm.sum(b)
                c = self.comm.sum(c)

                theta = 0.5 * atan2(2 * b.real, (an - c).real)
                enew = (an * cos(theta)**2 +
                        c * sin(theta)**2 +
                        b * sin(2.0 * theta)).real
                # theta can correspond either minimum or maximum
                if ( enew - kpt.eps_n[n] ) > 0.0: #we were at maximum
                    theta += pi / 2.0
                    enew = (an * cos(theta)**2 +
                            c * sin(theta)**2 +
                            b * sin(2.0 * theta)).real

                kpt.eps_n[n] = enew
                kpt.psit_nG[n] *= cos(theta)
                # kpt.psit_nG[n] += sin(theta) * phi_G
                axpy(sin(theta), phi_G, kpt.psit_nG[n])
                for a, P2_i in P2_ai.items():
                    P_i = kpt.P_ani[a][n]
                    P_i *= cos(theta)
                    P_i += sin(theta) * P2_i

                if nit < niter - 1:
                    Htpsit_G *= cos(theta)
                    # Htpsit_G += sin(theta) * Htphi_G
                    axpy(sin(theta), Htphi_G, Htpsit_G)
                    #adjust residuals
                    R_G[:] = Htpsit_G - kpt.eps_n[n] * kpt.psit_nG[n]

                    coef_ai = wfs.pt.dict()
                    for a, coef_i in coef_ai.items():
                        P_i = kpt.P_ani[a][n]
                        dO_ii = wfs.setups[a].O_ii
                        dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                        coef_i[:] = (dot(P_i, dH_ii) -
                                     dot(P_i * kpt.eps_n[n], dO_ii))
                    wfs.pt.add(R_G, coef_ai, kpt.q)
                    error_new = self.comm.sum(np.vdot(R_G, R_G).real)
                    if error_new / error < 0.30:
                        # print >> self.f, "cg:iters", n, nit+1
                        break
                    if (self.nbands_converge == 'occupied' and
                        kpt.f_n is not None and kpt.f_n[n] == 0.0):
                        # print >> self.f, "cg:iters", n, nit+1
                        break
                    error = error_new

            if kpt.f_n is None:
                weight = 1.0
            else:
                weight = kpt.f_n[n]
            if self.nbands_converge != 'occupied':
                weight = kpt.weight * float(n < self.nbands_converge)
            total_error += weight * error
            # if nit == 3:
            #   print >> self.f, "cg:iters", n, nit+1
                
        self.timer.stop('CG')
        return total_error
        
