"""Module defining  ``Eigensolver`` classes."""

from math import pi, sqrt, sin, cos, atan2

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!
import LinearAlgebra as linalg

from gpaw.utilities.blas import axpy, rk, r2k, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import unpack
from gpaw.eigensolvers import Eigensolver
from gpaw.mpi import run


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

    def __init__(self, timer, kpt_comm, gd, kin, typecode, nbands):

        Eigensolver.__init__(self, timer, kpt_comm, gd, kin, typecode, nbands)

        # Allocate arrays
        self.phi_G = gd.empty(typecode=typecode)
        self.phi_old_G = gd.empty(typecode=typecode)

        # self.f = open('CG_debug','w')

    def iterate_one_k_point(self, hamiltonian, kpt, niter=4):      
        """Do a conjugate gradient iterations for the kpoint"""
    
        phi_G = self.phi_G
        phi_old_G = self.phi_old_G

        self.diagonalize(hamiltonian, kpt)
            
        R_nG = self.work
        Htphi_G = self.work[0]

        R_nG[:] = self.Htpsit_nG
        self.timer.start('Residuals')        
        # optimize XXX 
        for R_G, eps, psit_G in zip(R_nG, kpt.eps_n, kpt.psit_nG):
            axpy(-eps, psit_G, R_G)  # R_G -= eps * psit_G

        run([nucleus.adjust_residual(R_nG, kpt.eps_n, kpt.s, kpt.u, kpt.k)
             for nucleus in hamiltonian.pt_nuclei])
        self.timer.stop()

        self.timer.start('CG')
        vt_G = hamiltonian.vt_sG[kpt.s]

        total_error = 0.0
        for n in range(kpt.nbands):
            R_G = R_nG[n]
            Htpsit_G = self.Htpsit_nG[n]
            gamma_old = 1.0
            phi_old_G[:] = 0.0
            error = self.comm.sum(real(num.vdot(R_G, R_G)))
            for nit in range(niter):
                if error < self.tolerance:
                    # print >> self.f, "cg:iters", n, nit
                    break

                pR_G = self.preconditioner(R_G, kpt.phase_cd, kpt.psit_nG[n],
                                           kpt.k_c)

                # New search direction
                gamma = self.comm.sum(real(num.vdot(pR_G, R_G)))
                phi_G[:] = -pR_G - gamma/gamma_old * phi_old_G
                gamma_old = gamma
                phi_old_G[:] = phi_G[:]
                
                # Calculate projections
                for nucleus in hamiltonian.pt_nuclei:
                    ni = nucleus.get_number_of_partial_waves()
                    nucleus.P2_i = num.zeros(ni, self.typecode)
                    if nucleus.in_this_domain:
                        nucleus.pt_i.integrate(phi_G, nucleus.P2_i, kpt.k)
                    else:
                        nucleus.pt_i.integrate(phi_G, None, kpt.k)

                # Orthonormalize phi_G to all bands
                self.timer.start('CG: orthonormalize')
                for nn in range(kpt.nbands):
                    overlap = num.vdot(kpt.psit_nG[nn], phi_G) * self.gd.dv
                    for nucleus in hamiltonian.my_nuclei:
                        P2_i = nucleus.P2_i
                        P_i = nucleus.P_uni[kpt.u, nn]
                        overlap += num.vdot(P_i, inner(nucleus.setup.O_ii, P2_i))
                    overlap = self.comm.sum(overlap)
                    # phi_G -= overlap * kpt.psit_nG[nn]
                    axpy(-overlap, kpt.psit_nG[nn], phi_G)
                    for nucleus in hamiltonian.my_nuclei:
                        nucleus.P2_i -= nucleus.P_uni[kpt.u, nn] * overlap

                norm = num.vdot(phi_G, phi_G) * self.gd.dv
                for nucleus in hamiltonian.my_nuclei:
                    norm += num.vdot(nucleus.P2_i, inner(nucleus.setup.O_ii, nucleus.P2_i))
                norm = self.comm.sum(norm)
                phi_G /= sqrt(real(norm))
                for nucleus in hamiltonian.my_nuclei:
                    nucleus.P2_i /= sqrt(real(norm))
                self.timer.stop()
                    
                #find optimum linear combination of psit_G and phi_G
                a = kpt.eps_n[n]

                hamiltonian.kin.apply(phi_G, Htphi_G, kpt.phase_cd)
                Htphi_G += phi_G * vt_G
                b = num.vdot(phi_G, Htpsit_G) * self.gd.dv
                c = num.vdot(phi_G, Htphi_G) * self.gd.dv
                for nucleus in hamiltonian.my_nuclei:
                    P_i = nucleus.P_uni[kpt.u, n]
                    P2_i = nucleus.P2_i
                    b += num.dot(P2_i, num.dot(unpack(nucleus.H_sp[kpt.s]),
                                                cc(P_i)))
                    c += num.dot(P2_i, num.dot(unpack(nucleus.H_sp[kpt.s]),
                                                cc(P2_i)))
                b = self.comm.sum(b)
                c = self.comm.sum(c)

                theta = 0.5*atan2(real(2*b), real(a-c))
                enew = real(a*cos(theta)**2 + c*sin(theta)**2 + b*sin(2.0*theta))
                # theta can correspond either minimum or maximum
                if ( enew - kpt.eps_n[n] ) > 0.00: #we were at maximum
                    theta += pi/2.0
                    enew = real(a*cos(theta)**2 + c*sin(theta)**2+b*sin(2.0*theta))
                kpt.eps_n[n] = enew
                kpt.psit_nG[n] *= cos(theta)
                # kpt.psit_nG[n] += sin(theta) * phi_G
                axpy(sin(theta), phi_G, kpt.psit_nG[n])
                for nucleus in hamiltonian.my_nuclei:
                    nucleus.P_uni[kpt.u, n] *= cos(theta)
                    nucleus.P_uni[kpt.u, n] += sin(theta) * nucleus.P2_i

                if nit < niter - 1:
                    Htpsit_G *= cos(theta)
                    # Htpsit_G += sin(theta) * Htphi_G
                    axpy(sin(theta), Htphi_G, Htpsit_G)
                    #adjust residuals
                    R_G[:] = Htpsit_G - kpt.eps_n[n] * kpt.psit_nG[n]

                    for nucleus in hamiltonian.pt_nuclei:
                        if nucleus.in_this_domain:
                            H_ii = unpack(nucleus.H_sp[kpt.s])
                            P_i = nucleus.P_uni[kpt.u, n]
                            coefs_i =  (num.dot(P_i, H_ii) -
                                        num.dot(P_i * kpt.eps_n[n], nucleus.setup.O_ii))
                            nucleus.pt_i.add(R_G, coefs_i, kpt.k, communicate=True)
                        else:
                            nucleus.pt_i.add(R_G, None, kpt.k, communicate=True)
                    error_new = self.comm.sum(real(num.vdot(R_G, R_G)))
                    if error_new / error < 0.30:
                        # print >> self.f, "cg:iters", n, nit+1
                        break
                    if not self.convergeall and kpt.f_n[n] == 0.0:
                        # print >> self.f, "cg:iters", n, nit+1
                        break
                    error = error_new

            weight = kpt.f_n[n]
            if self.convergeall: weight = 1.
            total_error += weight * error
            # if nit == 3:
            #   print >> self.f, "cg:iters", n, nit+1
                
        self.timer.stop()
        return total_error
        
