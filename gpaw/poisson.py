# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi
import sys

import numpy as npy

from gpaw.transformers import Transformer
from gpaw.operators import Laplace, LaplaceA, LaplaceB
from gpaw import PoissonConvergenceError
from gpaw.utilities.blas import axpy
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.ewald import Ewald


class PoissonSolver:
    def __init__(self, nn='M', relax='GS', eps=2e-10):
        self.nn = nn
        self.eps = eps
        self.charged_periodic_correction = None
        self.maxiter = 1000
        
        # Relaxation method
        if relax == 'GS':
            # Gauss-Seidel
            self.relax_method = 1
        elif relax == 'J':
            # Jacobi
            self.relax_method = 2
        else:
            raise NotImplementedError('Relaxation method %s' % relax)

    def set_grid_descriptor(self, gd): # XXX
        # This method exists only because it is nice to know the gd size
        # before also having to allocate the arrays (which would be done in
        # the actual initialize).  Maybe 'initialize' should be 'allocate'
        self.gd = gd

    def initialize(self, load_gauss=False):
        gd = self.gd
        scale = -0.25 / pi 
        self.dv = gd.dv

        if self.nn == 'M':
            if (gd.cell_cv - npy.diag(gd.cell_cv.diagonal())).any():
                raise RuntimeError('Cannot use Mehrstellen stencil with non orthogonal cell.')

            self.operators = [LaplaceA(gd, -scale)]
            self.B = LaplaceB(gd)
        else:
            self.operators = [Laplace(gd, scale, self.nn)]
            self.B = None

        self.rhos = [gd.empty()]
        self.phis = [None]
        self.residuals = [gd.empty()]
        self.interpolators = []
        self.restrictors = []

        level = 0
        self.presmooths = [2]
        self.postsmooths = [1]
        
        # Weights for the relaxation,
        # only used if 'J' (Jacobi) is chosen as method
        self.weights = [2.0/3.0]
        
        while level < 4:
            try:
                gd2 = gd.coarsen()
            except ValueError:
                break
            self.operators.append(Laplace(gd2, scale, 1))
            self.phis.append(gd2.empty())
            self.rhos.append(gd2.empty())
            self.residuals.append(gd2.empty())
            self.interpolators.append(Transformer(gd2, gd))
            self.restrictors.append(Transformer(gd, gd2))
            self.presmooths.append(4)
            self.postsmooths.append(4)
            self.weights.append(1.0)
            level += 1
            gd = gd2

        self.levels = level
        self.step = 0.66666666 / self.operators[0].get_diagonal_element()
        self.presmooths[level] = 8
        self.postsmooths[level] = 8

        if load_gauss:
            self.load_gauss()

    def load_gauss(self):
        if not hasattr(self, 'rho_gauss'):
            gauss = Gaussian(self.gd)
            self.rho_gauss = gauss.get_gauss(0)
            self.phi_gauss = gauss.get_gauss_pot(0)

    def solve(self, phi, rho, eps=None, charge=0, maxcharge=1e-6,
              zero_initial_phi=False):

        if eps is None:
            eps = self.eps

        # Determine charge (if set to None)
        if charge == None:
            charge = self.gd.integrate(rho)

        if abs(charge) < maxcharge:
            # System is charge neutral. Use standard solver
            return self.solve_neutral(phi, rho, eps=eps)
        
        elif abs(charge) > maxcharge and self.gd.pbc_c.all():
            # System is charged and periodic. Subtract a homogeneous
            # background charge
            background = charge / npy.product(self.gd.cell_c)

            if self.charged_periodic_correction == None:
                print "+-----------------------------------------------------+"
                print "| Calculating charged periodic correction using the   |"
                print "| Ewald potential from a lattice of point charges in  |"
                print "| a homogenous background density                     |"
                print "+-----------------------------------------------------+"
                ewald = Ewald(self.gd.cell_cv)
                self.charged_periodic_correction = ewald.get_electrostatic_potential([.0,.0,.0], npy.array([[.0,.0,.0]]), [-1], 0)
                print "Potential shift will be ", \
                      self.charged_periodic_correction , "Ha."
                       
            # Set initial guess for potential
            if zero_initial_phi:
                phi[:] = 0.0
            else:
                phi -= charge * self.charged_periodic_correction
            
            iters = self.solve_neutral(phi, rho - background, eps=eps)
            phi += charge * self.charged_periodic_correction
            return iters            
        
        elif abs(charge) > maxcharge and not self.gd.pbc_c.any():
            # The system is charged and in a non-periodic unit cell.
            # Determine the potential by 1) subtract a gaussian from the
            # density, 2) determine potential from the neutralized density
            # and 3) add the potential from the gaussian density.

            # Load necessary attributes
            self.load_gauss()

            # Remove monopole moment
            q = charge / npy.sqrt(4 * pi) # Monopole moment
            rho_neutral = rho - q * self.rho_gauss # neutralized density

            # Set initial guess for potential
            if zero_initial_phi:
                phi[:] = 0.0
            else:
                axpy(-q, self.phi_gauss, phi) #phi -= q * self.phi_gauss

            # Determine potential from neutral density using standard solver
            niter = self.solve_neutral(phi, rho_neutral, eps=eps)

            # correct error introduced by removing monopole
            axpy(q, self.phi_gauss, phi) #phi += q * self.phi_gauss
            
            return niter
        else:
            # System is charged with mixed boundaryconditions
            raise NotImplementedError
    
    def solve_neutral(self, phi, rho, eps=2e-10):
        self.phis[0] = phi

        if self.B is None:
            self.rhos[0][:] = rho
        else:
            self.B.apply(rho, self.rhos[0])
        
        niter = 1
        maxiter = self.maxiter
        while self.iterate2(self.step) > eps and niter < maxiter:
            niter += 1
        if niter == maxiter:
            charge = npy.sum(rho.ravel()) * self.dv
            print 'CHARGE, eps:', charge, eps
            msg = 'Poisson solver did not converge in %d iterations!' % maxiter
            raise PoissonConvergenceError(msg)
        
        # Set the average potential to zero in periodic systems
        if npy.alltrue(self.gd.pbc_c):
            phi_ave = self.gd.comm.sum(npy.sum(phi.ravel()))
            N_c = self.gd.get_size_of_global_array()
            phi_ave /= npy.product(N_c)
            phi -= phi_ave

        return niter
    
    def iterate(self, step, level=0):
        residual = self.residuals[level]
        niter = 0
        while True:
            niter += 1
            if level > 0 and niter == 1:
                residual[:] = -self.rhos[level]
            else:
                self.operators[level].apply(self.phis[level], residual)
                residual -= self.rhos[level]
            error = self.gd.comm.sum(npy.vdot(residual, residual))
            if niter == 1 and level < self.levels:
                self.restrictors[level].apply(residual, self.rhos[level + 1])
                self.phis[level + 1][:] = 0.0
                self.iterate(4.0 * step, level + 1)
                self.interpolators[level].apply(self.phis[level + 1], residual)
                self.phis[level] -= residual
                continue
            residual *= step
            self.phis[level] -= residual
            if niter == 2:
                break
            
        return error
    
    def iterate2(self, step, level=0):
        """Smooths the solution in every multigrid level"""

        residual = self.residuals[level]

        if level < self.levels:
            self.operators[level].relax(self.relax_method,
                                        self.phis[level],
                                        self.rhos[level],
                                        self.presmooths[level],
                                        self.weights[level])

            self.operators[level].apply(self.phis[level], residual)
            residual -= self.rhos[level]
            self.restrictors[level].apply(residual,
                                          self.rhos[level + 1])
            self.phis[level + 1][:] = 0.0
            self.iterate2(4.0 * step, level + 1)
            self.interpolators[level].apply(self.phis[level + 1], residual)
            self.phis[level] -= residual

        self.operators[level].relax(self.relax_method,
                                    self.phis[level],
                                    self.rhos[level],
                                    self.postsmooths[level],
                                    self.weights[level])
        if level == 0:
            self.operators[level].apply(self.phis[level], residual)
            residual -= self.rhos[level]
            error = self.gd.comm.sum(npy.dot(residual.ravel(),
                                             residual.ravel())) * self.dv
            return error

    def estimate_memory(self, mem):
        # XXX Memory estimate works only for J or possibly GS !!

        # Poisson (rhos, phis, residuals, interpolators, restrictors)
        # Multigrid adds a factor of 1.14
        nfinebytes = self.gd.bytecount()
        nbytes = nfinebytes / 8
        mem.subnode('poisson1', 8 * 4 * 1.14 * nbytes)
        mem.subnode('poisson2', 6 * 1.14 * nbytes)
        mem.subnode('poisson3', 12 * 1.14 * nbytes)
        mem.subnode('Laplacian', nbytes)


from numpy.fft import fftn, ifftn
from gpaw.utilities.complex import real
from gpaw.utilities.tools import construct_reciprocal


class PoissonFFTSolver(PoissonSolver):
    def __init__(self):
        self.charged_periodic_correction = None

    """FFT implementation of the poisson solver"""
    def initialize(self, gd, load_gauss=False):
        # XXX this won't work now, but supposedly this class will be deprecated
        # in favour of FFTPoissonSolver, no?
        self.gd = gd
        if self.gd.comm.size > 1:
            raise RuntimeError('Cannot do parallel FFT.')
        self.k2, self.N3 = construct_reciprocal(self.gd)
        if load_gauss:
            gauss = Gaussian(self.gd)
            self.rho_gauss = gauss.get_gauss(0)
            self.phi_gauss = gauss.get_gauss_pot(0)

    def solve_neutral(self, phi, rho, eps=None):
        phi[:] = real(ifftn(fftn(rho) * 4 * pi / self.k2))
        return 1

    def solve_screened(self, phi, rho, screening=0):
        phi[:] = real(ifftn(fftn(rho) * 4 * pi / (self.k2 + screening**2)))
        return 1


class FFTPoissonSolver(PoissonSolver):
    """FFT poisson-solver for general unit cells."""
    
    relax_method = 0
    nn = 999
    
    def __init__(self, eps=2e-10):
        self.charged_periodic_correction = None
        self.eps = eps

    def set_grid_descriptor(self, gd):
        if gd.comm.size > 1:
            raise RuntimeError('Cannot do parallel FFT.')
        assert gd.pbc_c.all()
        PoissonSolver.set_grid_descriptor(self, gd)

    def initialize(self):
        self.k2_Q, self.N3 = construct_reciprocal(self.gd)

    def solve_neutral(self, phi_g, rho_g, eps=None):
        phi_g[:] = ifftn(fftn(rho_g) * 4.0 * pi / self.k2_Q).real
        return 1
