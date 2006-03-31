# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi
import sys

import Numeric as num

from gridpaw.transformers import Interpolator, Restrictor
from gridpaw.operators import Laplace, LaplaceA, LaplaceB
from gridpaw.utilities.gauss import Gaussian

class PoissonSolver:
    def __init__(self, gd, out=sys.stdout):
        self.gd = gd
        scale = -0.25 / pi 
        print >> out, 'poisson solver:'
        self.dv = gd.dv
        self.operators = [LaplaceA(gd, -scale)]
        self.B = LaplaceB(gd)
        self.rhos = [gd.new_array()]
        self.phis = [None]
        self.residuals = [gd.new_array()]
        self.interpolators = [Interpolator(gd, 1)]
        self.restrictors = [Restrictor(gd, 1)]

        level = 0
        self.presmooths=[2]
        self.postsmooths=[1]
        while level < 4:
            try:
                gd = gd.coarsen()
            except ValueError:
                break
            self.operators.append(Laplace(gd, scale, 1))
            self.phis.append(gd.new_array())
            self.rhos.append(gd.new_array())
            self.residuals.append(gd.new_array())
            self.interpolators.append(Interpolator(gd, 1))
            self.restrictors.append(Restrictor(gd, 1))
            self.presmooths.append(4)
            self.postsmooths.append(4)
            level += 1
            print >> out, level, gd.N_c
                    
        self.levels = level
        self.step = 0.66666666 / self.operators[0].get_diagonal_element()
        self.presmooths[level]=8
        self.postsmooths[level]=8
        
    def solve(self, phi, rho, eps=1e-9, charge=False):
        self.phis[0] = phi
        if charge != False:
            phi_gauss = self.neutralize(rho, charge)

        self.B.apply(rho, self.rhos[0])
        niter = 1
        while self.iterate2(self.step) > eps and niter < 300:
            niter += 1
        if niter == 300:
##        if niter == 3000:
            charge = num.sum(rho.flat) * self.dv
            print 'CHARGE:', charge
            raise RuntimeError('Poisson solver did not converge!')

        if charge != False:
            phi += phi_gauss
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
            error = self.gd.comm.sum(num.vdot(residual, residual))
            if niter == 1 and level < self.levels:
                self.restrictors[level].apply(residual, self.rhos[level + 1])
                self.phis[level + 1][:] = 0.0
                self.iterate(4.0 * step, level + 1)
                self.interpolators[level + 1].apply(self.phis[level + 1],
                                                    residual)
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
            self.operators[level].relax(self.phis[level],self.rhos[level],
                                        self.presmooths[level])
            self.operators[level].apply(self.phis[level], residual)
            residual -= self.rhos[level]
            self.restrictors[level].apply(residual,
                                          self.rhos[level + 1])
            self.phis[level + 1][:] = 0.0
            self.iterate2(4.0 * step, level + 1)
            self.interpolators[level + 1].apply(self.phis[level + 1],
                                                residual)
            self.phis[level] -= residual

        self.operators[level].relax(self.phis[level],self.rhos[level],
                                    self.postsmooths[level])
        if level == 0:
            self.operators[level].apply(self.phis[level], residual)
            residual -= self.rhos[level]
            error = self.gd.domain.comm.sum(num.dot(residual.flat,
                                                residual.flat))*self.dv
            return error

    def neutralize(self, rho, charge):
        if charge == None:
            charge = self.gd.integrate(rho)
            
        if abs(charge) < 1e-7:
            return 0

        if not hasattr(self, 'ng'):
            gauss = Gaussian(self.gd)
            self.ng = gauss.get_gauss(0) / (2 * num.sqrt(pi))
            self.vg = gauss.get_gauss_pot(0) / (2 * num.sqrt(pi))

        rho -= charge * self.ng

        return charge * self.vg
