# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi
import sys

import Numeric as num

from gridpaw.transformers import Interpolator, Restrictor
from gridpaw.operators import Laplace, LaplaceA, LaplaceB


class PoissonSolver:
    def __init__(self, gd, out=sys.stdout):
        self.gd = gd
        scale = -0.25 / pi 
        print >> out, 'poisson solver:'
        self.dv = gd.dv
        self.AB = True
        if self.AB:
            self.operators = [LaplaceA(gd, -scale)]
            self.B = LaplaceB(gd)
            self.rhos = [gd.array()]
        else:
            self.operators = [Laplace(gd, scale, 3)]
            self.rhos = [None]
            
        self.phis = [None]
        self.residuals = [gd.array()]
        self.interpolators = [Interpolator(gd, 1)]
        self.restrictors = [Restrictor(gd, 1)]

        level = 0
        while level < 4:
            try:
                gd = gd.coarsen()
            except ValueError:
                break
            self.operators.append(Laplace(gd, scale, 1))
            self.phis.append(gd.array())
            self.rhos.append(gd.array())
            self.residuals.append(gd.array())
            self.interpolators.append(Interpolator(gd, 1))
            self.restrictors.append(Restrictor(gd, 1))
            level += 1
            print >> out, level, gd.ng
            
        self.levels = level
        self.eps = 1e-9
        self.step = 0.66666666 / self.operators[0].get_diagonal_element()
        
    def solve(self, phi, rho):
        self.phis[0] = phi

        if self.AB:
            self.B.apply(rho, self.rhos[0])
        else:
            self.rhos[0] = rho

        niter = 1


        while self.iterate(self.step) > self.eps and niter < 3000:
            niter += 1
##        if niter == 300:
        if niter == 3000:
            charge = num.sum(rho.flat) * self.dv
            print 'CHARGE:', charge
            raise RuntimeError('Poisson solver did not converge!')
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
            error = self.gd.domain.comm.sum(num.dot(residual.flat,
                                                    residual.flat))
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
