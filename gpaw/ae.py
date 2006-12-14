# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""All-electron calculations with GPAW.

This module implements a special type of Setup object that allows us
to do all-electron calculations for hydrogen atoms - all the PAW stuff
is turned off.  Therefore, the densities will have sharp cusps at the
nuclei, and dense grids are needed for accurate results.

This feature is only useful for testing!
"""

from math import pi, sqrt

import Numeric as num

from gpaw.spline import Spline
from gpaw.utilities import erf


class AllElectronSetup:
    def __init__(self, symbol, xcfunc, nspins):
        assert symbol == 'H'
        self.symbol = symbol + '.ae'
        self.xcname = xcfunc.get_name()
        self.softgauss = True
        
        self.fingerprint = ''

        self.Nv = 1
        self.Nc = 0
        self.Z = 1
        self.X_p = num.array([0.])
        self.ExxC = 0.

        self.n_j = [1]
        self.l_j = [0]
        self.f_j = [1]
        self.eps_j = []

        ng = 150
        beta = 0.4
        rcut = 0.9
        rcut2 = 2 * rcut
        gcut = 1 + int(rcut * ng / (rcut + beta))
        gcut2 = 1 + int(rcut2 * ng / (rcut2 + beta))

        g = num.arange(ng, typecode=num.Float)
        r_g = beta * g / (ng - g)

        self.ni = 1
        self.niAO = 1

        spline0 = Spline(0, 0.5, [0.0, 0.0, 0.0])
        
        # Construct splines:
        self.nct = spline0
        self.vbar = spline0

        # Step function:
        stepf = sqrt(4 * pi) * num.ones(ng)
        stepf[gcut:] = 0.0
        self.stepf = Spline(0, rcut2, stepf, r_g=r_g, beta=beta)

        self.pt_j = [spline0]

        # Cutoff for atomic orbitals used for initial guess:
        rcut3 = 8.0
        gcut3 = 1 + int(rcut3 * ng / (rcut3 + beta))
        self.phit_j = [Spline(0, rcut3, 2 * num.exp(-r_g), r_g=r_g, beta=beta)]

        self.Delta_pL = num.zeros((1, 1), num.Float)
        self.Delta0 = -1 / sqrt(4 * pi)
        self.MB = 0
        self.M_p = num.zeros(1, num.Float)
        self.MB_p = num.zeros(1, num.Float)
        self.M_pp = num.zeros((1, 1), num.Float)
        self.Kc = 0
        self.E = 0
        self.O_ii = num.zeros((1, 1), num.Float)
        self.K_p = num.zeros(1, num.Float)
        
        self.xc_correction = self
        self.xc = self
        self.xc.set_functional(xcfunc)
        
        self.lmax = 0

        rcutsoft = rcut2
        rcgauss = 0.02

        r = 0.02 * rcutsoft * num.arange(51, typecode=num.Float)

        alpha = rcgauss**-2
        alpha2 = 15.0 / rcutsoft**2
        self.M = -sqrt(alpha / 2 / pi)

        vt0 = 4 * pi * (num.array([erf(x) for x in sqrt(alpha) * r]) -
                        num.array([erf(x) for x in sqrt(alpha2) * r]))
        vt0[0] = 8 * sqrt(pi) * (sqrt(alpha) - sqrt(alpha2))
        vt0[1:] /= r[1:]
        vt0[-1] = 0.0
        self.vhat_l = [Spline(0, rcutsoft, vt0)]

        self.rcutsoft = rcutsoft
        self.rcut = rcut
        self.alpha = alpha
        self.alpha2 = alpha2

        d_0 = 4 / sqrt(pi)
        g = alpha2**1.5 * num.exp(-alpha2 * r**2)
        g[-1] = 0.0
        self.ghat_l = [Spline(0, rcutsoft, d_0 * g)]

    # xc_correction methods:
    def calculate_energy_and_derivatives(self, D_sp, H_sp):
        H_sp[:] = 0.0
        return 0.0
    def set_functional(self, xcfunc):
        self.xcfunc = xcfunc
    
    def print_info(self, out):
        print >> out, 'All-electron calculation.'
