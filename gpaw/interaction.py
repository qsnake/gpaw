# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi, sqrt

import sys
import numpy as npy

from gpaw.gauss import I
from gpaw.spherical_harmonics import YL
from gpaw.utilities import fac, warning

GAUSS = False


d_l = [fac[l] * 2**(2 * l + 2) / sqrt(pi) / fac[2 * l + 1]
       for l in range(3)]

class GInteraction2:
    def __init__(self, setupa, setupb):
        self.softgauss = setupa.softgauss
        self.alpha = setupa.alpha
        self.beta = setupb.alpha
        self.alpha2 = setupa.alpha2
        self.beta2 = setupb.alpha2
        self.lmaxa = setupa.lmax
        self.lmaxb = setupb.lmax
        self.v_LL = npy.zeros(((self.lmaxa + 1)**2, (self.lmaxb + 1)**2))
        self.dvdr_LLc = npy.zeros(((self.lmaxa + 1)**2,
                                  (self.lmaxb + 1)**2,
                                  3))

##         rcutcomp = setupa.rcutcomp + setupb.rcutcomp
##         rcutfilter = setupa.rcutfilter + setupb.rcutfilter
##         rcutproj = max(setupa.rcut_j) + max(setupb.rcut_j)
##         rcore = setupa.rcore + setupb.rcore
##         self.cutoffs = ('Summed cutoffs: %4.2f(comp), %4.2f(filt), '
##                         '%4.2f(core), %4.2f(proj) Bohr' % (
##             rcutcomp, rcutfilter, rcore, rcutproj))
##         self.mindist = rcutproj - .3

    def __call__(self, R):
##         dist = sqrt(npy.sum(R**2))
##         if dist > 0 and dist < self.mindist:
##             print >> sys.stderr, warning('Atomic distance: %4.2f Bohr.\n%s' % (
##                 dist, self.cutoffs))

        if not self.softgauss:
            return (self.v_LL, -self.dvdr_LLc)
        for la in range(self.lmaxa + 1):
            for ma in range(2 * la + 1):
                La = la**2 + ma
                for lb in range(self.lmaxb + 1):
                    for mb in range(2 * lb + 1):
                        Lb = lb**2 + mb
                        f = npy.zeros(4)
                        f2 = npy.zeros(4)
                        for ca, xa in YL[La]:
                            for cb, xb in YL[Lb]:
                                f += ca * cb * I(R, xa, xb,
                                                 self.alpha, self.beta)
                                f2 += ca * cb * I(R, xa, xb,
                                                 self.alpha2, self.beta2)
                        x = d_l[la] * d_l[lb]
                        f *= x * self.alpha**(1.5 + la) * \
                                 self.beta**(1.5 + lb)
                        f2 *= x * self.alpha2**(1.5 + la) * \
                                  self.beta2**(1.5 + lb)
##                         if npy.sometrue(R):
##                             assert npy.dot(R, R) > 0.1
##                         else:
##                             f[:] = 0.0
##                             if La == Lb:
##                                 f[0] = I_l[la] / self.rcut**(2 * la + 1)

                        self.v_LL[La, Lb] = f[0] - f2[0]
                        self.dvdr_LLc[La, Lb] = f[1:] - f2[1:]
        return (self.v_LL, -self.dvdr_LLc)
