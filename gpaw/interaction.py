# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi, sqrt

import Numeric as num

from gpaw.gauss import I
from gpaw.spherical_harmonics import YL
from gpaw.polynomium import I_l
from gpaw.utilities import fac


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
        self.v_LL = num.zeros(((self.lmaxa + 1)**2, (self.lmaxb + 1)**2),
                              num.Float)
        self.dvdr_LLc = num.zeros(((self.lmaxa + 1)**2,
                                  (self.lmaxb + 1)**2,
                                  3),
                                  num.Float)

    def __call__(self, R):
        if not self.softgauss:
            return (self.v_LL, -self.dvdr_LLc)
        for la in range(self.lmaxa + 1):
            for ma in range(2 * la + 1):
                La = la**2 + ma
                for lb in range(self.lmaxb + 1):
                    for mb in range(2 * lb + 1):
                        Lb = lb**2 + mb
                        f = num.zeros(4, num.Float)
                        f2 = num.zeros(4, num.Float)
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
##                         if num.sometrue(R):
##                             assert num.dot(R, R) > 0.1
##                         else:
##                             f[:] = 0.0
##                             if La == Lb:
##                                 f[0] = I_l[la] / self.rcut**(2 * la + 1)

                        self.v_LL[La, Lb] = f[0] - f2[0]
                        self.dvdr_LLc[La, Lb] = f[1:] - f2[1:]
        return (self.v_LL, -self.dvdr_LLc)
