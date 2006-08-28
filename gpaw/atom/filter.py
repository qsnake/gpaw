# Copyright (C) 2006  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi, log, sqrt

import Numeric as num
from FFT import real_fft, inverse_real_fft

from gpaw.utilities import fac

"""Fourier filtering

This module is an implementation of this Fourier filtering scheme:

*A general and efficient pseudopotential Fourier filtering scheme for
real space methods using mask functions*, Maxim Tafipolsky, Rochus
Schmid, J Chem Phys. 2006 May 7;124:174102.

Only difference is that we use a gaussian for the mask function.  The
filtering is used for the projector functions and for the zero
potential."""

# 3D-Fourier transform:
#
#                 /     _ _
# ~         ^    |  _  iq.r          ^
# f (q) Y  (q) = | dr e    f (r) Y  (r)
#  l     lm      |          l     lm
#               /
#
# Radial part:
#
#                  /
#   ~        __ l |  2
#   f (q) = 4||i  | r dr j (qr) f (r)
#    l            |           l      l
#                /
#

class Filter:
    """Mask function Fourier filter"""
    
    def __init__(self, r_g, dr_g, rcut, h):
        """Construct filter.

        The radial grid is defined by r(g) and dr/dg(g) (`r_g` and
        `dr_g`), `rcut` is the cutoff radius, and `h` is the target
        grid spacing used in the calculation."""

        for g, r in enumerate(r_g):
            if r > rcut:
                self.gcut = gcut = g
                break
            
        N = 200
        self.r_g = r_g = r_g[:gcut].copy()  # will be modified later!
        self.dr_g = dr_g[:gcut]

        # Matrices for Bessel transform:
        q1 = 5 * pi / h / N
        self.q_i = q_i = q1 * num.arange(N)
        self.c = sqrt(2 * q1 / pi) 
        self.sinqr_ig = num.sin(q_i[:, None] * r_g) * self.c
        self.cosqr_ig = num.cos(q_i[:, None] * r_g) * self.c

        # Cutoff function:
        qmax = pi / h
        alpha = 1.1
        qcut = qmax / alpha
        icut = 1 + int(qcut / q1)
        beta = 5 * log(10) / (alpha - 1.0)**2
        self.cut_i = num.ones(N, num.Float)
        self.cut_i[icut:] = num.exp(
            -num.clip(0, 400, beta * (q_i[icut:] / qcut - 1.0)**2))

        # Mask function:
        gamma = 3 * log(10) / rcut**2
        self.m_g = num.exp(-gamma * r_g**2)
        
        # We will need to divide by these two!  Remove zeros:
        q_i[0] = 1.0
        r_g[0] = 1.0

    def filter(self, f_g, l=0):
        """Filter radial function.

        The function to be filtered is::

          f(r)     ^
          ---- Y  (r)
           r    lm
           
        Output is::

                l     ^
          g(r) r  Y  (r),
                   lm

        where the filtered radial part ``g(r)`` is returned."""
        
        r_g = self.r_g
        q_i = self.q_i

        fdrim_g = f_g[:self.gcut] * self.dr_g / self.m_g / r_g

        #         sin(x)
        # j (x) = ------,
        #  0        x
        #
        #         sin(x)   cos(x)
        # j (x) = ------ - ------,
        #  1         2       x
        #           x
        #
        #           3    1            3
        # j (x) = (--- - -) sin(x) - --- cos(x).
        #  2         3   x             2
        #           x                 x

        if l == 0:
            fq_i = num.dot(self.sinqr_ig, fdrim_g * r_g) * self.cut_i
            fr_g = num.dot(fq_i, self.sinqr_ig)
        elif l == 1:
            fq_i = num.dot(self.sinqr_ig, fdrim_g) / q_i
            fq_i -= num.dot(self.cosqr_ig, r_g * fdrim_g)
            fq_i[0] = 0.0
            fq_i *= self.cut_i
            fr_g = num.dot(fq_i / q_i, self.sinqr_ig) / r_g
            fr_g -= num.dot(fq_i, self.cosqr_ig)
        elif l == 2:
            fq_i = 3 * num.dot(self.sinqr_ig, fdrim_g / r_g) / q_i**2
            fq_i -= num.dot(self.sinqr_ig, fdrim_g * r_g)
            fq_i -= 3 * num.dot(self.cosqr_ig, fdrim_g) / q_i
            fq_i[0] = 0.0
            fq_i *= self.cut_i
            fr_g = 3 * num.dot(fq_i / q_i**2, self.sinqr_ig) / r_g**2
            fr_g -= num.dot(fq_i, self.sinqr_ig)
            fr_g -= 3 * num.dot(fq_i / q_i, self.cosqr_ig) / r_g
        else:
            raise NotImplementedError
    
        a_g = num.zeros(len(f_g), num.Float)
        a_g[:self.gcut] = fr_g * self.m_g / r_g**(l + 1)
        
        #            n 
        #           2 n!     n
        # j (x) = --------- x   for  x << 1.
        #  n      (2n + 1)! 
        #
        # This formula is used for finding the value of
        #
        #       -l
        # f(r) r    for r -> 0
        #
        c = 2.0**l * fac[l] / fac[2 * l + 1] * self.c
        a_g[0] = num.dot(fq_i, q_i**(l + 1)) * c

        return a_g

if __name__ == '__main__':
    rc = 1.1
    gamma = 1.95
    rc2 = rc * gamma
    M = 300
    beta = 0.3
    gcut = 1 + int(M * rc / (beta + rc))
    g_g = num.arange(M)
    r_g = beta * g_g / (M - g_g)
    drdg_g = beta * M / (M - g_g)**2

    x_g = r_g / rc
    p_g = 1 - x_g**2 * (3 - 2 * x_g)
    p_g[gcut:] = 0.0
    #p_g = num.exp(-num.clip(5.0 * r_g**2, 0, 400))

    h = 0.4
    f = Filter(r_g, drdg_g, rc2, h)
    pf0_g = f.filter(p_g)
    pf1_g = f.filter(p_g * r_g**1, 1)
    pf2_g = f.filter(p_g * r_g**2, 2)

    if 0:
        for i in range(200):
            print 5 * pi / h * i / 200, pf0_g[i], pf1_g[i], pf2_g[i]
    if 1:
        for r, p, pf0, pf1, pf2 in zip(r_g, p_g, pf0_g, pf1_g, pf2_g):
            print r, p, pf0, pf1, pf2
            if r > rc2:
                break
