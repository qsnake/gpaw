from math import pi, log, sqrt

import Numeric as num
from FFT import real_fft, inverse_real_fft

from gridpaw.utilities import fac

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
#          /
# ~       |  2
# f (q) = | r dr j (qr) f (r)  XXX ??
#  l      |       l      l
#        /
#

class Filter:
    """Mask function Fourier filter"""
    
    def __init__(self, r_g, dr_g, rcut, rcut2, h):
        """Construct filter.

        The radial grid is defined by r(g) and dr/dg(g) (`r_g` and
        `dr_g`), `rcut` is the original cutoff radius, `rcut2` is the
        new cutoff and `h` is the target grid spacing used in the
        calculation."""
        
        self.N = N = 256
        self.r_g = r_g.copy()  # will be modified later!
        self.dr_g = dr_g
        self.rcut = rcut

        # Matrices for Bessel transform:
        q1 = pi / rcut2
        self.q_i = q1 * num.arange(N + 1)
        self.sinqr_ig = num.sin(self.q_i[:, None] * r_g)
        self.cosqr_ig = num.cos(self.q_i[:, None] * r_g)

        # Cutoff function:
        qmax = pi / h
        alpha = 1.1
        qcut = qmax / alpha
        icut = 1 + int(qcut / q1)
        beta = 5 * log(10) / (alpha - 1.0)**2
        self.cut_i = num.ones(N + 1, num.Float)
        icut2 = int(rcut2 / h /alpha * (1 + sqrt(400 / beta)))  # numpy!
        self.cut_i[icut:icut2] = num.exp(
            -beta * (self.q_i[icut:icut2] / qcut - 1.0)**2)
        self.cut_i[icut2:] = 0.0

        # Mask function:
        gamma = 3 * log(10) / rcut2**2
        self.m_g = num.exp(-gamma * r_g**2)
        self.r_n = rcut2 / N * num.arange(2 * N)
        self.m_n = num.exp(-gamma * self.r_n**2)

        # We will need to divide by these three!  Remove zeros:
        self.q_i[0] = 1.0
        self.r_n[0] = 1.0
        self.r_g[0] = 1.0

    def filter(self, f_g, l=0):
        """Filter radial function.

        Input f(r) and angular momentum quantum number `l`, where::

                  l
          f(r) ~ r   for  r << 1

        Output is::

                -l
          f(r) r

        on a linear grid with N points in the interval [0, rcut2[.

        """
        
        r_g = self.r_g
        r_n = self.r_n
        q_i = self.q_i
        fdrim_g = f_g[:len(self.r_g)] * self.dr_g / self.m_g

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
            fr_n = -inverse_real_fft(1.0j * fq_i)
        elif l == 1:
            fq_i = num.dot(self.sinqr_ig, fdrim_g) / q_i
            fq_i -= num.dot(self.cosqr_ig, r_g * fdrim_g)
            fq_i[0] = 0.0
            fq_i *= self.cut_i
            fr_n = -inverse_real_fft(1.0j * fq_i / q_i) / r_n
            fr_n -= inverse_real_fft(fq_i)
        elif l == 2:
            fq_i = 3.0 * num.dot(self.sinqr_ig, fdrim_g / r_g) / q_i**2
            fq_i -= num.dot(self.sinqr_ig, fdrim_g * r_g)
            fq_i -= 3.0 * num.dot(self.cosqr_ig, fdrim_g) / q_i
            fq_i[0] = 0.0
            fq_i *= self.cut_i
            fr_n = -3.0 * inverse_real_fft(1.0j * fq_i / q_i**2) / r_n**2
            fr_n += inverse_real_fft(1.0j * fq_i)
            fr_n -= 3.0 * inverse_real_fft(fq_i / q_i) / r_n
        else:
            raise NotImplementedError

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
        c = 2.0**l * fac[l] / fac[2 * l + 1]
        fr_n[0] = num.dot(fq_i, q_i**(l + 1)) / self.N * c
        
        fr_n *= self.N / self.rcut * self.m_n
        return (fr_n / r_n**(l + 1))[:self.N]

if 0:
    rc = 1.1
    gamma = 1.95
    rc2 = rc * gamma
    M = 256
    beta = 0.3
    gcut = 1 + int(M * rc / (beta + rc))
    g_g = num.arange(gcut)
    r_g = beta * g_g / (M - g_g)
    drdg_g = beta * M / (M - g_g)**2

    x_g = r_g / rc
    p_g = 1 - x_g**2 * (3 - 2 * x_g)
    p_g[gcut:] = 0.0
    #p_g = num.exp(-5.0 * r_g**2)

    f = Filter(r_g, drdg_g, rc, rc2, 0.4)
    p_n = f.filter(p_g)
    p1_n = f.filter(p_g * r_g, l=1)
    p2_n = f.filter(p_g * r_g**2, l=2)

    for r, p in zip(r_g, p_g):
        print r, p
    print '&'
    for n in range(512):
        print n * rc2 / 512, p_n[n], p1_n[n], p2_n[n]
