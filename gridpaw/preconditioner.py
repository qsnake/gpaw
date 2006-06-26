# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi

import Numeric as num

from gridpaw.transformers import Restrictor, Interpolator
from gridpaw.operators import Laplace


class Preconditioner:
    def __init__(self, gd0, kin0, typecode):
        gd1 = gd0.coarsen()
        gd2 = gd1.coarsen()
        self.kin0 = kin0
        self.kin1 = Laplace(gd1, -0.5, 1, typecode)
        self.kin2 = Laplace(gd2, -0.5, 1, typecode)
        self.scratch0 = gd0.new_array(2, typecode)
        self.scratch1 = gd1.new_array(3, typecode)
        self.scratch2 = gd2.new_array(3, typecode)
        self.step = 0.66666666 / kin0.get_diagonal_element()
        self.restrictor0 = Restrictor(gd0, 1, typecode).apply
        self.restrictor1 = Restrictor(gd1, 1, typecode).apply
        self.interpolator2 = Interpolator(gd2, 1, typecode).apply
        self.interpolator1 = Interpolator(gd1, 1, typecode).apply
        
    def __call__(self, residual, phases, phit, kpt):
        step = self.step
        d0, q0 = self.scratch0
        r1, d1, q1 = self.scratch1
        r2, d2, q2 = self.scratch2
        self.restrictor0(-residual, r1, phases)
        d1[:] = 4 * step * r1
        self.kin1.apply(d1, q1, phases)
        q1 -= r1
        self.restrictor1(q1, r2, phases)
        d2[:] = 16 * step * r2
        self.kin2.apply(d2, q2, phases)
        q2 -= r2
        d2 -= 16 * step * q2
        self.interpolator2(d2, q1, phases)
        d1 -= q1
        self.kin1.apply(d1, q1, phases)
        q1 -= r1
        d1 -= 4 * step * q1
        self.interpolator1(-d1, d0, phases)
        self.kin0.apply(d0, q0, phases)
        q0 -= residual
        d0 -= step * q0
        d0 *= -1.0
        return d0


class Teter:
    def __init__(self, gd, kin, typecode):
        print 'Teter, Payne and Allan FFT Preconditioning of residue vector'
        self.typecode = typecode
        dims = gd.n_c.copy()
        dims.shape = (3, 1, 1, 1)
        icell = 1.0 / num.array(gd.domain.cell_c)
        icell.shape = (3, 1, 1, 1)
        q_cq = ((num.indices(gd.n_c) + dims / 2) % dims - dims / 2) * icell
        self.q2_q = num.sum(q_cq**2)

        self.r_cG = num.indices(gd.n_c, num.Float) / dims
        self.r_cG.shape = (3, -1)

        self.cache = {}
        
    def __call__(self, R_G, phases, phit_G, kpt_c):
        from FFT import fftnd, inverse_fftnd
        if kpt_c is None:
            phit_q = fftnd(phit_G)
        else:
            phase_G = self.cache.get(kpt_c)
            if phase_G is None:
                phase_G = num.exp(-2j * pi * num.dot(kpt_c, self.r_cG))
                phase_G.shape = phit_G.shape
                self.cache[kpt_c] = phase_G
            phit_q = fftnd(phit_G * phase_G)
            
        norm = num.vdot(phit_q, phit_q)
        h_q = phit_q * num.conjugate(phit_q) * self.q2_q / norm
        ekin = num.sum(h_q.flat)
        x_q = self.q2_q / ekin
        
        K_q = x_q * 8.0
        K_q += 12.0
        K_q *= x_q
        K_q += 18.0
        K_q *= x_q
        K_q += 27.0
        K_q /= (K_q + 16.0 * x_q**4)
       
        if kpt_c is None:
            return inverse_fftnd(K_q * fftnd(R_G)).astype(num.Float)
        else:
            return inverse_fftnd(K_q * fftnd(phase_G * R_G)) / phase_G
