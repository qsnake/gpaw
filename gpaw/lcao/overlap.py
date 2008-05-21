from math import sqrt, pi

import numpy as npy
from numpy.fft import ifft

from gpaw.spline import Spline
from gpaw.spherical_harmonics import Y,Yl
from gpaw.gaunt import gaunt
from gpaw.utilities import fac


# Generate the coefficients for the Fourier-Bessel transform
C = []
a = 0.0
n = 5
for n in range(n):
    c = npy.zeros(n+1, complex)
    for s in range(n + 1):
        a = (1.0j)**s * fac[n + s] / (fac[s] * 2**s * fac[n - s])
        a *= (-1.0j)**(n + 1)
        c[s] = a
    C.append(c)

def fbt(l, f, r, k):
    """Fast Bessel transform.

    The following integral is calculated using 2l+1 FFT's::

                    oo
                   /
              l+1 |  2           l
      g(k) = k    | r dr j (kr) r f (r)
                  |       l
                 /
                  0
    """

    dr = r[1]
    m = len(k)
    g = npy.zeros(m)
    for n in range(l + 1):
        g += (dr * 2 * m * k**(l - n) *
              ifft(C[l][n] * f * r**(1 + l - n), 2 * m)[:m].real)
    return g

class TwoCenterIntegrals:
    """ Two-center integrals class.

    This class implements a Fourier-space calculation of two-center
    integrals.
    """

    def __init__(self, setups, ng):
        self.rcmax = 0.0
        for setup in setups:
            for phit in setup.phit_j:
                rc = phit.get_cutoff()
                if rc > self.rcmax:
                    self.rcmax = rc

        for setup in setups:
            for pt in setup.pt_j:
                rc = pt.get_cutoff()
                assert rc < self.rcmax

        self.ng = ng 
        self.dr = self.rcmax / self.ng
        self.r_g = npy.arange(self.ng) * self.dr
        self.Q = 4 * self.ng
        self.dk = 2 * pi / self.Q / self.dr
        self.k = npy.arange(self.Q // 2) * self.dk

        phit_g = npy.zeros(self.ng) 
        phit_jq = {}
        for setup in setups:
            for j, phit in enumerate(setup.phit_j):
                l = phit.get_angular_momentum_number()
                id = (setup.symbol, j)
                phit_g[0:self.ng] = [phit(r) for r in self.r_g[0:self.ng]]
                phit_q = fbt(l, phit_g, self.r_g, self.k)
                phit_jq[id] = (l, phit_q)

        pt_g = npy.zeros(self.ng) 
        pt_jq = {}
        for setup in setups:
            for j, pt in enumerate(setup.pt_j):
                l = pt.get_angular_momentum_number()
                id = (setup.symbol, j)
                pt_g[0:self.ng] = [pt(r) for r in self.r_g[0:self.ng]]
                pt_q = fbt(l, pt_g, self.r_g, self.k)
                pt_jq[id] = (l, pt_q)
                
        self.S = {}
        self.T = {}
        for id1, (l1, phit1_q) in phit_jq.items():
            for id2, (l2, phit2_q) in phit_jq.items():
                s = self.calculate_spline(phit1_q, phit2_q, l1, l2)
                self.S[(id1, id2)] = s
                t = self.calculate_spline(0.5 * phit1_q * self.k**2, phit2_q,
                                          l1, l2, kinetic_energy=True)
                self.T[(id1, id2)] = t
                
        self.P = {}
        for id1, (l1, pt1_q) in pt_jq.items():
            for id2, (l2, phit2_q) in phit_jq.items():
                p = self.calculate_spline(pt1_q, phit2_q, l1, l2)
                self.P[(id2, id1)] = p
                
        self.setups = setups # XXX

    def calculate_spline(self, phit1, phit2, l1, l2, kinetic_energy=False):
        S_g = npy.zeros(2 * self.ng)
        self.lmax = l1 + l2
        splines = []
        R = npy.arange(self.Q // 2) * self.dr
        R1 = R.copy()
        R1[0] = 1.0
        k1 = self.k.copy()
        k1[0] = 1.0
        for l in range(self.lmax % 2, self.lmax + 1, 2):
            S_g[:] = 0.0
            a_q = (phit1 * phit2)
            a_g = (8 * fbt(l, a_q * k1**(-2 - l1 - l2 - l), self.k, R) /
                   R1**(2 * l + 1))
            if l==0:
                a_g[0] = 8 * npy.sum(a_q * k1**(-l1 - l2)) * self.dk
            else:
                a_g[0] = a_g[1]  # XXXX
            a_g *= (-1)**((-l1 + l2 - l) / 2)
            S_g += a_g
            s = Spline(l, self.Q // self.ng / 2 * self.rcmax, S_g)
            splines.append(s)
        return splines

    def p(self, ida, idb, la, lb, r, R, rlY_lm, drlYdR_lmc):
        """ Returns the overlap between basis functions and projector
        functions. """
        
        derivative = bool(drlYdR_lmc)

        p_mi = npy.zeros((2 * la + 1, 2 * lb + 1))
        dPdR_cmi = None
        if derivative:
            dPdR_cmi = npy.zeros((3, 2 * la + 1, 2 * lb + 1))

        l = (la + lb) % 2
        for p in self.P[(ida, idb)]:
            # Wouldn't that *actually* be G_mmi ?
            G_mmm = gaunt[la**2:(la + 1)**2,
                          lb**2:(lb + 1)**2,
                          l**2:(l + 1)**2].transpose((0, 2, 1))
            P, dPdr = p.get_value_and_derivative(r)

            GrlY_mi = npy.dot(rlY_lm[l], G_mmm)
            
            p_mi += P * GrlY_mi

            # If basis function and projector are located on the same atom,
            # the overlap is translation invariant
            if derivative:
                if r < 1e-14:
                    dPdR_cmi[:] = 0.
                else:
                    Rhat = R / r
                    for c in range(3):
                        A_mi = P * npy.dot(drlYdR_lmc[l][:, c], G_mmm)
                        B_mi = dPdr * GrlY_mi * Rhat[c]
                        dPdR_cmi[c, :, :] += A_mi + B_mi

            l += 2
        return p_mi, dPdR_cmi
        
    def st_overlap3(self, ida, idb, la, lb, r, R, rlY_lm, drlYdR_lmc):
        """ Returns the overlap and kinetic energy matrices. """
        
        s_mm = npy.zeros((2 * lb + 1, 2 * la + 1))
        t_mm = npy.zeros((2 * lb + 1, 2 * la + 1))
        dSdR_cmm = None
        dTdR_cmm = None
        derivative = bool(drlYdR_lmc)
        if derivative:
            dSdR_cmm = npy.zeros((3, 2 * lb + 1, 2 * la + 1))
            dTdR_cmm = npy.zeros((3, 2 * lb + 1, 2 * la + 1))
        ssplines = self.S[(ida, idb)]
        tsplines = self.T[(ida, idb)]
        l = (la + lb) % 2
        for s, t in zip(ssplines, tsplines):
            G_mmm = gaunt[lb**2:(lb + 1)**2,
                          la**2:(la + 1)**2,
                          l**2:(l + 1)**2].transpose((0, 2, 1))
            
            S, dSdr = s.get_value_and_derivative(r)
            T, dTdr = t.get_value_and_derivative(r)

            GrlY_mm = npy.dot(rlY_lm[l], G_mmm)

            s_mm += S * GrlY_mm
            t_mm += T * GrlY_mm

            # If basis functions are located on the same atom,
            # the overlap is translation invariant
            if derivative:
                if r < 1e-14:
                    dSdR_cmm[:] = 0.
                    dTdR_cmm[:] = 0.
                else:
                    rl = r**l
                    Rhat = R / r
                    for c in range(3):
                        # Product differentiation - two terms
                        GdrlYdR_mm = npy.dot(drlYdR_lmc[l][:, c], G_mmm)
                        rlYRhat_mm = GrlY_mm * Rhat[c]

                        S1_mm = S * GdrlYdR_mm
                        S2_mm = dSdr * rlYRhat_mm
                        dSdR_cmm[c, :, :] += S1_mm + S2_mm

                        T1_mm = T * GdrlYdR_mm
                        T2_mm = dTdr * rlYRhat_mm
                        dTdR_cmm[c, :, :] += T1_mm + T2_mm
            
            l += 2
        return s_mm, t_mm, dSdR_cmm, dTdR_cmm


    #################  Old overlaps  ####################
    def st_overlap0(self, id1, id2, l1, l2, m1, m2, R):
        """ Returns the overlap and kinetic energy matrices. """
        
        l = (l1 + l2) % 2
        S = 0.0
        T = 0.0
        r = sqrt(npy.dot(R, R))
        L1 = l1**2 + m1
        L2 = l2**2 + m2
        ssplines = self.S[(id1, id2)]
        tsplines = self.T[(id1, id2)]
        for s, t in zip(ssplines, tsplines):
            sr = s(r)
            tr = t(r)
            for m in range(2 * l + 1):
                L = l**2 + m
                c = Y(L, R[0], R[1], R[2]) * gaunt[L1, L2, L]
                S += sr * c
                T += tr * c
            l += 2
        return S, T

    def st_overlap(self, id1, id2, l1, l2, m1, m2, R):
        """ Returns the overlap and kinetic energy matrices. """
        
        l = (l1 + l2) % 2
        S = 0.0
        T = 0.0
        r = sqrt(npy.dot(R, R))
        L1 = l1**2 + m1
        L2 = l2**2 + m2
        ssplines = self.S[(id1, id2)]
        tsplines = self.T[(id1, id2)]
        for s, t in zip(ssplines, tsplines):
            Y_m = npy.empty(2 * l + 1)
            Yl(l, R, Y_m)
            L = l**2
            S += s(r) * npy.dot(Y_m, gaunt[L1, L2, L:L + 2 * l + 1])
            T += t(r) * npy.dot(Y_m, gaunt[L1, L2, L:L + 2 * l + 1])
            l += 2
        return S, T
    
    def p_overlap0(self, id1, id2, l1, l2, m1, m2, R):
        """ Returns the overlap between basis functions and projector
        functions. """

        l = (l1 + l2) % 2
        P = 0.0
        r = sqrt(npy.dot(R, R))
        L1 = l1**2 + m1
        L2 = l2**2 + m2
        for p in self.P[(id1, id2)]:
            pr = p(r)
            for m in range(2 * l + 1):
                L = l**2 + m
                P += pr * Y(L, R[0], R[1], R[2]) * gaunt[L1, L2, L]
            l += 2
        return P

    def p_overlap(self, id1, id2, l1, l2, m1, m2, R):
        """ Returns the overlap between basis functions and projector
        functions. """

        l = (l1 + l2) % 2
        P = 0.0
        r = sqrt(npy.dot(R, R))
        L1 = l1**2 + m1
        L2 = l2**2 + m2
        for p in self.P[(id2, id1)]:
            Y_m = npy.empty(2 * l + 1)
            Yl(l, R, Y_m)
            L = l**2
            P += p(r) * npy.dot(Y_m, gaunt[L1, L2, L:L + 2 * l + 1])
            l += 2
        return P
 
    # Testing
    def test(self, h, id1, id2, m1=0, m2=0, out=False):
        from gpaw.domain import Domain
        from gpaw.grid_descriptor import GridDescriptor
        from gpaw.localized_functions import create_localized_functions, \
             LocFuncBroadcaster
        from gpaw.operators import Laplace
        
        phit1 = self.setups[0].phit_j[id1[1]]
        phit2 = self.setups[1].phit_j[id2[1]]
        l1 = phit1.get_angular_momentum_number()
        l2 = phit2.get_angular_momentum_number()
        rc = self.rcmax + 1.0
        n = int(2 * rc / h / 4) * 4 +8
        domain = Domain((4 * rc, 4 * rc, 4 * rc), (False, False, False))
        gd = GridDescriptor(domain, (2 * n, 2 * n, 2 * n))
        f = create_localized_functions([phit1], gd, (0.25, 0.25, 0.25))
        a = gd.zeros()
        c1 = npy.zeros(2 * l1 + 1)
        c1[m1] = 1
        c2 = npy.zeros(2 * l2 + 1)
        c2[m2] = 1
        f.add(a, c1)
        kina = gd.zeros() 
        kin = Laplace(gd, -0.5) 
        kin.apply(a, kina) 
        for i in range(21):
            x = 0.25 + 0.5 * i / 20
            y = 0.25 + 0.4 * 1 / 20
            z = 0.25 + 0.3 * i / 20
            g = create_localized_functions([phit2], gd, (x, y, z))
            b = gd.zeros()
            g.add(b, c2)
            s = gd.integrate(a * b)
            t = gd.integrate(kina * b)
            d = [-(x - 0.25) * 4 * rc, -(y - 0.25) * 4 * rc,
                 -(z - 0.25) * 4 * rc]
            S, T = self.st_overlap(id1, id2, l1, l2, m1, m2, d)
            r = sqrt(npy.dot(d, d))
            if out:
                print 'S:',r, s, S
                print 'T:',r, t, T


    def test_fixed_distance(self, h, id1, id2, r, m1=0, m2=0, out=False):
        from gpaw.domain import Domain
        from gpaw.grid_descriptor import GridDescriptor
        from gpaw.localized_functions import create_localized_functions, \
             LocFuncBroadcaster
        from gpaw.operators import Laplace
        
        phit1 = self.setups[0].phit_j[id1[1]]
        phit2 = self.setups[1].phit_j[id2[1]]
        l1 = phit1.get_angular_momentum_number()
        l2 = phit2.get_angular_momentum_number()
        rc = self.rcmax + 1.0
        n = int(2 * rc / h / 4) * 4 +8
        domain = Domain((4 * rc, 4 * rc, 4 * rc), (False, False, False))
        gd = GridDescriptor(domain, (2 * n, 2 * n, 2 * n))
        f = create_localized_functions([phit1], gd, (0.25, 0.25, 0.25))
        a = gd.zeros()
        c1 = npy.zeros(2 * l1 + 1)
        c1[m1] = 1
        c2 = npy.zeros(2 * l2 + 1)
        c2[m2] = 1
        f.add(a, c1)
        kina = gd.zeros() 
        kin = Laplace(gd, -0.5) 
        kin.apply(a, kina) 
        rx = r[0] + 0.25
        ry = r[1] + 0.25
        rz = r[2] + 0.25
        d = [-(rx - 0.25) * 4 * rc, -(ry - 0.25) * 4 * rc,
             -(rz - 0.25) * 4 * rc]
        g = create_localized_functions([phit2], gd, [rx, ry, rz])
        b = gd.zeros()
        g.add(b, c2)
        s = gd.integrate(a * b)
        t = gd.integrate(kina * b)
        S, T = self.st_overlap(id1, id2, l1, l2, m1, m2, d)
        r = sqrt(npy.dot(d, d))
        if out:
            print 'S:',r, s, S
            print 'T:',r, t, T
        return s, S, t, T
