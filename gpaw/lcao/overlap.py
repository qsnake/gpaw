from math import sqrt, pi
import Numeric as num
from FFT import inverse_real_fft, real_fft
from gpaw.spline import Spline
from gpaw.spherical_harmonics import Y
from gpaw.gaunt import gaunt
from gpaw.utilities import fac


c = [num.array([1j]), num.array([1j, -1])]
for l in range(2, 5):
    b = num.zeros(l + 1, num.Complex)
    b[0:l] = (2 * l + 1) * c[-1]
    b[2:] = -c[-2]
    c.append(b)

class TwoCenterIntegrals:
    """ Two-center integrals class.

    This class implements a Fourier-space calculation of two-center
    integrals. """

    def __init__(self, setups):
        self.rcmax = 0.0
        for setup in setups:
            for phit in setup.phit_j:
                l = phit.get_angular_momentum_number()
                rc = phit.get_cutoff()
                if rc > self.rcmax:
                    self.rcmax = rc
        print self.rcmax
        ng = 2**7
        nq = ng / 2 + 1
        self.a = 2
        self.ngp = self.a * ng 
        nq = ng / 2 + 1 
        self.nqp = self.ngp / 2 + 1
        phit_g = num.zeros(self.ngp, num.Float) 
        phit_jq = {}
        dr = 2 * self.rcmax / ng
        drp = 2 * self.rcmax / self.ngp
        r_g = num.arange(ng) * dr 
        self.r_gp = num.arange(self.ngp) * drp 
        self.dk = 2 * pi / self.rcmax
        k_q = num.arange(nq) * self.dk
        self.k_qp = num.arange(self.nqp) * self.dk
        phit_q = num.zeros(self.nqp, num.Complex)
        for setup in setups:
            for j, phit in enumerate(setup.phit_j):
                l = phit.get_angular_momentum_number()
                id = (setup.symbol, j)
                phit_g[0:ng] = [phit(r) for r in r_g]
                phit_q[:] = 0.0
                if 1:
                    for i in range(l+1):
                        phit_q += (-sqrt(2 / pi) * (-1j)**l *  self.k_qp**i *
                                   (c[l][i] * real_fft(self.r_gp**(1 + i) *
                                                       phit_g)).real * drp)
                        phit_q[1:] /= self.k_qp[1:]**(l + 1 - i)
                else:
                    phit_q = (-sqrt(2 / pi) *
                              real_fft(phit_g * self.r_gp).imag * drp)
                if l == 0:
                    phit_q[0] = -(sqrt(2 / pi) / 2 *
                                  2**l * fac[l] / fac[2 * l + 1] *
                                  num.dot(self.r_gp**(2 + l), phit_g) * drp)
                else:
                    phit_q[0] = 0.0
                phit_jq[id] = (l, phit_q.copy())
                #print l, phit_q[:4]

        self.splines = {}
        for id1, (l1, phit1_q) in phit_jq.items():
            for id2, (l2, phit2_q) in phit_jq.items():
                st = self.calculate_spline(phit1_q, phit2_q, l1, l2)
                self.splines[(id1, id2)] = st
        self.setups = setups
            
    def c2f(a):
        b = num.zeros(a.shape[0])
        for i in range(a.shape[0]):
            b[i] = float[a[i]]
        return b
            


    def calculate_spline(self, phit1, phit2, l1, l2):
        # Overlap spline:
        #S_g = -inverse_real_fft(1j * self.k_qp * phit1 * phit2) \
        #     * self.ngp * self.dk * self.a**3 / 2       
        #S_g[1:] /= self.r_gp[1:]
        S_g = num.zeros(self.ngp, num.Float)
        r_g = num.arange(self.ngp) * self.a * self.rcmax / self.ngp
        #a_q = num.zeros(self.ngp, num.Complex)
        self.lmax = l1 + l2
        Ssplines = []
        for l in range(self.lmax % 2, self.lmax + 1, 2):
            S_g[:] = 0.0
            for i in range(l + 1):
                a_q = (num.conjugate(phit1) * phit2 * self.ngp *
                       1j**(l2 - l1) * self.dk * self.a**3 / 2)
                a_q[1:] /= self.k_qp[1:]**(l - 1 -i)
                a_q[0] = 0
                a_g = -inverse_real_fft(c[l][i] * a_q).real
                a_g[1:] /= self.r_gp[1:]**(l + 1 - i)
                if l == 0:
                    a_g[0] = (num.dot(self.k_qp**2, abs(phit1) * abs(phit2)) *
                              self.dk * self.a**3 / 2)
                else:
                    a_g[0] = 0.0
                a_g *= 4 * pi
                S_g += a_g
            #print S_g[:4], l
            S_g[1:] /= r_g[1:]**l 
            S_g[0] = (4 * pi * 2**l * fac[l] / fac[2 * l + 1] *
                      num.dot(self.k_qp**(2 + l),
                              abs(phit1) * abs(phit2)) *
                      self.dk * self.a**3 / 2)
            print S_g[:5], l
            s = Spline(l, self.a * self.rcmax, S_g[:self.nqp])
            Ssplines.append(s)
        return Ssplines

    def overlap(self, id1, id2, l1, l2, m1, m2, R):
        l = self.lmax % 2
        S = 0.0
        r = sqrt(num.dot(R, R))
        L1 = l1**2 + m1
        L2 = l2**2 + m2
        for s in self.splines[(id1, id2)]:
            for m in range(2 * l + 1):
                L = l**2 + m
                S += s(r) * Y(L, R[0], R[1], R[2]) * gaunt[L1, L2, L]
            l += 2
        return S    
    
        '''
        # Kinetic Energy spline:
        T_g = -inverse_real_fft(1j * self.k_qp**3 * phit1 * phit2) \
             * self.ngp * self.dk / self.a**3       
        T_g[1:] /= self.r_gp[1:]
        T_g[0] =  num.dot(self.k_qp**4, phit1 * phit2) * self.dk / self.a**3
        T_g *= G * 4 * pi * Y
        t = Spline(0, self.a * self.rcmax, T_g[:self.nqp])
        print s(0.2)
        return s, t '''

    def testb(self, h):
        from gpaw.domain import Domain
        from gpaw.grid_descriptor import GridDescriptor
        from gpaw.localized_functions import create_localized_functions, \
             LocFuncBroadcaster

        print self.splines
        sspline = self.splines[(('H', 0), ('Li', 0))]
        phit = self.setups[0].phit_j[0]
        phitb = self.setups[1].phit_j[0]
        rc = phit.get_cutoff() + 1.0
        n = int(2 * rc / h / 4) * 4 + 8
        domain = Domain((4 * rc, 2 * rc, 2 * rc), (False, False, False))
        gd = GridDescriptor(domain, (2 * n, n, n))
        f = create_localized_functions([phit], gd, (0.25, 0.5, 0.5))
        a = gd.zeros()
        c = num.ones(1, num.Float)
        f.add(a, c)
        for i in range(21):
            x = 0.25 + 0.5 * i / 20
            g = create_localized_functions([phitb], gd, (x, 0.5, 0.5))
            b = gd.zeros()
            g.add(b, c)
            s = gd.integrate(a * b)
            d = (x - 0.25) * 4 * rc
            print d, s, sspline[0](d)

    def test2(self, h):
        from gpaw.domain import Domain
        from gpaw.grid_descriptor import GridDescriptor
        from gpaw.localized_functions import create_localized_functions, \
             LocFuncBroadcaster
        from gpaw.operators import Laplace
        
        phit = self.setup.phit_j[0]
        rc = phit.get_cutoff() + 1.0
        n = int(2 * rc / h / 4) * 4 + 8
        domain = Domain((4 * rc, 2 * rc, 2 * rc), (False, False, False))
        gd = GridDescriptor(domain, (2 * n, n, n))
        kin = Laplace(gd, -0.5)
        f = create_localized_functions([phit], gd, (0.25, 0.5, 0.5))
        a = gd.new_array()
        kina = gd.new_array()
        c = num.ones(1, num.Float)
        f.add(a, c)
        kin.apply(a, kina)
        for i in range(21):
            x = 0.25 + 0.5 * i / 20
            g = create_localized_functions([phit], gd, (x, 0.5, 0.5))
            b = gd.new_array()
            g.add(b, c)
            s = gd.integrate(kina * b)
            d = (x - 0.25) * 4 * rc
            print d, s, self.tt(d)
