from math import sqrt, pi
import Numeric as num
from FFT import inverse_real_fft, real_fft, inverse_fft
from gpaw.spline import Spline
from gpaw.spherical_harmonics import Y
from gpaw.gaunt import gaunt
from gpaw.utilities import fac

C = [num.array([-1.0j]),
     num.array([-1.0, -1.0j]),
     num.array([1.0j, -3.0, -3.0j])]

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
    g = num.zeros(m, num.Float)
    for n in range(l + 1):
        g += (dr * 2 * m * k**(l - n) *
              inverse_fft(C[l][n] * f * r**(1 + l - n), 2 * m)[:m].real)
    return g

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
        self.ng = 2**9
        phit_g = num.zeros(self.ng, num.Float) 
        phit_jq = {}
        self.dr = self.rcmax / self.ng
        self.r_g = num.arange(self.ng) * self.dr
        self.P = 4 * 2**9
        self.dk = 2 * pi / self.P / self.dr
        self.k = num.arange(self.P // 2) * self.dk
        phit_q = num.zeros(self.P // 2, num.Float)
        for setup in setups:
            for j, phit in enumerate(setup.phit_j):
                l = phit.get_angular_momentum_number()
                id = (setup.symbol, j)
                phit_g[0:self.ng] = [phit(r) for r in self.r_g[0:self.ng]]
                phit_q[:] = 0.0
                a_q = fbt(l, phit_g, self.r_g, self.k)
                phit_q += a_q
                phit_jq[id] = (l, phit_q.copy())
        self.splines = {}
        for id1, (l1, phit1_q) in phit_jq.items():
            for id2, (l2, phit2_q) in phit_jq.items():
                st = self.calculate_spline(phit1_q, phit2_q, l1, l2)
                self.splines[(id1, id2)] = st
        self.setups = setups

    def calculate_spline(self, phit1, phit2, l1, l2):
        S_g = num.zeros(2 * self.ng, num.Float)
        self.lmax = l1 + l2
        Ssplines = []
        R = num.arange(self.P // 2) * self.dr
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
                a_g[0] = 8 * num.sum(a_q * k1**(-l1 - l2)) * self.dk
            a_g *= (-1)**((l1 - l2 - l) / 2)
            S_g += a_g
            s = Spline(l, self.P // self.ng / 2 * self.rcmax, S_g)
            Ssplines.append(s)
        return Ssplines

    def overlap(self, id1, id2, l1, l2, m1, m2, R):
        l = (l1 + l2) % 2
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

    # Testing
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
