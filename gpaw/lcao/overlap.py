from math import sqrt, pi
import Numeric as num
from FFT import inverse_real_fft, real_fft
from gpaw.spline import Spline

class TwoCenterIntegrals:
    def __init__(self, setups):
        self.rcmax = 0.0
        for setup in setups:
            for phit in setup.phit_j:
                l = phit.get_angular_momentum_number()
                rc = phit.get_cutoff()
                print rc
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
        for setup in setups:
            for j, phit in enumerate(setup.phit_j):
                id = (setup.symbol, j)
                phit_g[0:ng] = [phit(r) for r in r_g]
                phit_q = (-sqrt(2 / pi) *
                          real_fft(phit_g * self.r_gp).imag * drp)
                phit_q[1:] /= self.k_qp[1:]
                phit_q[0] = (sqrt(2 / pi) *
                             num.dot(self.r_gp**2, phit_g) * drp)
                phit_jq[id] = phit_q

        self.splines = {}
        for id1, phit1_q in phit_jq.items():
            for id2, phit2_q in phit_jq.items():
                st = self.calculate_spline(phit1_q, phit2_q)
                self.splines[(id1, id2)] = st

    def calculate_spline(self, phit1, phit2):
        # Overlap spline:
        S_g = -inverse_real_fft(1j * self.k_qp * phit1 * phit2) \
             * self.ngp * self.dk * self.a**3 / 2       
        S_g[1:] /= self.r_gp[1:]
        S_g[0] = (num.dot(self.k_qp**2, phit1 *
                          phit2) * self.dk * self.a**3 / 2)
        Y = 1 / sqrt(4 * pi)
        G = Y
        S_g *= G * 4 * pi * Y
        s = Spline(0, self.a * self.rcmax, S_g[:self.nqp])

        # Kinetic Energy spline:
        T_g = -inverse_real_fft(1j * self.k_qp**3 * phit1 * phit2) \
             * self.ngp * self.dk / self.a**3       
        T_g[1:] /= self.r_gp[1:]
        T_g[0] =  num.dot(self.k_qp**4, phit1 * phit2) * self.dk / self.a**3
        T_g *= G * 4 * pi * Y
        t = Spline(0, self.a * self.rcmax, T_g[:self.nqp])
        return s, t

    def test(self, h):
        from gpaw.domain import Domain
        from gpaw.grid_descriptor import GridDescriptor
        from gpaw.localized_functions import create_localized_functions, \
             LocFuncBroadcaster

        phit = self.setup.phit_j[0]
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
            g = create_localized_functions([phit], gd, (x, 0.5, 0.5))
            b = gd.zeros()
            g.add(b, c)
            s = gd.integrate(a * b)
            d = (x - 0.25) * 4 * rc
            print d, s, self.ss(d)

    def testb(self, h, setups):
        from gpaw.domain import Domain
        from gpaw.grid_descriptor import GridDescriptor
        from gpaw.localized_functions import create_localized_functions, \
             LocFuncBroadcaster

        print self.splines
        sspline = self.splines[(('H', 0), ('Li', 0))][0]
        phit = setups[0].phit_j[0]
        phitb = setups[1].phit_j[0]
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
            print d, s, sspline(d)

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
        a = gd.zeros()
        kina = gd.zeros()
        c = num.ones(1, num.Float)
        f.add(a, c)
        kin.apply(a, kina)
        for i in range(21):
            x = 0.25 + 0.5 * i / 20
            g = create_localized_functions([phit], gd, (x, 0.5, 0.5))
            b = gd.zeros()
            g.add(b, c)
            s = gd.integrate(kina * b)
            d = (x - 0.25) * 4 * rc
            print d, s, self.tt(d)
