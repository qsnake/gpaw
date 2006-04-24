# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import sys
from math import pi, sqrt

import Numeric as num
num.inner = num.innerproduct # XXX numpy!
from LinearAlgebra import solve_linear_equations, inverse
from ASE.ChemicalElements.name import names

from gridpaw.atom.configurations import configurations
from gridpaw.version import version
from gridpaw.atom.all_electron import AllElectron, shoot
from gridpaw.polynomium import a_i, c_l
from gridpaw.utilities.lapack import diagonalize
from gridpaw.exx import constructX
from gridpaw.exx import atomic_exact_exchange as aExx


parameters = {
    #     (core,      cutoff,         extra projectors),  
    'H' : ('',        0.9),
    'He': ('',        1.5),
    'Li': ('[He]',    2.2),
    'Be': ('[He]',    1.5),
    'C' : ('[He]',    1.2),
    'N' : ('[He]',    1.2),
    'O' : ('[He]',    1.2,            {0: [1.0], 1: [1.0], 2: [1.0]}),
    'F' : ('[He]',    1.2,            {0: [1.0], 1: [1.0], 2: [1.0]}),
    'Ne': ('[He]',    1.8),    
    'Na': ('[Ne]',    2.3),
    'Mg': ('[Ne]',    2.0),
    'Al': ('[Ne]',    2.0),
    'Si': ('[Ne]',    1.7),
    'P' : ('[Ne]',    2.0),
    'S' : ('[Ne]',    1.87),
    'Cl': ('[Ne]',    1.5),
    'V' : ('[Ar]',   [2.4, 2.4, 2.2], {0: [0.8], 1: [-0.2], 2: [0.8]}),
    'Cr': ('[Ar]',   [2.4, 2.4, 2.2], {0: [0.8], 1: [-0.2], 2: [0.8]}),
    'Fe': ('[Ar]',    2.1),
    'Ni': ('[Ar]',    2.3),
    'Cu': ('[Ar]',   [2.3, 2.3, 2.1]),
    'Ga': ('[Ar]3d',  2.0),
    'As': ('[Ar]',    2.0),
    'Zr': ('[Ar]3d',  2.0),
    'Mo': ('[Kr]',   [2.8, 2.8, 2.3]),
    'Ru': ('[Kr]',   [2.5, 2.4, 2.5], {0: [0.8], 1: [0.0], 2: [0.8]}),
    'Pd': ('[Kr]',   [2.3, 2.2, 1.9], {0: [-0.3], 1: [-0.3], 2: [0.8]}),
    'Pt': ('[Xe]4f',  2.5),
    'Au': ('[Xe]4f',  2.5)
    }


class Generator(AllElectron):
    def __init__(self, symbol, xcname='LDA', scalarrel=False):
        AllElectron.__init__(self, symbol, xcname, scalarrel)


    def run(self, core, rcut, extra, gamma, scale_radius,
            logderiv=True, vt0=None, exx=False):

        self.core = core
        if type(rcut) is float:
            rcut_l = [rcut]
        else:
            rcut_l = rcut
            rcut = max(rcut_l)
        self.rcut = rcut
        self.rcut_l = rcut_l

        # Do all-electron calculation:
        AllElectron.run(self)

        print
        print 'Generating PAW setup'
        if core != '':
            print 'Frozen core:', core
            
        # So far - no ghost-states:
        self.ghost = False

        N = self.N
        r = self.r
        dr = self.dr
        d2gdr2 = self.d2gdr2
        beta = self.beta

        dv = r**2 * dr
        
        Z = self.Z

        n_j = self.n_j
        l_j = self.l_j
        f_j = self.f_j
        e_j = self.e_j
        nj = len(n_j)

        # Highest occupied atomic orbital:
        self.emax = max(e_j)
        
        # Parse core string:
        j = 0
        if core.startswith('['):
            a, core = core.split(']')
            core_symbol = a[1:]
            j = len(configurations[core_symbol][1])

        while core != '':
            assert n_j[j] == int(core[0])
            assert l_j[j] == 'spdf'.find(core[1])
            assert f_j[j] == 2 * (2 * l_j[j] + 1)
            j += 1
            core = core[2:]
        njcore = j
        self.njcore = njcore
        
        self.Nv = sum(f_j[njcore:])

        # Calculate the kinetic energy of the core states:
        Ekincore = 0.0
        for f, e, u in zip(f_j[:njcore], e_j[:njcore], self.u_j[:njcore]):
            u = num.where(abs(u) < 1e-160, 0, u)  # XXX Numeric!
            Ekincore += f * (e - num.sum((u**2 * self.vr * dr)[1:] / r[1:]))

        # Calculate core density:
        if njcore == 0:
            nc = num.zeros(N, num.Float)
        else:
            uc_j = self.u_j[:njcore]
            uc_j = num.where(abs(uc_j) < 1e-160, 0, uc_j)  # XXX Numeric!
            nc = num.dot(f_j[:njcore], uc_j**2) / (4 * pi)
            nc[1:] /= r[1:]**2
            nc[0] = nc[1]

        lmax = max(l_j[njcore:])

        # Order valence states with respect to angular momentum
        # quantum number:
        self.n_ln = n_ln = []
        self.f_ln = f_ln = []
        self.e_ln = e_ln = []
        for l in range(lmax + 1):
            n_n = []
            f_n = []
            e_n = []
            for j in range(njcore, nj):
                if l_j[j] == l:
                    n_n.append(n_j[j])
                    f_n.append(f_j[j])
                    e_n.append(e_j[j])
                    assert f_j[j] > 0.0
            n_ln.append(n_n)
            f_ln.append(f_n)
            e_ln.append(e_n)

        # Add extra projectors:
        if extra is not None:
            if len(extra) == 0:
                lmaxextra = 0
            else:
                lmaxextra = max(extra.keys())
            if lmaxextra > lmax:
                for l in range(lmax, lmaxextra):
                    n_ln.append([])
                    f_ln.append([])
                    e_ln.append([])
                lmax = lmaxextra
            for l in extra:
                for e in extra[l]:
                    if len(n_ln[l]) > 0:
                        n = n_ln[l][-1] + 1
                    else:
                        n = l + 1
                    n_ln[l].append(n)
                    f_ln[l].append(0.0)
                    e_ln[l].append(e)
        else:
            # Automatic:
            if [len(n_n) for n_n in n_ln] == [1, 0, 1]:
                # We have s- and d-channels, but no p-channel.  Add it:
                n = n_ln[0][0]
                e = e_ln[0][0]
                n_ln[1] = [n]
                f_ln[1] = [0.0]
                e_ln[1] = [e]
            elif [len(n_n) for n_n in n_ln] == [0, 0, 1]:
                # We have a d-channel, but no s- and p-channel.  Add them:
                n = n_ln[2][0]
                e = e_ln[2][0]
                n_ln[0] = [n]
                f_ln[0] = [0.0]
                e_ln[0] = [e]
                n_ln[1] = [n]
                f_ln[1] = [0.0]
                e_ln[1] = [e]

            # Make sure we have two projectors for each occupied channel:
            for l in range(lmax + 1):
                if len(n_ln[l]) < 2:
                    # Only one - add one more:
                    n = 1 + n_ln[l][0]
                    e = 1.0 + e_ln[l][0]

                    n_ln[l].append(n)
                    f_ln[l].append(0.0)
                    e_ln[l].append(e)

            if lmax < 2:
                # Add extra projector for l = lmax + 1:
                n = lmax + 2
                n_ln.append([n])
                f_ln.append([0.0])
                e_ln.append([0.0])
                lmax += 1

        self.lmax = lmax

        rcut_l.extend([rcut] * (lmax + 1 - len(rcut_l)))

        print 'Cutoffs:',
        for rc, s in zip(rcut_l, 'spd'):
            print 'rc(%s)=%.3f' % (s, rc),
        print
        print 'Kinetic energy of the core states: %.6f' % Ekincore

        # Allocate arrays:
        self.u_ln = u_ln = []  # phi * r
        self.s_ln = s_ln = []  # phi-tilde * r
        self.q_ln = q_ln = []  # p-tilde * r
        G_ln = []
        for l in range(lmax + 1):
            nn = len(n_ln[l])
            u_ln.append(num.zeros((nn, N), num.Float))
            s_ln.append(num.zeros((nn, N), num.Float))
            q_ln.append(num.zeros((nn, N), num.Float))
            G_ln.append(num.zeros((nn, N), num.Float))
            
        # Fill in all-electron wave functions:
        for l in range(lmax + 1):
            # Collect all-electron wave functions:
            u_n = [self.u_j[j] for j in range(njcore, nj) if l_j[j] == l]
            for n, u in enumerate(u_n):
                assert f_ln[l][n] > 0.0
                u_ln[l][n] = u

        # Grid-index corresponding to rcut:
        gcut = int(rcut * N / (rcut + beta))

        rcut2 = scale_radius * rcut
        gcut2 = int(rcut2 * N / (rcut2 + beta))

        # Outward integration of unbound states stops at 3 * rcut:
        gmax = int(3 * rcut * N / (3 * rcut + beta))
        assert gmax > gcut2
        
        # Calculate unbound extra states:
        c2 = -(r / dr)**2
        c10 = -d2gdr2 * r**2
        for l, (f_n, e_n, u_n) in enumerate(zip(f_ln, e_ln, u_ln)):
            for f, e, u in zip(f_n, e_n, u_n):
                if f == 0.0:
                    u[:] = 0.0
                    shoot(u, l, self.vr, e, self.r2dvdr, r, dr, c10, c2,
                          self.scalarrel, gmax=gmax)
                    u *= 1.0 / u[gcut]

        Nc = Z - self.Nv
        Nctail = 4 * pi * num.dot(nc[gcut:], dv[gcut:])
        print 'Core states: %d (r > %.3f: %.6f)' % (Nc, rcut, Nctail)
        assert Nctail < 1.1
        print 'Valence states: %d' % self.Nv

        # Calculate pseudo core density:
        nct = nc.copy()
        A = num.ones((4, 4), num.Float)
        A[0] = 1.0
        A[1] = r[gcut - 2:gcut + 2]**2
        A[2] = A[1]**2
        A[3] = A[1] * A[2]
        a = nc[gcut - 2:gcut + 2]
        a = solve_linear_equations(num.transpose(A), a)
        r2 = r[:gcut]**2
        nct[:gcut] = a[0] + r2 * (a[1] + r2 * (a[2] + r2 * a[3]))
        print 'Pseudo-core charge: %.6f' % (4 * pi * num.dot(nct, dv))
        
        # Calculate pseudo valence density:
        nt = self.n.copy()
        A = num.ones((4, 4), num.Float)
        A[0] = 1.0
        A[1] = r[gcut - 2:gcut + 2]**2
        A[2] = A[1]**2
        A[3] = A[1] * A[2]
        a = self.n[gcut - 2:gcut + 2]
        a = solve_linear_equations(num.transpose(A), a)
        r2 = r[:gcut]**2
        nt[:gcut] = a[0] + r2 * (a[1] + r2 * (a[2] + r2 * a[3]))
        nt += nct

        # Calculate the shape function:
        x = r / rcut
        gaussian = num.zeros(N, num.Float)
        self.gamma = gamma
        gaussian[:gcut2] = num.exp(-self.gamma * x[:gcut2]**2)
        gt = 4 * (self.gamma / rcut**2)**1.5 / sqrt(pi) * gaussian
        norm = num.dot(gt, dv)
        print norm, norm-1
        assert abs(norm - 1) < 1e-2
        gt /= norm
        
        # Construct projector functions:
        
        for l, G_n in enumerate(G_ln):
            x = r / rcut_l[l]
            Gl = num.zeros(N, num.Float)
            Gl[:gcut2] = num.exp(-self.gamma * x[:gcut2]**2)
            for n, G in enumerate(G_n):
                if n == 0:
                    G[:] = Gl * r**(l + 1 + 2 * n)
                else:
                    G[:] = (Gl * r**(l + 1) -
                            2.0 / (3 + 2 * l) * Gl * r**(l + 1 + 2 * n))

        print 'Pseudo-electron charge',
        Nt0 = 0.0
        while True:

            sys.stdout.write('.')
            sys.stdout.flush()
            # Calculate neutral smooth charge density:
            Nt = num.dot(nt, dv)
            rhot = nt - Nt * gt

            vHt = self.calculate_hartree_potential(rhot)
            vHt[1:] /= r[1:]
            vHt[0] = vHt[1]

            vXCt = num.zeros(N, num.Float)
            Exct = self.xc.get_energy_and_potential(nt, vXCt)
            vt = vHt + vXCt

            # Construct localized potential:
            A = num.ones((2, 2), num.Float)
            A[0] = 1.0
            A[1] = r[1:7:3]**2
            a = vt[1:7:3]
            a = solve_linear_equations(num.transpose(A), a)
            self.vbar0 = a[1] * rcut**2 / self.gamma
##            self.vbar0 = 0.0####################### XXXXXXXXX
            vbar = self.vbar0 * gaussian
            
            vt += vbar

            self.dH_lnn = dH_lnn = []
            self.dO_lnn = dO_lnn = []
            self.dK_lnn = dK_lnn = []
            self.ptcoef_ln = []
            for l, (e_n, u_n, s_n, q_n, G_n) in \
                    enumerate(zip(e_ln, u_ln, s_ln, q_ln, G_ln)):
                for n, e in enumerate(e_n):
                    a = self.integrate(l, vt, e, gcut2)
                    b = self.integrate(l, vt, e, gcut2, G_n[n])

                    C_kk = num.zeros((2, 2), num.Float)
                    d_k = num.zeros(2, num.Float)

                    C_kk[0, 0] = a[gcut2]
                    C_kk[0, 1] = b[gcut2]
                    d_k[0] = u_n[n][gcut2]

                    C_kk[1, 0] = a[gcut2 - 1]
                    C_kk[1, 1] = b[gcut2 - 1]
                    d_k[1] = u_n[n][gcut2 - 1]

                    d_k = solve_linear_equations(C_kk, d_k)
                    alpha = d_k[0]
                    eta = d_k[1]
                    
                    s_n[n] = alpha * a + eta * b
                    s_n[n, gcut2:] = u_n[n, gcut2:]

                    q_n[n] = G_n[n] * eta

                dO_nn = num.inner(u_n, u_n * dr) - num.inner(s_n, s_n * dr)
                e_nn = num.zeros((n + 1, n + 1), num.Float)
                e_nn.flat[::n + 2] = e_n
                A_nn = num.innerproduct(q_n, s_n * dr)
                dH_nn = num.dot(e_nn, dO_nn) - A_nn

                A_nn = num.innerproduct(G_n, s_n * dr)
                # Orthonormalize projector functions:
                A_nn = inverse(A_nn)
                q_n[:] = num.dot(A_nn, G_n)
                self.ptcoef_ln.append(A_nn)
                
                ku_n = [self.kin(l, u, e) for u, e in zip(u_n, e_n)]  
                ks_n = [self.kin(l, s) for s in s_n]
                
                # XXX Should only integrate to gcut2!
                dK_nn = 0.5 * (num.inner(u_n, ku_n * dr) -
                               num.inner(s_n, ks_n * dr))
                dK_nn += num.transpose(dK_nn).copy()

                dO_lnn.append(dO_nn)
                dH_lnn.append(dH_nn)
                dK_lnn.append(dK_nn)

            if abs(Nt - Nt0) < 1e-7:
                break
            Nt0 = Nt

##            print Nt, self.vbar0
            
            # ... and the soft valence density:
            nt0 = nt
            nt = num.zeros(N, num.Float)
            for f_n, s_n in zip(f_ln, s_ln):
                nt += num.dot(f_n, s_n**2) / (4 * pi)
            nt[1:] /= r[1:]**2
            nt[0] = nt[1]
            nt += nct
            nt += 0.5 * (nt0 - nt)

        print '%.6f' % (4 * pi * Nt)
        
        self.vt = vt
        
        print 'state    eigenvalue         norm'
        print '--------------------------------'
        for l, (n_n, f_n, e_n) in enumerate(zip(n_ln, f_ln, e_ln)):
            for n in range(len(e_n)):
                print '%d%s^%-2d: %12.6f' % (n_n[n], 'spd'[l], f_n[n], e_n[n]),
                if f_n[n] > 0.0:
                    print '%12.6f' % num.dot(s_ln[l][n]**2, dr)
                else:
                    print
        print '--------------------------------'

        if logderiv:
            # Calculate logarithmic derivatives:
            gld = gcut + 10
            assert gld < gmax
            print 'Calculating logarithmic derivatives at r=%.3f' % r[gld]
            print '(skip with [Ctrl-C])'

            try:
                u = num.zeros(N, num.Float)
                for l in range(3):
                    if l <= lmax:
                        dO_nn = dO_lnn[l]
                        dH_nn = dH_lnn[l]
                        q_n = q_ln[l]

                    fae = open(self.symbol + '.ae.ld.' + 'spd'[l], 'w')
                    fps = open(self.symbol + '.ps.ld.' + 'spd'[l], 'w')

                    ni = 300
                    e1 = -5.0
                    e2 = 1.0
                    e = e1
                    for i in range(ni):
                        # All-electron logarithmic derivative:
                        u[:] = 0.0
                        shoot(u, l, self.vr, e, self.r2dvdr, r, dr, c10, c2,
                              self.scalarrel, gmax=gld)
                        dudr = 0.5 * (u[gld + 1] - u[gld - 1]) / dr[gld]
                        print >> fae, e,  dudr / u[gld] - 1.0 / r[gld]

                        # PAW logarithmic derivative:
                        s = self.integrate(l, vt, e, gld)
                        if l <= lmax:
                            A_nn = dH_nn - e * dO_nn
                            s_n = [self.integrate(l, vt, e, gld, q) for q in q_n]
                            B_nn = num.inner(q_n, s_n * dr)
                            a_n = num.dot(q_n, s * dr)

                            B_nn = num.dot(A_nn, B_nn)
                            B_nn.flat[::len(a_n) + 1] += 1.0
                            c_n = solve_linear_equations(B_nn,
                                                         num.dot(A_nn, a_n))
                            s -= num.dot(c_n, s_n)

                        dsdr = 0.5 * (s[gld + 1] - s[gld - 1]) / dr[gld]
                        print >> fps, e, dsdr / s[gld] - 1.0 / r[gld]

                        e += (e2 - e1) / ni
            except KeyboardInterrupt:
                pass
            
        self.write(nc,'nc')
        self.write(nt, 'nt')
        self.write(nct, 'nct')
        self.write(vbar, 'vbar')
        self.write(vt, 'vt')

        for l, (n_n, f_n, u_n, s_n, q_n) in enumerate(zip(n_ln, f_ln,
                                                          u_ln, s_ln, q_ln)):
            for n, f, u, s, q in zip(n_n, f_n, u_n, s_n, q_n):
                if f == 0.0:
                    self.write(u, 'ae', n=n, l=l)
                self.write(s, 'ps', n=n, l=l)
                self.write(q, 'proj', n=n, l=l)

        for h in [0.05]:
            self.diagonalize(h)

        if exx:
            X_p = constructX(self)
            ExxC = aExx(self,'core-core')
        else:
            X_p = None
            ExxC = None
            
        self.write_xml(n_ln, f_ln, e_ln, u_ln, s_ln,
                      nc, nct, Ekincore, X_p, ExxC, scale_radius)

    def diagonalize(self, h):
        ng = 300
        print
        print 'Diagonalizing with gridspacing h=%.3f' % h
        R = h * num.arange(1, ng + 1)
        G = (self.N * R / (self.beta + R) + 0.5).astype(num.Int)
        G = num.clip(G, 1, self.N - 2)
        R1 = num.take(self.r, G - 1)
        R2 = num.take(self.r, G)
        R3 = num.take(self.r, G + 1)
        x1 = (R - R2) * (R - R3) / (R1 - R2) / (R1 - R3)
        x2 = (R - R1) * (R - R3) / (R2 - R1) / (R2 - R3)
        x3 = (R - R1) * (R - R2) / (R3 - R1) / (R3 - R2)
        def interpolate(f):
            f1 = num.take(f, G - 1)
            f2 = num.take(f, G)
            f3 = num.take(f, G + 1)
            return f1 * x1 + f2 * x2 + f3 * x3
        vt = interpolate(self.vt)
        for l in range(3):
            if l <= self.lmax:
                q_n = num.array([interpolate(q) for q in self.q_ln[l]])
                H = num.dot(num.transpose(q_n),
                           num.dot(self.dH_lnn[l], q_n)) * h
                S = num.dot(num.transpose(q_n),
                           num.dot(self.dO_lnn[l], q_n)) * h
            else:
                H = num.zeros((ng, ng), num.Float)
                S = num.zeros((ng, ng), num.Float)
            H.flat[::ng + 1] += vt + 1.0 / h**2 + l * (l + 1) / 2.0 / R**2
            H.flat[1::ng + 1] -= 0.5 / h**2
            H.flat[ng::ng + 1] -= 0.5 / h**2
            S.flat[::ng + 1] += 1.0
            e_n = num.zeros(ng, num.Float)
            diagonalize(H, e_n, S)
            if l <= self.lmax:
                f = self.f_ln[l][0]
                e = self.e_ln[l][0]
            else:
                f = 0.0
            e0 = e_n[0]
            if (f > 0 and abs(e - e0) > 0.01) or (f == 0 and e0 < self.emax):
                print 'GHOST-state in %s-channel at %.6f' % ('spd'[l], e0)
                self.ghost = True

    def integrate(self, l, vt, e, gld, q=None):
        r = self.r[1:]
        dr = self.dr[1:]
        s = num.zeros(self.N, num.Float)
        
        c0 = 0.5 * l * (l + 1) / r**2
        c1 = -0.5 * self.d2gdr2[1:]
        c2 = -0.5 * dr**-2
        
        fp = c2 + 0.5 * c1
        fm = c2 - 0.5 * c1
        f0 = c0 - 2 * c2

        f0 += vt[1:] - e
        if q is None:
            s[1] = r[1]**(l + 1)
            for g in range(gld):
                s[g + 2] = (-fm[g] * s[g] - f0[g] * s[g + 1]) / fp[g]
            return s

        s[1] = q[1] / (vt[0] - e)
        for g in range(gld):
            s[g + 2] = (q[g + 1] - fm[g] * s[g] - f0[g] * s[g + 1]) / fp[g]
        return s


    def write_xml(self, n_ln, f_ln, e_ln, u_ln, s_ln,
                 nc, nct, Ekincore, X_p, ExxC, scale_radius):
        xml = open(self.symbol + '.' + self.xcname, 'w')

        if self.ghost:
            raise SystemExit

        n_j = []
        l_j = []
        f_j = []
        e_j = []
        u_j = []
        s_j = []
        ptcoef_j = []
        j_ln = [[0 for f in f_n] for f_n in f_ln]
        j = 0
        for l, f_n in enumerate(f_ln):
            for n, f in enumerate(f_n):
                if f > 0:
                    f_j.append(f)
                    n_j.append(n_ln[l][n])
                    l_j.append(l)
                    e_j.append(e_ln[l][n])
                    u_j.append(u_ln[l][n])
                    s_j.append(s_ln[l][n])
                    ptcoef_j.append(self.ptcoef_ln[l][n])
                    j_ln[l][n] = j
                    j += 1
        for l, f_n in enumerate(f_ln):
            for n, f in enumerate(f_n):
                if f == 0:
                    f_j.append(0)
                    n_j.append(n_ln[l][n])
                    l_j.append(l)
                    e_j.append(e_ln[l][n])
                    u_j.append(u_ln[l][n])
                    s_j.append(s_ln[l][n])
                    ptcoef_j.append(self.ptcoef_ln[l][n])
                    j_ln[l][n] = j
                    j += 1
        nj = j

        dK_jj = num.zeros((nj, nj), num.Float)
        for l, j_n in enumerate(j_ln):
            for n1, j1 in enumerate(j_n):
                for n2, j2 in enumerate(j_n):
                    dK_jj[j1, j2] = self.dK_lnn[l][n1, n2]
        
        
        print >> xml, '<?xml version="1.0"?>'
        dtd = 'http://www.fysik.dtu.dk/campos/atomic_setup/paw_setup.dtd'

        print >> xml, '<!DOCTYPE paw_setup SYSTEM'
        print >> xml, '  "%s">' % dtd

        print >> xml, '<paw_setup version="0.2">'

        name = names[self.Z].title()
        comment1 = name + ' setup for the Projector Augmented Wave method.'
        comment2 = 'Units: Hartree and Bohr radii.'
        comment2 += ' ' * (len(comment1) - len(comment2)) 
        print >> xml, '  <!--', comment1, '-->'
        print >> xml, '  <!--', comment2, '-->'

        print >> xml, '  <atom symbol="%s" Z="%d" core="%d" valence="%d"/>' % \
              (self.symbol, self.Z, self.Z - self.Nv, self.Nv)
        if self.xcname == 'LDA':
            type = 'LDA'
            name = 'PW'
        else:
            type = 'GGA'
            name = self.xcname

        print >> xml, '  <xc_functional type="%s" name="%s"/>' % (type, name)
        if self.scalarrel:
            type = 'scalar-relativistic'
        else:
            type = 'non-relativistic'

        print >> xml, '  <generator type="%s" name="gridpaw-%s">' % \
              (type, version)
        print >> xml, '    Frozen core:', self.core or 'none'
        print >> xml, '  </generator>'

        print >> xml, '  <ae_energy kinetic="%f" xc="%f"' % \
              (self.Ekin, self.Exc)
        print >> xml, '             electrostatic="%f" total="%f"/>' % \
              (self.Epot, self.Ekin + self.Exc + self.Epot)

        print >> xml, '  <core_energy kinetic="%f"/>' % Ekincore

        print >> xml, '  <valence_states>'
        ids = []
        line = '    <state n="%d" l="%d" f=%s rc="%4.2f" e="%7.4f" id="%s"/>'
        for l, n, f, e in zip(l_j, n_j, f_j, e_j):
            f = '%-4s' % ('"%d"' % f)
            id = self.symbol + str(n) + 'spdf'[l]
            print >> xml, line % (n, l, f, self.rcut_l[l], e, id)
            ids.append(id)
        print >> xml, '  </valence_states>'

        print >> xml, '  <grid eq="r=a*i/(n-i)" a="%f" n="%d" i="0-%d"' % \
              (self.beta, self.N, self.N - 1), 'id="g1"/>'

        print >> xml, '  <shape_function type="poly3"/>'

        for name, a in [('ae_core_density', nc),
                        ('pseudo_core_density', nct)]:
            print >> xml, '  <%s grid="g1">\n    ' % name,
            for x in a * sqrt(4 * pi):
                print >> xml, '%16.12e' % x,
            print >> xml, '\n  </%s>' % name
        print >> xml, ('  <localized_potential c="%.12e" a="%.6e"/>' %
                       (self.vbar0 * sqrt(4 * pi), self.gamma))
        
        r = self.r
        for l, u, s, ptcoef in zip(l_j, u_j, s_j, ptcoef_j):
            id = ids.pop(0)
            for name, a in [('ae_partial_wave', u),
                            ('pseudo_partial_wave', s)]:
                print >> xml, ('  <%s state="%s" grid="g1">\n    ' %
                               (name, id)),
                p = a.copy()
                p[1:] /= r[1:]
                if l == 0:
                    # XXXXX go to higher order!!!!!
                    p[0] = (p[2] +
                            (p[1] - p[2]) * (r[0] - r[2]) / (r[1] - r[2]))
                for x in p:
                    print >> xml, '%16.12e' % x,
                print >> xml, '\n  </%s>' % name
            print >> xml, ('  <projector_function state="%s" ' % id +
                           'a="%.6e" s="%.6e">\n    ' % (self.gamma,
                                                         scale_radius)),
            for x in ptcoef:
                print >> xml, '%16.12e' % x,
            print >> xml, ('\n  </projector_function>')

        print >> xml, '  <kinetic_energy_differences>\n    ',
        for j1 in range(nj):
            for j2 in range(nj):
                print >> xml, '%16.12e' % dK_jj[j1, j2],
            print >> xml
        print >> xml, '  </kinetic_energy_differences>'

        if X_p is not None:
            print >>xml, '  <exact_exchange_X_matrix>\n    ',
            for x in X_p:
                print >> xml, '%16.12e' % x,
            print >>xml, '\n  </exact_exchange_X_matrix>'

            print >> xml, '  <exact_exchange core-core="%f"/>' % ExxC

        print >> xml, '</paw_setup>'


if __name__ == '__main__':
    for r in [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]:
        gen = Generator('Mg')#, scalarrel=True)
##        gen.Run('[Ar]', r)
##        gen.Run('[Xe]4f', r)
        gen.run('[Ne]', r)
