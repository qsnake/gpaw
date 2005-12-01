# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi, sqrt

import Numeric as num
from LinearAlgebra import solve_linear_equations, inverse
from ASE.ChemicalElements.name import names

from gridpaw.atom.configurations import configurations
from gridpaw.version import version
from gridpaw.atom.all_electron import AllElectron, shoot
from gridpaw.polynomium import a_i, c_l
from gridpaw.utilities.lapack import diagonalize


parameters = {
    #     (core,      rcut)  
    'H' : ('',        0.9),
    'He': ('',        1.5),
    'Li': ('[He]',    2.0),
    'Be': ('[He]',    1.5),
    'C' : ('[He]',    1.0),
    'N' : ('[He]',    1.1),
    'O' : ('[He]',    1.2, {0: [1.0], 1: [1.0], 2: [1.0]}),
    'F' : ('[He]',    1.2, {0: [1.0], 1: [1.0], 2: [1.0]}),
    'Ne': ('[He]',    2.3),    
    'Na': ('[Ne]',    2.3),
    'Mg': ('[Ne]',    2.2),
    'Al': ('[Ne]',    2.0),
    'Si': ('[Ne]',    2.0),
    'P' : ('[Ne]',    2.0),
    'S' : ('[Ne]',    1.87),
    'Cl': ('[Ne]',    1.5),
    'V' : ('[Ar]',   [2.4, 2.4, 2.2], {0: [0.8], 1: [-0.2], 2: [0.8]}),
    'Fe': ('[Ar]',    2.3),
    'Cu': ('[Ar]',   [2.3, 2.3, 2.1]),
    'Ga': ('[Ar]3d',  2.0),
    'As': ('[Ar]',    2.0),
    'Zr': ('[Ar]3d',  2.0),
    'Mo': ('[Kr]',   [2.8, 2.8, 2.3]),
    'Ru': ('[Kr]',   [2.5, 2.4, 2.5], {0: [0.8], 1: [0.0], 2: [0.8]}),
    'Pt': ('[Xe]4f',  2.5),
    'Au': ('[Xe]4f',  2.5)
    }


class Generator(AllElectron):
    def __init__(self, symbol, xcname='LDA', scalarrel=False):
        AllElectron.__init__(self, symbol, xcname, scalarrel)


    def run(self, core, rcut, extra, logderiv=True, vt0=None):

        self.core = core
        if type(rcut) is float:
            rcut_l = [rcut]
        else:
            rcut_l = rcut
            rcut = min(rcut_l)
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
            u = num.where(u < 1e-160, 0, u)  # XXX Numeric!
            Ekincore += f * (e - num.sum((u**2 * self.vr * dr)[1:] / r[1:]))

        # Calculate core density:
        if njcore == 0:
            nc = num.zeros(N, num.Float)
        else:
            uc_j = self.u_j[:njcore]
            uc_j = num.where(uc_j < 1e-160, 0, uc_j)  # XXX Numeric!
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
            if len(extra ) == 0:
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

        rcut_l.extend([max(rcut_l)] * (lmax + 1 - len(rcut_l)))
        
        print 'Cutoffs:',
        for rc, s in zip(rcut_l, 'spd'):
            print 'rc(%s)=%.3f' % (s, rc),
        print
        print 'Kinetic energy of the core states: %.6f' % Ekincore

        # Allocate arrays:
        self.u_ln = u_ln = []  # phi * r
        self.s_ln = s_ln = []  # phi-tilde * r
        self.q_ln = q_ln = []  # p-tilde * r
        for l in range(lmax + 1):
            nn = len(n_ln[l])
            u_ln.append(num.zeros((nn, N), num.Float))
            s_ln.append(num.zeros((nn, N), num.Float))
            q_ln.append(num.zeros((nn, N), num.Float))
            
        # Fill in all-electron wave functions:
        for l in range(lmax + 1):
            # Collect all-electron wave functions:
            u_n = [self.u_j[j] for j in range(njcore, nj) if l_j[j] == l]
            for n, u in enumerate(u_n):
                assert f_ln[l][n] > 0.0
                u_ln[l][n] = u

        # Grid-index corresponding to rcut:
        gcut = int(rcut * N / (rcut + beta))
        gcut_l = [int(rc * N / (rc + beta)) for rc in rcut_l]

        # Outward integration of unbound states stops at 2 * rcut:
        gmax = int(2 * rcut * N / (2 * rcut + beta))
        assert gmax > max(gcut_l)
        
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

        TM = False
        # Construct smooth wave functions:
        coefs = []
        for l, (u_n, s_n) in enumerate(zip(u_ln, s_ln)):
            nodeless = True
            gc = gcut_l[l]
            for u, s in zip(u_n, s_n):
                s[:] = u
                if TM:
                    A = num.ones((4, 4), num.Float)
                    A[:, 0] = 1.0
                    A[:, 1] = r[gc - 2:gc + 2]**2
                    A[:, 2] = A[:, 1]**2
                    A[:, 3] = A[:, 1] * A[:, 2]
                    a = num.log(u[gc - 2:gc + 2] /
                               r[gc - 2:gc + 2]**(l + 1))
                    a = solve_linear_equations(A, a)
                    r1 = r[:gc]
                    r2 = r1**2
                    rl1 = r1**(l + 1)
                    s[:gc] = rl1 * num.exp(a[0] + r2 *
                                            (a[1] + r2 *
                                             (a[2] + r2 *
                                              a[3])))
                else:
                    A = num.ones((4, 4), num.Float)
                    A[:, 0] = 1.0
                    A[:, 1] = r[gc - 2:gc + 2]**2
                    A[:, 2] = A[:, 1]**2
                    A[:, 3] = A[:, 1] * A[:, 2]
                    a = u[gc - 2:gc + 2] / r[gc - 2:gc + 2]**(l + 1)
                    a = solve_linear_equations(A, a)
                    r1 = r[:gc]
                    r2 = r1**2
                    rl1 = r1**(l + 1)
                    s[:gc] = rl1 * (a[0] +
                                      r2 * (a[1] + r2 * (a[2] + r2 * a[3])))

                coefs.append(a)
                if nodeless:
                    # The first state for each l must be nodeless:
##                    assert num.alltrue(s[1:gc] > 0.0)
                    nodeless = False

        Nc = Z - self.Nv
        Nctail = 4 * pi * num.dot(nc[gcut:], dv[gcut:])
        print 'Core states: %d (r > %.3f: %.6f)' % (Nc, rcut, Nctail)
        assert Nctail < 1.1
        print 'Valence states: %d' % self.Nv

        # Calculate soft core densities:
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
        
        # ... and the soft valence density:
        nt = num.zeros(N, num.Float)
        for f_n, s_n in zip(f_ln, s_ln):
            nt += num.dot(f_n, s_n**2) / (4 * pi)
        nt[1:] /= r[1:]**2
        nt[0] = nt[1]
        nt += nct

        # Calculate the shape function:
        x = r / rcut
        gt = num.zeros(N, num.Float)
        for i in range(4):
            gt += a_i[i] * x**i
        gt[gcut:] = 0.0
        gt *= c_l[0] / rcut**3
        norm = num.dot(gt, dv)
        print norm
        assert abs(norm - 1) < 1e-3
        gt /= norm
        
        # Calculate neutral smooth charge density:
        norm = num.dot(nt, dv)
        rhot = nt - norm * gt
        print 'Pseudo-electron charge: %.6f' % (4 * pi * norm)

        vHt = self.calculate_hartree_potential(rhot)
        vHt[1:] /= r[1:]
        vHt[0] = vHt[1]

        vXCt = num.zeros(N, num.Float)
        Exct = self.xc.get_energy_and_potential(nt, vXCt)
        self.vt = vt = vHt + vXCt

        # Construct localized potential:
        if vt0 is None:
            A = num.ones((2, 2), num.Float)
            A[0] = 1.0
            A[1] = r[gcut - 1:gcut + 1]**2
            a = vt[gcut - 1:gcut + 1]
            a = solve_linear_equations(num.transpose(A), a)
            r2 = r**2
            vbar = a[0] + r2 * a[1]
        else:
            A = num.ones((2, 2), num.Float)
            A[0] = r[gcut - 1:gcut + 1]**2
            A[1] = A[0]**2
            a = vt[gcut - 1:gcut + 1] - vt0
            a = solve_linear_equations(num.transpose(A), a)
            r2 = r**2
            vbar = vt0 + r2 * (a[0] + r2 * a[1])
        vbar -= vt
        vbar[gcut:] = 0.0
        vt += vbar
        
        # Construct projector functions:
        for l, (e_n, s_n, q_n) in enumerate(zip(e_ln, s_ln, q_ln)):
            for e, s, q in zip(e_n, s_n, q_n):
                a = coefs.pop(0)
                for k in range(3):
                    b = l + 1 + 2 * k
                    q += 0.5 * a[k + 1] * (l * (l + 1) -
                                           (b + 2) * (b + 1)) * r**b
                q += (vt - e) * s
                q[gcut_l[l]:] = 0.0

        # Calculate matrix elements:
        dK_ln1n2 = []
        self.dH_ln1n2 = dH_ln1n2 = []
        self.dO_ln1n2 = dO_ln1n2 = []
        for l, (e_n, u_n, s_n, q_n) in enumerate(zip(e_ln, u_ln,
                                                     s_ln, q_ln)):
            ku_n = [self.kin(l, u, e) for u, e in zip(u_n, e_n)]  
            ks_n = [self.kin(l, s) for s in s_n]
            dK_n1n2 = 0.5 * (num.innerproduct(u_n, ku_n * dr) -
                            num.innerproduct(s_n, ks_n * dr))
            dK_n1n2 += num.transpose(dK_n1n2)
            dO_n1n2 = (num.innerproduct(u_n, u_n * dr) -
                       num.innerproduct(s_n, s_n * dr))
            nn = len(e_n)
            e_n1n2 = num.zeros((nn, nn), num.Float)
            e_n1n2.flat[::nn + 1] = e_n
            A_n1n2 = num.innerproduct(q_n, s_n * dr)
            dH_n1n2 = num.dot(e_n1n2, dO_n1n2) - A_n1n2
            
            dK_ln1n2.append(dK_n1n2)
            dO_ln1n2.append(dO_n1n2)
            dH_ln1n2.append(dH_n1n2)

            # Orthonormalize projector functions:
            q_n[:] = num.dot(inverse(A_n1n2), q_n)

        print 'state    eigenvalue         norm'
        print '--------------------------------'
        for l, (n_n, f_n, e_n) in enumerate(zip(n_ln, f_ln, e_ln)):
            for n in range(len(e_n)):
                print '%d%s^%-2d: %12.6f' % (n_n[n], 'spd'[l], f_n[n], e_n[n]),
                if f_n[n] > 0.0:
                    print '%12.6f' % (1 - dO_ln1n2[l][n, n])
                else:
                    print
        print '--------------------------------'

        if logderiv:
            # Calculate logarithmic derivatives:
            gld = max(gcut_l) + 10
            assert gld < gmax
            print 'Calculating logarithmic derivatives at r=%.3f' % r[gld]
            print '(skip with [Ctrl-C])'

            try:
                u = num.zeros(N, num.Float)
                for l in range(3):
                    if l <= lmax:
                        dO_n1n2 = dO_ln1n2[l]
                        dH_n1n2 = dH_ln1n2[l]
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
                            A_n1n2 = dH_n1n2 - e * dO_n1n2
                            s_n = [self.integrate(l, vt, e, gld, q) for q in q_n]
                            B_n1n2 = num.innerproduct(q_n, s_n * dr)
                            a_n = num.dot(q_n, s * dr)

                            B_n1n2 = num.dot(A_n1n2, B_n1n2)
                            B_n1n2.flat[::len(a_n) + 1] += 1.0
                            c_n = solve_linear_equations(B_n1n2,
                                                         num.dot(A_n1n2, a_n))
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
            
        self.write_xml(n_ln, f_ln, e_ln, u_ln, s_ln, q_ln,
                      nc, nct, Ekincore, dK_ln1n2, vbar)

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
                           num.dot(self.dH_ln1n2[l], q_n)) * h
                S = num.dot(num.transpose(q_n),
                           num.dot(self.dO_ln1n2[l], q_n)) * h
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
            if (f > 0 and abs(e - e0) > 0.001) or (f == 0 and e0 < self.emax):
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


    def write_xml(self, n_ln, f_ln, e_ln, u_ln, s_ln, q_ln,
                 nc, nct, Ekincore, dK_ln1n2, vbar):
        xml = open(self.symbol + '.' + self.xcname, 'w')

        if self.ghost:
            raise RuntimeError

        print >> xml, '<?xml version="1.0"?>'
        dtd = 'http://www.fysik.dtu.dk/campos/atomic_setup/paw_setup.dtd'

        print >> xml, '<!DOCTYPE paw_setup SYSTEM'
        print >> xml, '  "%s">' % dtd

        print >> xml, '<paw_setup version="0.1">'

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
        for l, (n_n, f_n, e_n) in enumerate(zip(n_ln, f_ln, e_ln)):
            for n, f, e in zip(n_n, f_n, e_n):
                f = '%-4s' % ('"%d"' % f)
                id = self.symbol + str(n) + 'spdf'[l]
                print >> xml, line % (n, l, f, self.rcut_l[l], e, id)
                ids.append(id)
        print >> xml, '  </valence_states>'

        print >> xml, '  <grid eq="r=a*i/(n-i)" a="%f" n="%d" i="0-%d"' % \
              (self.beta, self.N, self.N - 1), 'id="g1"/>'

        print >> xml, '  <shape_function type="poly3"/>'

        for name, a in [('ae_core_density', nc),
                        ('pseudo_core_density', nct),
                        ('localized_potential', vbar)]:
            print >> xml, '  <%s grid="g1">\n    ' % name,
            for x in a * sqrt(4 * pi):
                print >> xml, '%16.12e' % x,
            print >> xml, '\n  </%s>' % name
        
        r = self.r
        for l, (u_n, s_n, q_n) in enumerate(zip(u_ln, s_ln, q_ln)):
            for u, s, q in zip(u_n, s_n, q_n):
                id = ids.pop(0)
                for name, a in [('ae_partial_wave', u),
                                ('pseudo_partial_wave', s),
                                ('projector_function', q)]:
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

        print >> xml, '  <kinetic_energy_differences>\n    ',

        nj = sum([len(dK_n1n2) for dK_n1n2 in dK_ln1n2])
        dK_j1j2 = num.zeros((nj, nj), num.Float)
        j1 = 0
        for dK_n1n2 in dK_ln1n2:
            j2 = j1 + len(dK_n1n2)
            dK_j1j2[j1:j2, j1:j2] = dK_n1n2
            j1 = j2
        assert j1 == nj
        for j1 in range(nj):
            for j2 in range(nj):
                print >> xml, '%16.12e' % dK_j1j2[j1, j2],
        print >> xml, '\n  </kinetic_energy_differences>'

        print >> xml, '</paw_setup>'


if __name__ == '__main__':
    for r in [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]:
        gen = Generator('Mg')#, scalarrel=True)
##        gen.Run('[Ar]', r)
##        gen.Run('[Xe]4f', r)
        gen.run('[Ne]', r)
