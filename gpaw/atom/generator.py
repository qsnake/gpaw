# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import sys
from math import pi, sqrt

import Numeric as num
num.inner = num.innerproduct # XXX numpy!
from FFT import real_fft, inverse_real_fft
from LinearAlgebra import solve_linear_equations, inverse
from ASE.ChemicalElements.name import names

from gpaw.atom.configurations import configurations
from gpaw.version import version
from gpaw.atom.all_electron import AllElectron, shoot
from gpaw.polynomium import a_i, c_l
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import hartree
from gpaw.exx import constructX
from gpaw.exx import atomic_exact_exchange as aExx
from gpaw.utilities import hartree
from gpaw.atom.filter import Filter


parameters = {
    #     (  core,  cutoff,  extra projectors)  
    'H' : ('',        0.9),
    'He': ('',        1.5),
    'Li': ('[He]',    2.1),
    'Be': ('[He]',    1.5),
    'C' : ('[He]',    1.0),
    'N' : ('[He]',    1.1),
    'O' : ('[He]',    1.2),# {0: [0.0], 1: [0.0], 2: [0.0]}),
    'F' : ('[He]',    1.2),# {0: [-0.99], 1: [-0.15], 2: [0.0]}),
    'Ne': ('[He]',    1.8),    
    'Na': ('[Ne]',    2.3),
    'Mg': ('[Ne]',    2.0),
    'Al': ('[Ne]',    2.0),
    'Si': ('[Ne]',    2.0),
    'P' : ('[Ne]',    2.0),
    'S' : ('[Ne]',    1.87),
    'Cl': ('[Ne]',    1.5),
    'Ar': ('[Ne]',    1.6),
    'V' : ('[Ar]',   [2.5, 2.4, 2.2]),
    'Cr': ('[Ar]',   [2.4, 2.4, 2.2]),
    'Fe': ('[Ar]',    2.3),
    'Ni': ('[Ar]',    2.3),
    'Cu': ('[Ar]',   [2.3, 2.2, 2.1]),
    'Ga': ('[Ar]3d',  2.2),
    'As': ('[Ar]',    2.0),
    'Kr': ('[Ar]3d',  2.2),
    'Zr': ('[Ar]3d',  2.0),
    'Mo': ('[Kr]',   [2.8, 2.8, 2.3]),
    'Ru': ('[Kr]',    2.6),
    'Pd': ('[Kr]',   [2.3, 2.5, 2.0]),
    'Ag': ('[Kr]',    2.5),
    'Pt': ('[Xe]4f', [2.3, 2.5, 2.0]),
    'Au': ('[Xe]4f',  2.5)
    }

class Generator(AllElectron):
    def __init__(self, symbol, xcname='LDA', scalarrel=False):
        AllElectron.__init__(self, symbol, xcname, scalarrel)


    def run(self, core, rcut, extra,
            logderiv=True, vt0=None, exx=False):

        self.core = core
        if type(rcut) is float:
            rcut_l = [rcut]
        else:
            rcut_l = rcut
            rcut = max(rcut_l)
        self.rcut = rcut
        self.rcut_l = rcut_l

        Z = self.Z

        n_j = self.n_j
        l_j = self.l_j
        f_j = self.f_j
        e_j = self.e_j

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

        if 2 in l_j[njcore:]:
            # We have a bound valence d-state.  Add bound s- and
            # p-states if not already there:
            for l in [0, 1]:
                if l not in l_j[njcore:]:
                    n_j.append(1 + l + l_j.count(l))
                    l_j.append(l)
                    f_j.append(0.0)
                    e_j.append(-0.01)

        nj = len(n_j)
                    
        self.Nv = sum(f_j[njcore:])

        # Do all-electron calculation:
        AllElectron.run(self)

        # Highest occupied atomic orbital:
        self.emax = max(e_j)
        
        N = self.N
        r = self.r
        dr = self.dr
        d2gdr2 = self.d2gdr2
        beta = self.beta

        dv = r**2 * dr
        
        print
        print 'Generating PAW setup'
        if core != '':
            print 'Frozen core:', core
            
        # So far - no ghost-states:
        self.ghost = False

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
                nn = -1
                for e in extra[l]:
                    n_ln[l].append(nn)
                    f_ln[l].append(0.0)
                    e_ln[l].append(e)
                    nn -= 1
        else:
            # Automatic:

            # Make sure we have two projectors for each occupied channel:
            for l in range(lmax + 1):
                if len(n_ln[l]) < 2:
                    # Only one - add one more:
                    assert len(n_ln[l]) == 1
                    n_ln[l].append(-1)
                    f_ln[l].append(0.0)
                    e_ln[l].append(1.0 + e_ln[l][0])

            if lmax < 2:
                # Add extra projector for l = lmax + 1:
                n_ln.append([-1])
                f_ln.append([0.0])
                e_ln.append([0.0])
                lmax += 1

        self.lmax = lmax

        rcut_l.extend([rcut] * (lmax + 1 - len(rcut_l)))

        print 'Cutoffs:',
        for rc, s in zip(rcut_l, 'spdf'):
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
                u_ln[l][n] = u

        # Grid-index corresponding to rcut:
        gcut = 1 + int(rcut * N / (rcut + beta))
        gcut_l = [1 + int(rc * N / (rc + beta)) for rc in rcut_l]

        rcut2 = 2 * rcut
        gcut2 = 1 + int(rcut2 * N / (rcut2 + beta))

        # Outward integration of unbound states stops at 3 * rcut:
        gmax = int(3 * rcut * N / (3 * rcut + beta))
        assert gmax > gcut2
        
        # Calculate unbound extra states:
        c2 = -(r / dr)**2
        c10 = -d2gdr2 * r**2
        for l, (n_n, e_n, u_n) in enumerate(zip(n_ln, e_ln, u_ln)):
            for n, e, u in zip(n_n, e_n, u_n):
                if n < 0:
                    u[:] = 0.0
                    shoot(u, l, self.vr, e, self.r2dvdr, r, dr, c10, c2,
                          self.scalarrel, gmax=gmax)
                    u *= 1.0 / u[gcut]

        Nc = Z - self.Nv
        Nctail = 4 * pi * num.dot(nc[gcut:], dv[gcut:])
        print 'Core states: %d (r > %.3f: %.6f)' % (Nc, rcut, Nctail)
        assert Nctail < 1.1
        print 'Valence states: %d' % self.Nv

        # Construct smooth wave functions:
        coefs = []
        for l, (u_n, s_n) in enumerate(zip(u_ln, s_ln)):
            nodeless = True
            gc = gcut_l[l]
            for u, s in zip(u_n, s_n):
                s[:] = u
                if 1:
                    A = num.ones((4, 4), num.Float)
                    A[:, 0] = 1.0
                    A[:, 1] = r[gc - 2:gc + 2]**2
                    A[:, 2] = A[:, 1]**2
                    A[:, 3] = A[:, 1] * A[:, 2]
                    a = u[gc - 2:gc + 2] / r[gc - 2:gc + 2]**(l + 1)
                    if 0:#l < 2 and nodeless:
                        a = num.log(a)
                    a = solve_linear_equations(A, a)
                    r1 = r[:gc]
                    r2 = r1**2
                    rl1 = r1**(l + 1)
                    y = a[0] + r2 * (a[1] + r2 * (a[2] + r2 * (a[3])))
                    if 0:#l < 2 and nodeless:
                        y = num.exp(y)
                    s[:gc] = rl1 * y
                else:
                    A = num.zeros((5, 5), num.Float)
                    A[:4, 0] = 1.0
                    A[:4, 1] = r[gc - 2:gc + 2]**2
                    A[:4, 2] = A[:4, 1]**2
                    A[:4, 3] = A[:4, 1] * A[:4, 2]
                    A[:4, 4] = A[:4, 2]**2
                    A[4, 4] = 1.0
                    a = u[gc - 2:gc + 3] / r[gc - 2:gc + 3]**(l + 1)
                    def f(x):
                        a[4] = x
                        b = solve_linear_equations(A, a)
                        r1 = r[:gc]
                        r2 = r1**2
                        rl1 = r1**(l + 1)
                        y = b[0] + r2 * (b[1] + r2 * (b[2] + r2 * (b[3] + r2 * b[4])))
                        s[:gc] = rl1 * y
                        return num.dot(s**2, dr)
                    x1 = 0
                    x2 = 1
                    f1 = f(x1)
                    f2 = f(x2)
                    while abs(f1 - 1) > 1e-6:
                        x3 = 0#???????????????
                    print f(0),f(1.0), f(0), f(.5);adfg
                coefs.append(a)
                if nodeless:
                    if not num.alltrue(s[1:gc] > 0.0):
                        print ('Error:  The %d%s pseudo wave has a node!' %
                               (n_ln[l][0], 'spdf'[l]))
                        raise SystemExit
                    # Only the first state for each l must be nodeless:
                    nodeless = False

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
        
        # ... and the soft valence density:
        nt = num.zeros(N, num.Float)
        for f_n, s_n in zip(f_ln, s_ln):
            nt += num.dot(f_n, s_n**2) / (4 * pi)
        nt[1:] /= r[1:]**2
        nt[0] = nt[1]
        nt += nct

        # Calculate the shape function:
        x = r / rcut
        gaussian = num.zeros(N, num.Float)
        self.gamma = gamma = 10.0
        gaussian[:gcut2] = num.exp(-gamma * x[:gcut2]**2)
        gaussian2 = num.zeros(N, num.Float)
        gaussian2[:gcut2] = num.exp(-gamma * x[:gcut2]**2)
        gt = 4 * (gamma / rcut**2)**1.5 / sqrt(pi) * gaussian
        norm = num.dot(gt, dv)
##        print norm, norm-1
        assert abs(norm - 1) < 1e-2
        gt /= norm
        

        # Calculate neutral smooth charge density:
        Nt = num.dot(nt, dv)
        rhot = nt - Nt * gt
        print 'Pseudo-electron charge', 4 * pi * Nt

        vHt = num.zeros(N, num.Float)
        hartree(0, rhot * r * dr, self.beta, self.N, vHt)
        vHt[1:] /= r[1:]
        vHt[0] = vHt[1]

        vXCt = num.zeros(N, num.Float)
        Exct = self.xc.get_energy_and_potential(nt, vXCt)
        vt = vHt + vXCt

        # Construct zero potential:
        if 0:
            A = num.ones((2, 2), num.Float)
            A[0] = 1.0
            A[1] = r[1:7:3]**2
            a = vt[1:7:3]
            a = solve_linear_equations(num.transpose(A), a)
            self.vbar0 = a[1] * rcut**2 / gamma

            self.vbar0 = 0.0
            vbar = self.vbar0 * gaussian2

            vt += vbar
        else:
            A = num.ones((2, 2), num.Float)
            A[0] = 1.0
            A[1] = r[gcut - 1:gcut + 1]**2
            #A[2] = A[1]**2
            a = vt[gcut - 1:gcut + 1]
            a = solve_linear_equations(num.transpose(A), a)
            r2 = r**2
            vbar = a[0] + r2 * (a[1])## + r2 * a[2])
            vbar -= vt
            vbar[gcut:] = 0.0
            vt += vbar

        
        # Construct projector functions:
        for l, (e_n, s_n, q_n) in enumerate(zip(e_ln, s_ln, q_ln)):
            gc = gcut_l[l]
            for e, s, q in zip(e_n, s_n, q_n):
                if 1:
                    q[:] = self.kin(l, s)
                else:
                    a = coefs.pop(0)
                    for k in range(3):
                        b = l + 1 + 2 * k
                        q += 0.5 * a[k + 1] * (l * (l + 1) -
                                               (b + 2) * (b + 1)) * r**b
                q += (vt - e) * s
                q[gcut:] = 0.0

        filter = Filter(r, dr, rcut2, 0.4).filter

        vbar = filter(vbar * r)
        
        # Calculate matrix elements:
        self.dK_lnn = dK_lnn = []
        self.dH_lnn = dH_lnn = []
        self.dO_lnn = dO_lnn = []
        for l, (e_n, u_n, s_n, q_n) in enumerate(zip(e_ln, u_ln,
                                                     s_ln, q_ln)):
            A_nn = num.innerproduct(s_n, q_n * dr)
            # Do a LU decomposition of A:
            nn = len(e_n)
            if nn == 1:
                L_nn = num.array([[1.0]])
                U_nn = num.array([[A_nn[0, 0]]])
            else:
                L_nn = num.array([[1.0, 0.0],
                                  [A_nn[1, 0] / A_nn[0, 0], 1.0]])
                U_nn = num.array([[A_nn[0, 0], A_nn[0, 1]],
                                  [0.0, A_nn[1, 1] - L_nn[1, 0] * A_nn[0, 1]]])

            dO_nn = (num.innerproduct(u_n, u_n * dr) -
                     num.innerproduct(s_n, s_n * dr))

            e_nn = num.zeros((nn, nn), num.Float)
            e_nn.flat[::nn + 1] = e_n
            dH_nn = num.dot(dO_nn, e_nn) - A_nn

            q_n[:] = num.dot(inverse(num.transpose(U_nn)), q_n)
            s_n[:] = num.dot(inverse(L_nn), s_n)
            u_n[:] = num.dot(inverse(L_nn), u_n)

            dO_nn = num.dot(num.dot(inverse(L_nn), dO_nn),
                            inverse(num.transpose(L_nn)))
            dH_nn = num.dot(num.dot(inverse(L_nn), dH_nn),
                            inverse(num.transpose(L_nn)))

            ku_n = [self.kin(l, u, e) for u, e in zip(u_n, e_n)]  
            ks_n = [self.kin(l, s) for s in s_n]
            dK_nn = 0.5 * (num.innerproduct(u_n, ku_n * dr) -
                           num.innerproduct(s_n, ks_n * dr))
            dK_nn += num.transpose(dK_nn).copy()

            dK_lnn.append(dK_nn)
            dO_lnn.append(dO_nn)
            dH_lnn.append(dH_nn)

            for n, q in enumerate(q_n):
                q[:] = filter(q, l) * r**(l + 1)

            A_nn = num.innerproduct(s_n, q_n * dr)
            q_n[:] = num.dot(inverse(num.transpose(A_nn)), q_n)
            
        self.vt = vt

        print 'state    eigenvalue         norm'
        print '--------------------------------'
        for l, (n_n, f_n, e_n) in enumerate(zip(n_ln, f_ln, e_ln)):
            for n in range(len(e_n)):
                if n_n[n] > 0:
                    f = '(%d)' % f_n[n]
                    print '%d%s%-4s: %12.6f %12.6f' % (
                        n_n[n], 'spdf'[l], f, e_n[n],
                        num.dot(s_ln[l][n]**2, dr))
                else:
                    print '*%s    : %12.6f' % ('spdf'[l], e_n[n])
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

                    fae = open(self.symbol + '.ae.ld.' + 'spdf'[l], 'w')
                    fps = open(self.symbol + '.ps.ld.' + 'spdf'[l], 'w')

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
                if n < 0:
                    self.write(u, 'ae', n=n, l=l)
                self.write(s, 'ps', n=n, l=l)
                self.write(q, 'proj', n=n, l=l)

        for h in [0.05]:
            self.diagonalize(h)

        self.vn_j = vn_j = []
        self.vl_j = vl_j = []
        self.vf_j = vf_j = []
        self.ve_j = ve_j = []
        self.vu_j = vu_j = []
        self.vs_j = vs_j = []
        self.vq_j = vq_j = []
        j_ln = [[0 for f in f_n] for f_n in f_ln]
        j = 0
        for l, n_n in enumerate(n_ln):
            for n, nn in enumerate(n_n):
                if nn > 0:
                    vf_j.append(f_ln[l][n])
                    vn_j.append(nn)
                    vl_j.append(l)
                    ve_j.append(e_ln[l][n])
                    vu_j.append(u_ln[l][n])
                    vs_j.append(s_ln[l][n])
                    vq_j.append(q_ln[l][n])
                    j_ln[l][n] = j
                    j += 1
        for l, n_n in enumerate(n_ln):
            for n, nn in enumerate(n_n):
                if nn < 0:
                    vf_j.append(0)
                    vn_j.append(nn)
                    vl_j.append(l)
                    ve_j.append(e_ln[l][n])
                    vu_j.append(u_ln[l][n])
                    vs_j.append(s_ln[l][n])
                    vq_j.append(q_ln[l][n])
                    j_ln[l][n] = j
                    j += 1
        nj = j

        self.dK_jj = num.zeros((nj, nj), num.Float)
        for l, j_n in enumerate(j_ln):
            for n1, j1 in enumerate(j_n):
                for n2, j2 in enumerate(j_n):
                    self.dK_jj[j1, j2] = self.dK_lnn[l][n1, n2]
        
        if exx:
            X_p = constructX(self)
            ExxC = aExx(self, 'core-core')
        else:
            X_p = None
            ExxC = None
            
        self.write_xml(vl_j, vn_j, vf_j, ve_j, vu_j, vs_j, vq_j,
                       nc, nct, nt, Ekincore, X_p, ExxC, vbar)

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
        print
        print 'state   all-electron     PAW'
        print '-------------------------------'
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
            error = diagonalize(H, e_n, S)
            if error != 0:
                raise RuntimeError
            ePAW = e_n[0]
            if l <= self.lmax and self.n_ln[l][0] > 0:
                eAE = self.e_ln[l][0]
                print '%d%s:   %12.6f %12.6f' % (self.n_ln[l][0],
                                                 'spdf'[l], eAE, ePAW),
                if abs(eAE - ePAW) > 0.014:
                    print '  GHOST-STATE!'
                    self.ghost = True
                else:
                    print
            else:
                print '*%s:                %12.6f' % ('spdf'[l], ePAW),
                if ePAW < self.emax:
                    print '  GHOST-STATE!'
                    self.ghost = True
                else:
                    print
        print '-------------------------------'

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


    def write_xml(self, vl_j, vn_j, vf_j, ve_j, vu_j, vs_j, vq_j,
                  nc, nct, nt, Ekincore, X_p, ExxC, vbar):
        xml = open(self.symbol + '.' + self.xcname, 'w')

        if self.ghost:
            raise SystemExit

        print >> xml, '<?xml version="1.0"?>'
        print >> xml, '<paw_setup version="0.6">'

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

        print >> xml, '  <generator type="%s" name="gpaw-%s">' % \
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
        line1 = '    <state n="%d" l="%d" f=%s rc="%5.3f" e="%8.5f" id="%s"/>'
        line2 = '    <state       l="%d"        rc="%5.3f" e="%8.5f" id="%s"/>'
        for l, n, f, e in zip(vl_j, vn_j, vf_j, ve_j):
            if n > 0:
                f = '%-4s' % ('"%d"' % f)
                id = '%s-%d%s' % (self.symbol, n, 'spdf'[l])
                print >> xml, line1 % (n, l, f, self.rcut_l[l], e, id)
            else:
                id = '%s-%s%d' % (self.symbol, 'spdf'[l], -n)
                print >> xml, line2 % (l, self.rcut_l[l], e, id)
            ids.append(id)
        print >> xml, '  </valence_states>'

        print >> xml, ('  <radial_grid eq="r=a*i/(n-i)" a="%f" n="%d" ' +
                       'istart="0" iend="%d" id="g1"/>') % \
                       (self.beta, self.N, self.N - 1)

        rcgauss = self.rcut / sqrt(self.gamma)
        print >> xml, ('  <shape_function type="gauss" rc="%.12e"/>' %
                       rcgauss)

        for name, a in [('ae_core_density', nc),
                        ('pseudo_core_density', nct),
                        ('pseudo_valence_density', nt - nct),
                        ('zero_potential', vbar)]:
            print >> xml, '  <%s grid="g1">\n    ' % name,
            for x in a * sqrt(4 * pi):
                print >> xml, '%16.12e' % x,
            print >> xml, '\n  </%s>' % name

        r = self.r
        for l, u, s, q, in zip(vl_j, vu_j, vs_j, vq_j):
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

        print >> xml, '  <kinetic_energy_differences>',
        nj = len(self.dK_jj)
        for j1 in range(nj):
            print >> xml, '\n    ',
            for j2 in range(nj):
                print >> xml, '%16.12e' % self.dK_jj[j1, j2],
        print >> xml, '\n  </kinetic_energy_differences>'

        if X_p is not None:
            print >>xml, '  <exact_exchange_X_matrix>\n    ',
            for x in X_p:
                print >> xml, '%16.12e' % x,
            print >>xml, '\n  </exact_exchange_X_matrix>'

            print >> xml, '  <exact_exchange core-core="%f"/>' % ExxC

        print >> xml, '</paw_setup>'
