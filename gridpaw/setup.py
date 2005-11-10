# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
from math import log, pi, sqrt
import sys

import Numeric as num
from ASE.ChemicalElements.name import names

from gridpaw.read_setup import PAWXMLParser
from gridpaw import setup_home
from gridpaw.gaunt import gaunt as G_L1L2L
from gridpaw.spline import Spline
from gridpaw.grid_descriptor import RadialGridDescriptor
from gridpaw.utilities import unpack, erf
from gridpaw.xc_atom import XCAtom
from gridpaw.xc_functional import XCOperator
from gridpaw.polynomium import a_i, c_l, b_lj


fac = [1.0]
for n in range(1, 10):
    fac.append(fac[-1] * n)


class Hartree:
    #
    #     2
    # 1  d        l(l + 1)         __
    # - ---(vr) - -------- v = - 4 || n
    # r   2           2
    #   dr           r
    #
    #   2        2
    #  d        d g  dr 2 d        l(l + 1)  dr 2          __      dr 2 
    # ---(vr) + --- (--)  --(vr) - -------- (--)  vr = - 4 || n r (--)
    #   2         2  dg   dg           2     dg                    dg
    # dg        dr                    r
    #
    def __init__(self, a1_g, a2_lg, a3_g, r_g, dr_g):
        self.a1_g = a1_g
        self.a2_lg = a2_lg
        self.a3_g = a3_g
        self.r_g = r_g
        self.dr_g = dr_g

    def solve(self, n_g, l):
        # for N = 6:
        #
        # vr_5 = 0
        #
        # | a1_0 a2_0 a3_0   0    0  |   | vr_0 |   | c_1 |
        # |   0  a1_1 a2_1 a3_1   0  |   | vr_1 |   | c_2 |
        # |   0    0  a1_2 a2_2 a3_2 | * | vr_2 | = | c_3 |
        # |   0    0    0  a1_3 a2_3 |   | vr_3 |   | c_4 |
        # |   0    0    0    0  a1_4 |   | vr_4 |   | c_5 |
        #               __
        #             4 || Q
        # v  <--  v + ------
        #               l+1
        #              r
        #
        gcut = len(n_g)
        Q = num.dot(self.r_g**(2 + l) * self.dr_g, n_g)
##        print 'Q:', Q
        c_g = -4 * pi * self.r_g * self.dr_g**2 * n_g
        vr_g = num.zeros(gcut, num.Float)
        g = gcut - 2
        vr_g[g] = (c_g[g + 1] - self.a2_lg[l, g] * vr_g[g + 1]) / self.a1_g[g]
        while g > 0:
            g -= 1
            vr_g[g] = (c_g[g + 1]
                      - self.a2_lg[l, g] * vr_g[g + 1]
                      - self.a3_g[g] * vr_g[g + 2]) / self.a1_g[g]

        if l == 0:
            vr_g += 4 * pi * Q
        else:
            vr_g[1:] += 4 * pi * Q / (2 * l + 1) * self.r_g[1:]**-l
        return vr_g * self.r_g * self.dr_g


class Setup:
    def __init__(self, symbol, xcfunc, lmax=0, nspins=1, softgauss=True):
        xcname = xcfunc.get_xc_name()
        self.symbol = symbol
        self.xcname = xcname
        self.softgauss = softgauss
        
        filename = os.path.join(setup_home, symbol + '.' + xcname)

        (Z, Nc, Nv,
         e_total, e_kinetic, e_electrostatic, e_xc,
         e_kinetic_core,
         n_j, l_j, f_j, eps_j, rcut_j, id_j,
         grid, rcgauss,
         nc_g, nct_g, vbar_g,
         phi_jg, phit_jg, pt_jg,
         e_kin_j1j2,
         self.fingerprint,
         filename) = PAWXMLParser().parse(filename)

        self.filename = filename
        self.Nv = Nv
        self.Z = Z

        nj0 = len(l_j)
        e_kin_j1j2.shape = (nj0, nj0)

        nj = 3
        if Z > 4:
            nj = 5
        if Z > 18:
            nj = 6

        if nj > nj0:
            nj = nj0
            
        e_kin_j1j2 = e_kin_j1j2[:nj, :nj].copy()
        
        n_j = n_j[:nj]
        l_j = l_j[:nj]
        f_j = f_j[:nj]
        eps_j = eps_j[:nj]
        rcut_j = rcut_j[:nj]
        id_j = id_j[:nj]
        phi_jg = phi_jg[:nj]
        phit_jg = phit_jg[:nj]
        pt_jg = pt_jg[:nj]

        self.l_j = l_j
        self.f_j = f_j
        
        imin, imax = [int(x) for x in grid['i'].split('-')]
        ng = imax - imin + 1
##        print grid['eq']
        if grid['eq'] == 'r=a*i/(n-i)':
            a = float(grid['a'])
            assert int(grid['n']) == ng
        else:
            raise RuntimeError, 'Unknown grid type: ' + grid['eq']

        i = num.arange(ng, typecode=num.Float)
        r_g = a * i / (ng - i)
        dr_g = a * ng / (ng - i)**2
        d2gdr2 = -2 * ng * a / (a + r_g)**3
        
##        print e_kin_j1j2

        # Normalize everthing:
        Nc0 = sqrt(4 * pi) * num.dot(nc_g, r_g**2 * dr_g)
##        print 'Core electrons', Nc0
        if Nc > 0:
            nc_g *= Nc / Nc0
            nct_g *= Nc / Nc0
            
        for j in range(nj):
            norm = num.dot((phi_jg[j] * r_g)**2, dr_g)
##            print id_j[j], norm
            x = 1.0 / sqrt(norm)
            phi_jg[j] *= x
            phit_jg[j] *= x
            pt_jg[j] /= x
            e_kin_j1j2[j] *= x
            e_kin_j1j2[:, j] *= x

        # Find cutoff for core density:
        if Nc == 0:
            gcore = 7
        else:
            N = 0.0
            g = ng - 1
            while N < 1e-7:
                N += sqrt(4 * pi) * nc_g[g] * r_g[g]**2 * dr_g[g]
                g -= 1
            gcore = g
        rcore = r_g[gcore]

        ni = 0
        niAO = 0
        i = 0
        j = 0
        jlL_i = []
        self.nk = 0
        lmax1 = 0
        for l, f in zip(l_j, f_j):
            self.nk += 3 + l * (1 + 2 * l)
            if l > lmax1:
                lmax1 = l
            if f > 0:
                niAO += 2 * l + 1
            for m in range(2 * l + 1):
                jlL_i.append((j, l, l**2 + m))
                i += 1
            j += 1
        ni = i
        self.ni = ni
        self.niAO = niAO
        
        if 2 * lmax1 < lmax:
            lmax = 2 * lmax1

        if lmax1 > 1:
            lmax1 = 1
            
        np = ni * (ni + 1) / 2
        nq = nj * (nj + 1) / 2
        Lmax = (lmax + 1)**2
        
        # Find *one* cutoff for partial waves:
        rcut = max(rcut_j)
        gcut = int(ng * rcut / (a + rcut))
##        rcut = r_g[gcut]  # XXXXXX
##        print 'rcut:', rcut
        
        # Construct splines:
        self.nct = Spline(0, rcore, nct_g, r_g, a, zero=True)
        self.vbar = Spline(0, rcut, vbar_g, r_g, a, zero=True)

        self.vHt = None
        
        def grr(phi_g, l, r_g):
            w_g = phi_g.copy()
            if l > 0:
                w_g[1:] /= r_g[1:]**l
                w1, w2 = w_g[1:3]
                r0, r1, r2 = r_g[0:3]
                w_g[0] = w2 + (w1 - w2) * (r0 - r2) / (r1 - r2) 
            return w_g
        
        self.pt_j = []
        for j, pt_g in enumerate(pt_jg):
            l = l_j[j]
            self.pt_j.append(Spline(l, rcut, grr(pt_g, l, r_g),
                                    r_g, a, zero=1))

        cutoff = 8.0 # ????????
        self.wtLCAO_j = []
        for j, phit_g in enumerate(phit_jg):
            if f_j[j] > 0:
                l = l_j[j]
                self.wtLCAO_j.append(Spline(l, cutoff, grr(phit_g, l, r_g),
                                            r_g, a, zero=True))

        a1_g = 1.0 - 0.5 * (d2gdr2 * dr_g**2)[1:gcut + 1]
        a2_lg = -2.0 * num.ones((lmax + 1, gcut), num.Float)
        x_g = (dr_g[1:gcut + 1] / r_g[1:gcut + 1])**2
        for l in range(1, lmax + 1):
            a2_lg[l] -= l * (l + 1) * x_g
        a3_g = 1.0 + 0.5 * (d2gdr2 * dr_g**2)[1:gcut + 1]

        r_g = r_g[:gcut].copy()
        dr_g = dr_g[:gcut].copy()
        phi_jg = num.array([phi_g[:gcut] for phi_g in phi_jg])
        phit_jg = num.array([phit_g[:gcut] for phit_g in phit_jg])
        nc_g = nc_g[:gcut].copy()
        nct_g = nct_g[:gcut].copy()
        vbar_g = vbar_g[:gcut].copy()

        T_Lqp = num.zeros((Lmax, nq, np), num.Float)
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL_i:
            for j2, l2, L2 in jlL_i[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) / 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) / 2
                T_Lqp[:, q, p] = G_L1L2L[L1, L2, :Lmax]
                p += 1
            i1 += 1

        GAUSS = False
        g_lg = num.zeros((lmax + 1, gcut), num.Float)
        if GAUSS:
            g_lg[0] = 4 / rcgauss**3 / sqrt(pi) * num.exp(-(r_g / rcgauss)**2)
            for l in range(1, lmax + 1):
                g_lg[l] = 2.0 / (2 * l + 1) / rcgauss**2 * r_g * g_lg[l - 1]
        else:
            x_g = r_g / rcut
            s_g = num.zeros(gcut, num.Float)
            for i in range(4):
                s_g += a_i[i] * x_g**i
            for l in range(lmax + 1):
                g_lg[l] = c_l[l] / rcut**(3 + l) * x_g**l * s_g 
                
        for l in range(lmax + 1):
            g_lg[l] /= num.dot(r_g**(l + 2) * dr_g, g_lg[l])

        n_qg = num.zeros((nq, gcut), num.Float)
        nt_qg = num.zeros((nq, gcut), num.Float)
        q = 0
        for j1 in range(nj):
            for j2 in range(j1, nj):
                n_qg[q] = phi_jg[j1] * phi_jg[j2]
                nt_qg[q] = phit_jg[j1] * phit_jg[j2]
                q += 1

        Delta_lq = num.zeros((lmax + 1, nq), num.Float)
        for l in range(lmax + 1):
            Delta_lq[l] = num.dot(n_qg - nt_qg, r_g**(2 + l) * dr_g)
            
        self.Delta_pL = num.zeros((np, Lmax), num.Float)
        for l in range(lmax + 1):
            L = l**2
            delta_p = num.dot(Delta_lq[l], T_Lqp[L])
            for m in range(2 * l + 1):
                self.Delta_pL[:, L + m] = delta_p

        Delta = num.dot(nc_g - nct_g, r_g**2 * dr_g) - Z / sqrt(4 * pi)
        self.Delta0 = Delta

        H = Hartree(a1_g, a2_lg, a3_g, r_g, dr_g).solve
        
        wnc_g = H(nc_g, l=0)
        wnct_g = H(nct_g, l=0)

        wg_lg = [H(g_lg[l], l) for l in range(lmax + 1)]

        wn_lqg = [num.array([H(n_qg[q], l) for q in range(nq)])
                  for l in range(lmax + 1)]
        wnt_lqg = [num.array([H(nt_qg[q], l) for q in range(nq)])
                   for l in range(lmax + 1)]

        rdr_g = r_g * dr_g
        dv_g = r_g * rdr_g
        A = 0.5 * num.dot(nc_g, wnc_g)
        A -= sqrt(4 * pi) * Z * num.dot(rdr_g, nc_g)
        mct_g = nct_g + Delta * g_lg[0]
        wmct_g = wnct_g + Delta * wg_lg[0]
        A -= 0.5 * num.dot(mct_g, wmct_g)
        self.M = A
        AB = -num.dot(dv_g * nct_g, vbar_g)
        self.MB = AB
        
        A_q = 0.5 * (num.dot(wn_lqg[0], nc_g)
                     + num.dot(n_qg, wnc_g))
        A_q -= sqrt(4 * pi) * Z * num.dot(n_qg, rdr_g)

        A_q -= 0.5 * (num.dot(wnt_lqg[0], mct_g)
                     + num.dot(nt_qg, wmct_g))
        A_q -= 0.5 * (num.dot(mct_g, wg_lg[0])
                      + num.dot(g_lg[0], wmct_g)) * Delta_lq[0]
        self.M_p = num.dot(A_q, T_Lqp[0])
        
        AB_q = -num.dot(nt_qg, dv_g * vbar_g)
        self.MB_p = num.dot(AB_q, T_Lqp[0])

        A_lq1q2 = []
        for l in range(lmax + 1):
            A_q1q2 = 0.5 * num.dot(n_qg, num.transpose(wn_lqg[l]))
            A_q1q2 -= 0.5 * num.dot(nt_qg, num.transpose(wnt_lqg[l]))
            A_q1q2 -= 0.5 * num.outerproduct(Delta_lq[l],
                                            num.dot(wnt_lqg[l], g_lg[l]))
            A_q1q2 -= 0.5 * num.outerproduct(num.dot(nt_qg, wg_lg[l]),
                                            Delta_lq[l])
            A_q1q2 -= 0.5 * num.dot(g_lg[l], wg_lg[l]) * \
                      num.outerproduct(Delta_lq[l], Delta_lq[l])
            A_lq1q2.append(A_q1q2)

        self.M_pp = num.zeros((np, np), num.Float)
        L = 0
        for l in range(lmax + 1):
            for m in range(2 * l + 1):
                self.M_pp += num.dot(num.transpose(T_Lqp[L]),
                                    num.dot(A_lq1q2[l], T_Lqp[L]))
                L += 1

##        print A_lq1q2
##        print T_Lqp
##        print self.M, self.M_p, self.M_pp
        # Make a radial grid descriptor:
        rgd = RadialGridDescriptor(r_g, dr_g)

        xc = XCOperator(xcfunc, rgd, nspins)

        self.xc = XCAtom(xc,
                         [grr(phi_g, l_j[j], r_g)
                          for j, phi_g in enumerate(phi_jg)],
                         [grr(phit_g, l_j[j], r_g)
                          for j, phit_g in enumerate(phit_jg)],
                         nc_g / sqrt(4 * pi), nct_g / sqrt(4 * pi),
                         rgd, [(j, l_j[j]) for j in range(nj)],
                         2 * lmax1, e_xc)

        self.rcut = rcut

        # Dont forget to change the onsite interaction energy for soft = 0 XXX
        if softgauss:
            if symbol == 'Cu':
                rcut2 = self.rcut + 2.4
            else:
                rcut2 = self.rcut + 1.4
        else:
            rcut2 = self.rcut

        self.rcut2 = rcut2
        
        if xcname != self.xcname:
            raise RuntimeError, 'Not the correct XC-functional!'
        
        # Use atomic all-electron energies as reference:
        self.Kc = e_kinetic_core - e_kinetic
        self.M -= e_electrostatic
        self.E = e_total

        self.O_ii = sqrt(4.0 * pi) * unpack(self.Delta_pL[:, 0].copy())

        K_q = []
        for j1 in range(nj):
            for j2 in range(j1, nj):
                K_q.append(e_kin_j1j2[j1, j2])
        self.K_p = sqrt(4 * pi) * num.dot(K_q, T_Lqp[0])
        
        self.lmax = lmax

        r = 0.01 * rcut2 * num.arange(101, typecode=num.Float)
        gc = int(100 * rcut / rcut2) + 1
        alpha = rcgauss**-2
        self.alpha = alpha
        if softgauss:
            assert lmax <= 2
            alpha2 = 22.0 / rcut2**2

            if GAUSS:
                vt0 = 4 * pi * (num.array([erf(x) for x in sqrt(alpha) * r]) -
                                num.array([erf(x) for x in sqrt(alpha2) * r]))
                vt0[0] = 8 * sqrt(pi) * (sqrt(alpha) - sqrt(alpha2))
                vt0[1:] /= r[1:]
            else:
                vt0 = -4 * pi * num.array([erf(y) for y in sqrt(alpha2) * r])
                vt0[1:] /= r[1:]
                vt0[0] = -8 * sqrt(pi) * sqrt(alpha2)
                vt00 = vt0.copy()
                x = (r / rcut)[:gc]
                for j in range(6):
                    vt0[:gc] += b_lj[0, j] / rcut * x**j
                vt0[gc:] += 4 * pi / r[gc:]

##             f = open('tmp', 'w')
##             for R, V in zip(r, vt0):
##                 s = a_i[0]+a_i[2]*(R/rcut)**2+a_i[3]*(R/rcut)**3
##                 if R > rcut:
##                     s = 0.0
## ##                print >> f, R, V, 4/sqrt(pi)*(exp(-alpha2*R**2)*alpha2**1.5)
##                 print >> f, R, V, (s * c_l[0] / rcut**3 - 0.000000000000000*
##                                    4/sqrt(pi)*exp(-alpha2*R**2)*alpha2**1.5)
##             print alpha, alpha2
##             print 4/sqrt(pi) *alpha**1.5
##             print 4/sqrt(pi) *alpha2**1.5
## #            stop
            vt_l = [vt0]
            if lmax >= 1:
                arg = num.clip(alpha2 * r**2, 0.0, 700.0)
                e2 = num.exp(-arg)
                if GAUSS:
                    arg = num.clip(alpha * r**2, 0.0, 700.0)
                    e = num.exp(-arg)
                    vt1 = vt0 / 3 - 8 * sqrt(pi) / 3 * (sqrt(alpha) * e -
                                                        sqrt(alpha2) * e2)
                    vt1[0] = 16 * sqrt(pi) / 9 * (alpha**1.5 - alpha2**1.5)
                    vt1[1:] /= r[1:]**2
                else:
                    vt1 = vt00 / 3 + 8 * sqrt(pi) / 3 * sqrt(alpha2) * e2
                    vt1[1:] /= r[1:]**2
                    vt1[0] = -16 * sqrt(pi) / 9 * alpha2**1.5
                    for j in range(6):
                        vt1[:gc] += b_lj[1, j] / rcut**3 * x**j
                    vt1[gc:] += 4 * pi / 3 / r[gc:]**3
                vt_l.append(vt1)
                if lmax >= 2:
                    if GAUSS:
                        vt2 = vt0 / 5 - 8 * sqrt(pi) / 5 * \
                              (sqrt(alpha) * (1 + 2 * alpha * r**2 / 3) * e -
                               sqrt(alpha2) * (1 + 2 * alpha2 * r**2 / 3) * e2)
                        vt2[0] = 32 * sqrt(pi) / 75 * (alpha**2.5 -
                                                       alpha2**2.5)
                        vt2[1:] /= r[1:]**4
                    else:
                        vt2 = vt00 / 5 + 8 * sqrt(pi) / 5 * \
                              sqrt(alpha2) * (1 + 2 * alpha2 * r**2 / 3) * e2 
                        vt2[1:] /= r[1:]**4
                        vt2[0] = -32 * sqrt(pi) / 75 * alpha2**2.5
                        for j in range(6):
                            vt2[:gc] += b_lj[2, j] / rcut**5 * x**j
                        vt2[gc:] += 4 * pi / 5 / r[gc:]**5
                    vt_l.append(vt2)

            self.Deltav_l = []
            for l in range(lmax + 1):
                vtl = vt_l[l]
                vtl[-1] = 0.0
                self.Deltav_l.append(Spline(l, rcut2, vtl))

        else:
            alpha2 = alpha
            self.Deltav_l = [Spline(l, rcut2, 0 * r)
                             for l in range(lmax + 1)]

        self.alpha2 = alpha2

        if GAUSS or softgauss:
            d_l = [fac[l] * 2**(2 * l + 2) / fac[2 * l + 1] / sqrt(pi)
                   for l in range(3)]
            g = alpha2**1.5 * num.exp(-alpha2 * r**2)
            g[-1] = 0.0
            self.gt_l = [Spline(l, rcut2, d_l[l] * alpha2**l * g)
                         for l in range(lmax + 1)]
        else:
            x = r / rcut
            s = num.zeros(101, num.Float)
            for i in range(4):
                s += a_i[i] * x**i
            s[gc:] = 0.0
            self.gt_l = [Spline(l, rcut2,
                                c_l[l] / rcut**(3 + l) * x**l * s)
                         for l in range(lmax + 1)]

        # Construct atomic density matrix for the ground state (to be
        # used for testing):
        self.D_sp = num.zeros((1, np), num.Float)
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL_i:
            occ = f_j[j1] / (2.0 * l1 + 1)
            for j2, l2, L2 in jlL_i[i1:]:
                if j1 == j2 and L1 == L2:
                    self.D_sp[0, p] = occ
                p += 1
            i1 += 1

    def print_info(self, out):
        pass
    """
        print >> out, symbol + ':'
        print >> out, '  name   :', names[Z]
        print >> out, '  Z      :', Z
        print >> out, '  file   :', filename
        print >> out, '  cutoffs: %5.3f (%4.2f Bohr), (%5.3f, %d)' \
              % (a0 * rcut, rcut, a0 * rcut2, lmax)
#        print >> out, '  alphas :', self.alpha, self.alpha2
        print >> out, '  valence states:'
        for n, l, f, eps in zip(n_j, l_j, f_j, eps_j):
            if f > 0:
                print >> out, '    %d%s(%d) %7.3f' % (n, 'spd'[l], f, Ha * eps)
            else:
                print >> out, '     %s    %7.3f' % ('spd'[l], Ha * eps)
        print >> out"""

    def calculate_rotations(self, R_slm1m2):
        nsym = len(R_slm1m2)
        self.R_sii = num.zeros((nsym, self.ni, self.ni), num.Float)
        i1 = 0
        for l in self.l_j:
            i2 = i1 + 2 * l + 1
            for s, R_lm1m2 in enumerate(R_slm1m2):
                self.R_sii[s, i1:i2, i1:i2] = R_slm1m2[s][l]
            i1 = i2

    def symmetrize(self, a, D_aii, map_sa):
        D_ii = num.zeros((self.ni, self.ni), num.Float)
        for s, R_ii in enumerate(self.R_sii):
            D_ii += num.dot(R_ii, num.dot(D_aii[map_sa[s][a]],
                                              num.transpose(R_ii)))
        return D_ii / len(map_sa)

    # Get rid of all these methods: XXXXXXXXXXXXXXXXXXXXXX
    def get_smooth_core_density(self):
        return self.nct

    def get_number_of_atomic_orbitals(self):
        return self.niAO

    def get_number_of_partial_waves(self):
        return self.ni
    
    def get_number_of_derivatives(self):
        return self.nk
    
    def get_recommended_grid_spacing(self):
        return 0.2 # self.h ???  XXXXXXXXXXXXX

    def get_projectors(self):
        return self.pt_j

    def get_atomic_orbitals(self):
        return self.wtLCAO_j

    def delete_atomic_orbitals(self):
        del self.wtLCAO_j
        del self.vHt  # XXXXXXXXXXX

    def get_atomic_hartree_potential(self):
        return self.vHt
    
    def get_shape_function(self):
        return self.gt_l
    
    def get_potential(self):
        return self.Deltav_l

    def get_localized_potential(self):
        return self.vbar

    def get_number_of_valence_electrons(self):
        return self.Nv
