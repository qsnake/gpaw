# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
from math import log, pi, sqrt
import sys

import Numeric as num
from ASE.ChemicalElements.name import names

from gpaw.read_setup import PAWXMLParser
from gpaw.read_basis import BasisSetXMLParser
from gpaw.gaunt import gaunt as G_LLL
from gpaw.spline import Spline
from gpaw.grid_descriptor import RadialGridDescriptor
from gpaw.utilities import unpack, erf, fac, hartree, pack2
from gpaw.xc_correction import XCCorrection
from gpaw.xc_functional import XCRadialGrid
from gpaw.kli import XCKLICorrection, XCGLLBCorrection
from LinearAlgebra import inverse as inv

def create_setup(symbol, xcfunc, lmax=0, nspins=1, type='paw', basis=None):
    if type == 'ae':
        from gpaw.ae import AllElectronSetup
        return AllElectronSetup(symbol, xcfunc, nspins)

    if type == 'hgh':
        from gpaw.hgh import HGHSetup
        return HGHSetup(symbol, xcfunc, nspins)

    if type == 'hgh.sc':
        from gpaw.hgh import HGHSetup
        return HGHSetup(symbol, xcfunc, nspins, semicore=True)

    return Setup(symbol, xcfunc, lmax, nspins, type, basis)


class Setup:
    """Attributes:

    ========== ============================================
    Name       Description
    ========== ============================================
    ``Z``      Charge
    ``l_j``    List of angular momentum l values
    ``I4_iip`` Integrals over 4 all electron wave functions
    ========== ============================================

    """
    def __init__(self, symbol, xcfunc, lmax=0, nspins=1,
                 type='paw', basis=None):
        self.xcfunc = xcfunc
        self.xcname = xcfunc.get_name()
        self.setupname = xcfunc.get_setup_name()
        self.softgauss = False
        xcname = self.xcname
        zero_reference = xcfunc.hybrid > 0

        self.type = type
        if type != 'paw':
            symbol += '.' + type
        self.symbol = symbol

        (Z, Nc, Nv,
         e_total, e_kinetic, e_electrostatic, e_xc,
         e_kinetic_core,
         n_j, l_j, f_j, eps_j, rcut_j, id_j,
         ng, beta,
         nc_g, nct_g, vbar_g, rcgauss,
         phi_jg, phit_jg, pt_jg,
         e_kin_jj, X_p, ExxC,
         tauc_g, tauct_g,
         self.fingerprint,
         filename,
         self.phicorehole_g,
         self.fcorehole,
         eigcorehole,
         ekincorehole,
         core_response) = PAWXMLParser().parse(self.symbol, self.setupname,
                                               zero_reference)

        self.filename = filename

        self.Nv = Nv
        self.Nc = Nc
        self.Z = Z
        self.X_p = X_p
        self.ExxC = ExxC

        nj = len(l_j)
        e_kin_jj.shape = (nj, nj)

        self.n_j = n_j
        self.l_j = l_j
        self.f_j = f_j
        self.eps_j = eps_j

        g = num.arange(ng, typecode=num.Float)
        r_g = beta * g / (ng - g)
        dr_g = beta * ng / (ng - g)**2
        d2gdr2 = -2 * ng * beta / (beta + r_g)**3

        # compute inverse overlap coefficients B_ii
        B_jj = num.zeros((len(l_j),len(l_j)), num.Float)
        for i1 in range(len(l_j)):
            for i2 in range(len(l_j)):
                B_jj[i1][i2] = num.dot( r_g**2 * dr_g, pt_jg[i1] * pt_jg[i2] )

        def delta(i,j):
            #assert type(i) == int
            #assert type(j) == int
            if i == j: return 1
            else: return 0

        size=0
        for l1 in l_j:
            for m1 in range(2 * l1 + 1):
                    size +=1
        self.B_ii = num.zeros((size,size), num.Float)

        i1=0
        for n1, l1 in enumerate(l_j):
            for m1 in range(2 * l1 + 1):
                i2=0
                for n2, l2 in enumerate(l_j):
                    for m2 in range(2 * l2 + 1):
                        self.B_ii[i1][i2] = delta(l1,l2)*delta(m1,m2)*B_jj[n1][n2]
                        i2 +=1
                i1 +=1

        # Find Fourier-filter cutoff radius:
        g = ng - 1
        while pt_jg[0][g] == 0.0:
            g -= 1
        gcutfilter = g + 1
        self.rcutfilter = rcutfilter = r_g[gcutfilter]

        rcutmax = max(rcut_j)
        rcut2 = 2 * rcutmax
        gcutmax = 1 + int(rcutmax * ng / (rcutmax + beta))
        gcut2 = 1 + int(rcut2 * ng / (rcut2 + beta))

        # Find cutoff for core density:
        if Nc == 0:
            self.rcore = rcore = 0.5
        else:
            N = 0.0
            g = ng - 1
            while N < 1e-7:
                N += sqrt(4 * pi) * nc_g[g] * r_g[g]**2 * dr_g[g]
                g -= 1
            self.rcore = rcore = r_g[g]

        ni = 0
        niAO = 0
        i = 0
        j = 0
        jlL_i = []
        for l, n in zip(l_j, n_j):
            if n > 0:
                niAO += 2 * l + 1
            for m in range(2 * l + 1):
                jlL_i.append((j, l, l**2 + m))
                i += 1
            j += 1
        ni = i
        self.ni = ni
        self.niAO = niAO

        np = ni * (ni + 1) // 2
        nq = nj * (nj + 1) // 2

        lcut = max(l_j)
        if 2 * lcut < lmax:
            lcut = (lmax + 1) // 2

        if self.phicorehole_g is not None:
            print "ok"
            self.calculate_oscillator_strengths(r_g, dr_g, phi_jg)

        # Construct splines:
        self.nct = Spline(0, rcore, nct_g, r_g=r_g, beta=beta)
        self.vbar = Spline(0, rcutfilter, vbar_g, r_g=r_g, beta=beta)

        # Construct splines for core kinetic energy density:
        if tauct_g is None:
            tauct_g = num.zeros(ng, num.Float)
        self.tauct = Spline(0, rcore, tauct_g, r_g=r_g, beta=beta)

        # Step function:
        stepf = sqrt(4 * pi) * num.ones(ng)
        stepf[gcutmax:] = 0.0
        self.stepf = Spline(0, rcutfilter, stepf, r_g=r_g, beta=beta)

        self.pt_j = []
        for j in range(nj):
            l = l_j[j]
            self.pt_j.append(Spline(l, rcutfilter, grr(pt_jg[j], l, r_g),
                                    r_g=r_g, beta=beta))

        if basis is None:
            self.create_basis_functions(phit_jg, beta, ng, rcut2, gcut2, r_g)
        else:
            self.read_basis_functions(basis)

        r_g = r_g[:gcut2].copy()
        dr_g = dr_g[:gcut2].copy()
        phi_jg = num.array([phi_g[:gcut2] for phi_g in phi_jg])
        phit_jg = num.array([phit_g[:gcut2] for phit_g in phit_jg])
        nc_g = nc_g[:gcut2].copy()
        nct_g = nct_g[:gcut2].copy()
        vbar_g = vbar_g[:gcut2].copy()
        tauc_g = tauc_g[:gcut2].copy()




        if xcname == 'GLLB':
            core_response = core_response[:gcut2].copy()

        if self.phicorehole_g is not None:
            self.phicorehole_g = self.phicorehole_g[:gcut2].copy()

        Lcut = (2 * lcut + 1)**2
        T_Lqp = num.zeros((Lcut, nq, np), num.Float)
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL_i:
            for j2, l2, L2 in jlL_i[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) // 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) // 2
                T_Lqp[:, q, p] = G_LLL[L1, L2, :Lcut]
                p += 1
            i1 += 1

        g_lg = num.zeros((lmax + 1, gcut2), num.Float)
        g_lg[0] = 4 / rcgauss**3 / sqrt(pi) * num.exp(-(r_g / rcgauss)**2)
        for l in range(1, lmax + 1):
            g_lg[l] = 2.0 / (2 * l + 1) / rcgauss**2 * r_g * g_lg[l - 1]

        for l in range(lmax + 1):
            g_lg[l] /= num.dot(r_g**(l + 2) * dr_g, g_lg[l])

        n_qg = num.zeros((nq, gcut2), num.Float)
        nt_qg = num.zeros((nq, gcut2), num.Float)
        q = 0
        for j1 in range(nj):
            for j2 in range(j1, nj):
                n_qg[q] = phi_jg[j1] * phi_jg[j2]
                nt_qg[q] = phit_jg[j1] * phit_jg[j2]
                q += 1

        Delta_lq = num.zeros((lmax + 1, nq), num.Float)
        for l in range(lmax + 1):
            Delta_lq[l] = num.dot(n_qg - nt_qg, r_g**(2 + l) * dr_g)

        Lmax = (lmax + 1)**2
        self.Delta_pL = num.zeros((np, Lmax), num.Float)
        for l in range(lmax + 1):
            L = l**2
            for m in range(2 * l + 1):
                delta_p = num.dot(Delta_lq[l], T_Lqp[L + m])
                self.Delta_pL[:, L + m] = delta_p

        Delta = num.dot(nc_g - nct_g, r_g**2 * dr_g) - Z / sqrt(4 * pi)
        self.Delta0 = Delta

        def H(n_g, l):
            yrrdr_g = num.zeros(gcut2, num.Float)
            nrdr_g = n_g * r_g * dr_g
            hartree(l, nrdr_g, beta, ng, yrrdr_g)
            yrrdr_g *= r_g * dr_g
            return yrrdr_g

        wnc_g = H(nc_g, l=0)
        wnct_g = H(nct_g, l=0)

        wg_lg = [H(g_lg[l], l) for l in range(lmax + 1)]

        wn_lqg = [num.array([H(n_qg[q], l) for q in range(nq)])
                  for l in range(2 * lcut + 1)]
        wnt_lqg = [num.array([H(nt_qg[q], l) for q in range(nq)])
                   for l in range(2 * lcut + 1)]

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

        A_lqq = []
        for l in range(2 * lcut + 1):
            A_qq = 0.5 * num.dot(n_qg, num.transpose(wn_lqg[l]))
            A_qq -= 0.5 * num.dot(nt_qg, num.transpose(wnt_lqg[l]))
            if l <= lmax:
                A_qq -= 0.5 * num.outerproduct(Delta_lq[l],
                                               num.dot(wnt_lqg[l], g_lg[l]))
                A_qq -= 0.5 * num.outerproduct(num.dot(nt_qg, wg_lg[l]),
                                               Delta_lq[l])
                A_qq -= 0.5 * num.dot(g_lg[l], wg_lg[l]) * \
                        num.outerproduct(Delta_lq[l], Delta_lq[l])
            A_lqq.append(A_qq)

        self.M_pp = num.zeros((np, np), num.Float)
        L = 0
        for l in range(2 * lcut + 1):
            for m in range(2 * l + 1):
                self.M_pp += num.dot(num.transpose(T_Lqp[L]),
                                     num.dot(A_lqq[l], T_Lqp[L]))
                L += 1

        # Make a radial grid descriptor:
        rgd = RadialGridDescriptor(r_g, dr_g)

        if xcfunc.xcname == 'KLI':
            # For debugging purposes, pass a LDA XCCorrection
            # to XCKLICorrection class
            from gpaw.xc_functional import XCFunctional
            xc = XCRadialGrid(XCFunctional('LDAx', nspins), rgd, nspins)
            xc_lda_correction = XCCorrection(
                xc,
                [grr(phi_g, l_j[j], r_g) for j, phi_g in enumerate(phi_jg)],
                [grr(phit_g, l_j[j], r_g) for j, phit_g in enumerate(phit_jg)],
                nc_g / sqrt(4 * pi), nct_g / sqrt(4 * pi),
                rgd, [(j, l_j[j]) for j in range(nj)],
                2 * lcut, e_xc)

            # Create XCKLICorrection with necessary parameters
            self.xc_correction = XCKLICorrection(xcfunc.xc, rgd.r_g, rgd.dr_g,
                                                 beta, ng, nspins,
                                                 self.M_pp, self.X_p,
                                                 self.ExxC,
                                                 [grr(phi_g, l_j[j], r_g) for j, phi_g in enumerate(phi_jg)],
                                                 [grr(phit_g, l_j[j], r_g) for j, phit_g in enumerate(phit_jg)],
                                                 [(j, l_j[j]) for j in range(nj)],
                                                 xc_lda_correction)

        elif xcfunc.xcname == "GLLB":
            self.xc_correction = XCGLLBCorrection(
                xcfunc.xc,
                [grr(phi_g, l_j[j], r_g) for j, phi_g in enumerate(phi_jg)],
                [grr(phit_g, l_j[j], r_g) for j, phit_g in enumerate(phit_jg)],
                nc_g / sqrt(4 * pi), nct_g / sqrt(4 * pi),
                rgd, [(j, l_j[j]) for j in range(nj)],
                2 * lcut, e_xc, core_response)
        elif xcfunc.xcname == "TPSS":
            xc = XCRadialGrid(xcfunc, rgd, nspins)

            self.xc_correction = XCCorrection(
                xc,
                [grr(phi_g, l_j[j], r_g) for j, phi_g in enumerate(phi_jg)],
                [grr(phit_g, l_j[j], r_g) for j, phit_g in enumerate(phit_jg)],
                nc_g / sqrt(4 * pi), nct_g / sqrt(4 * pi),
                rgd, [(j, l_j[j]) for j in range(nj)],
                2 * lcut, e_xc, self.phicorehole_g, self.fcorehole, nspins,tauc_g)
        else:
            xc = XCRadialGrid(xcfunc, rgd, nspins)

            self.xc_correction = XCCorrection(
                xc,
                [grr(phi_g, l_j[j], r_g) for j, phi_g in enumerate(phi_jg)],
                [grr(phit_g, l_j[j], r_g) for j, phit_g in enumerate(phit_jg)],
                nc_g / sqrt(4 * pi), nct_g / sqrt(4 * pi),
                rgd, [(j, l_j[j]) for j in range(nj)],
                2 * lcut, e_xc, self.phicorehole_g, self.fcorehole, nspins)

        if self.softgauss:
            rcutsoft = rcut2####### + 1.4
        else:
            rcutsoft = rcut2

        self.rcutsoft = rcutsoft

        if xcname != self.xcname:
            raise RuntimeError('Not the correct XC-functional!')

        # Use atomic all-electron energies as reference:
        self.Kc = e_kinetic_core - e_kinetic
        self.M -= e_electrostatic
        self.E = e_total

        self.O_ii = sqrt(4.0 * pi) * unpack(self.Delta_pL[:, 0].copy())

        self.Delta_Lii = num.zeros((ni, ni, Lmax), num.Float)
        for L in range(Lmax):
            self.Delta_Lii[:,:,L] = unpack(self.Delta_pL[:, L].copy())

        K_q = []
        for j1 in range(nj):
            for j2 in range(j1, nj):
                K_q.append(e_kin_jj[j1, j2])
        self.K_p = sqrt(4 * pi) * num.dot(K_q, T_Lqp[0])

        self.lmax = lmax

        r = 0.02 * rcutsoft * num.arange(51, typecode=num.Float)
##        r = 0.04 * rcutsoft * num.arange(26, typecode=num.Float)
        alpha = rcgauss**-2
        self.alpha = alpha
        if self.softgauss:
            assert lmax <= 2
            alpha2 = 22.0 / rcutsoft**2
            alpha2 = 15.0 / rcutsoft**2

            vt0 = 4 * pi * (num.array([erf(x) for x in sqrt(alpha) * r]) -
                            num.array([erf(x) for x in sqrt(alpha2) * r]))
            vt0[0] = 8 * sqrt(pi) * (sqrt(alpha) - sqrt(alpha2))
            vt0[1:] /= r[1:]
            vt_l = [vt0]
            if lmax >= 1:
                arg = num.clip(alpha2 * r**2, 0.0, 700.0)
                e2 = num.exp(-arg)
                arg = num.clip(alpha * r**2, 0.0, 700.0)
                e = num.exp(-arg)
                vt1 = vt0 / 3 - 8 * sqrt(pi) / 3 * (sqrt(alpha) * e -
                                                    sqrt(alpha2) * e2)
                vt1[0] = 16 * sqrt(pi) / 9 * (alpha**1.5 - alpha2**1.5)
                vt1[1:] /= r[1:]**2
                vt_l.append(vt1)
                if lmax >= 2:
                    vt2 = vt0 / 5 - 8 * sqrt(pi) / 5 * \
                          (sqrt(alpha) * (1 + 2 * alpha * r**2 / 3) * e -
                           sqrt(alpha2) * (1 + 2 * alpha2 * r**2 / 3) * e2)
                    vt2[0] = 32 * sqrt(pi) / 75 * (alpha**2.5 -
                                                   alpha2**2.5)
                    vt2[1:] /= r[1:]**4
                    vt_l.append(vt2)

            self.vhat_l = []
            for l in range(lmax + 1):
                vtl = vt_l[l]
                vtl[-1] = 0.0
                self.vhat_l.append(Spline(l, rcutsoft, vtl))

        else:
            alpha2 = alpha
            self.vhat_l = None
            #self.vhat_l = [Spline(l, rcutsoft, 0 * r)
            #                 for l in range(lmax + 1)]

        self.alpha2 = alpha2

        d_l = [fac[l] * 2**(2 * l + 2) / sqrt(pi) / fac[2 * l + 1]
               for l in range(lmax + 1)]
        g = alpha2**1.5 * num.exp(-alpha2 * r**2)
        g[-1] = 0.0
        self.ghat_l = [Spline(l, rcutsoft, d_l[l] * alpha2**l * g)
                     for l in range(lmax + 1)]

        self.rcutcomp = sqrt(10) * rcgauss
        self.rcut_j = rcut_j


        # compute inverse overlap coefficients C_ii
        self.C_ii = -num.dot(self.O_ii, inv(num.identity(size) + num.dot(self.B_ii, self.O_ii)))

    def create_basis_functions(self, phit_jg, beta, ng, rcut2, gcut2, r_g):
        # Cutoff for atomic orbitals used for initial guess:
        rcut3 = 8.0  # XXXXX Should depend on the size of the atom!
        gcut3 = 1 + int(rcut3 * ng / (rcut3 + beta))

        # We cut off the wave functions smoothly at rcut3 by the
        # following replacement:
        #
        #            /
        #           | f(r),                                   r < rcut2
        #  f(r) <- <  f(r) - a(r) f(rcut3) - b(r) f'(rcut3),  rcut2 < r < rcut3
        #           | 0,                                      r > rcut3
        #            \
        #
        # where a(r) and b(r) are 4. order polynomials:
        #
        #  a(rcut2) = 0,  a'(rcut2) = 0,  a''(rcut2) = 0,
        #  a(rcut3) = 1, a'(rcut3) = 0
        #  b(rcut2) = 0, b'(rcut2) = 0, b''(rcut2) = 0,
        #  b(rcut3) = 0, b'(rcut3) = 1
        #
        x = (r_g[gcut2:gcut3] - rcut2) / (rcut3 - rcut2)
        a_g = 4 * x**3 * (1 - 0.75 * x)
        b_g = x**3 * (x - 1) * (rcut3 - rcut2)

        self.phit_j = []
        for j, phit_g in enumerate(phit_jg):
            if self.n_j[j] > 0:
                l = self.l_j[j]
                phit = phit_g[gcut3]
                dphitdr = ((phit - phit_g[gcut3 - 1]) /
                           (r_g[gcut3] - r_g[gcut3 - 1]))
                phit_g[gcut2:gcut3] -= phit * a_g + dphitdr * b_g
                phit_g[gcut3:] = 0.0
                self.phit_j.append(Spline(l, rcut3,
                                          grr(phit_g, l, r_g),
                                          r_g=r_g, beta=beta))

    def read_basis_functions(self, basis_name):
        parser = BasisSetXMLParser()
        (l_j, rc, phit_jg, filename) = parser.parse(self.symbol, basis_name)

        self.phit_j = []
        for l, phit_g in zip(l_j, phit_jg):
            self.phit_j.append(Spline(l, rc, phit_g))

    def print_info(self, text):
        if self.phicorehole_g is None:
            text(self.symbol + '-setup:')
        else:
            text('%s-setup (%.1f core hole):' % (self.symbol, self.fcorehole))
        text('  name   :', names[self.Z])
        text('  Z      :', self.Z)
        text('  valence:', self.Nv)
        if self.phicorehole_g is None:
            text('  core   : %d' % self.Nc)
        else:
            text('  core   : %.1f' % self.Nc)
        text('  charge :', self.Z - self.Nv - self.Nc)
        text('  file   :', self.filename)
        text(('  cutoffs: %4.2f(comp), %4.2f(filt), %4.2f(core) Bohr,'
              ' lmax=%d' % (self.rcutcomp, self.rcutfilter,
                            self.rcore, self.lmax)))
        text('  valence states:')
        j = 0
        for n, l, f, eps in zip(self.n_j, self.l_j, self.f_j, self.eps_j):
            if n > 0:
                f = '(%d)' % f
                text('    %d%s%-4s %7.3f Ha   %4.2f Bohr' % (
                    n, 'spdf'[l], f, eps, self.rcut_j[j]))
            else:
                text('    *%s     %7.3f Ha   %4.2f Bohr' % (
                    'spdf'[l], eps, self.rcut_j[j]))
            j += 1

        text()

    def calculate_rotations(self, R_slmm):
        nsym = len(R_slmm)
        self.R_sii = num.zeros((nsym, self.ni, self.ni), num.Float)
        i1 = 0
        for l in self.l_j:
            i2 = i1 + 2 * l + 1
            for s, R_lmm in enumerate(R_slmm):
                self.R_sii[s, i1:i2, i1:i2] = R_lmm[l]
            i1 = i2

    def symmetrize(self, a, D_aii, map_sa):
        D_ii = num.zeros((self.ni, self.ni), num.Float)
        for s, R_ii in enumerate(self.R_sii):
            D_ii += num.dot(R_ii, num.dot(D_aii[map_sa[s][a]],
                                              num.transpose(R_ii)))
        return D_ii / len(map_sa)

    def get_partial_waves(self):
        """Return spline representation of partial waves and densities."""

        # Load setup data from XML file:
        (Z, Nc, Nv,
         e_total, e_kinetic, e_electrostatic, e_xc,
         e_kinetic_core,
         n_j, l_j, f_j, eps_j, rcut_j, id_j,
         ng, beta,
         nc_g, nct_g, vbar_g, rcgauss,
         phi_jg, phit_jg, pt_jg,
         e_kin_jj, X_p, ExxC,
         tauc_g, tauct_g,
         fingerprint,
         filename,
         core_hole_state,
         fcorehole,
         core_hole_e,
         core_hole_e_kin,
         core_response) = PAWXMLParser().parse(self.symbol, self.setupname)

        # cutoffs
        nj = len(l_j)
        rcut2 = 2 * max(rcut_j)
        gcut2 = 1 + int(rcut2 * ng / (rcut2 + beta))

        # radial grid
        g = num.arange(ng, typecode=num.Float)
        r_g = beta * g / (ng - g)

        # Construct splines:
        nc_g[gcut2:] = nc_g[gcut2:] = 0.0
        nc = Spline(0, rcut2, nc_g, r_g=r_g, beta=beta, points=1000)
        nct = Spline(0, rcut2, nct_g, r_g=r_g, beta=beta, points=1000)
        if tauc_g is None:
            tauc_g = num.zeros(nct_g.shape,num.Float)
            tauct_g = tauc_g
        tauc = Spline(0, rcut2, tauc_g, r_g=r_g, beta=beta, points=1000)
        tauct = Spline(0, rcut2, tauct_g, r_g=r_g, beta=beta, points=1000)
        phi_j = []
        phit_j = []
        for j, (phi_g, phit_g) in enumerate(zip(phi_jg, phit_jg)):
            l = l_j[j]
            phi_g[gcut2:] = phit_g[gcut2:] = 0.0
            phi_j.append(Spline(l, rcut2, grr(phi_g, l, r_g), r_g=r_g,
                                 beta=beta, points=100))
            phit_j.append(Spline(l, rcut2, grr(phit_g, l, r_g), r_g=r_g,
                                 beta=beta, points=100))
        return phi_j, phit_j, nc, nct, tauc, tauct

    def calculate_oscillator_strengths(self, r_g, dr_g, phi_jg):
        self.A_ci = num.zeros((3, self.ni), num.Float)
        nj = len(phi_jg)
        i = 0
        for j in range(nj):
            l = self.l_j[j]
            if l == 1:
                a = num.dot(r_g**3 * dr_g, phi_jg[j] * self.phicorehole_g)

                for m in range(3):
                    c = (m + 1) % 3
                    self.A_ci[c, i] = a
                    i += 1
            else:
                i += 2 * l + 1
        assert i == self.ni


    def four_phi_integrals(self):
        """Calculate four-phi integral.

        Calculate the integral over the product of four all electron
        functions in the augmentation sphere, i.e.::

          /
          | d vr  ( phi_i1 phi_i2 phi_i3 phi_i4
          /         - phit_i1 phit_i2 phit_i3 phit_i4 ),

        where phi_i1 is a all electron function and phit_i1 is its
        smooth partner.
        """

        if hasattr(self, 'I4_iip'):
##            print "already done"
            return # job already done

        # Load setup data from XML file:
        (Z, Nc, Nv,
         e_total, e_kinetic, e_electrostatic, e_xc,
         e_kinetic_core,
         n_j, l_j, f_j, eps_j, rcut_j, id_j,
         ng, beta,
         nc_g, nct_g, vbar_g, rcgauss,
         phi_jg, phit_jg, pt_jg,
         e_kin_jj, X_p, ExxC,
         tauc_g, tauct_g,
         fingerprint,
         filename,
         core_hole_state,
         fcorehole,
         core_hole_e,
         core_hole_e_kin,
         core_response) = PAWXMLParser().parse(self.symbol, self.setupname)

        # radial grid
        g = num.arange(ng, typecode=num.Float)
        r_g = beta * g / (ng - g)
        dr_g = beta * ng / (ng - g)**2

        # compute radial parts
        nj = len(self.l_j)
        R_llll = num.zeros((nj,nj,nj,nj), num.Float)
        for i1 in range(nj):
            for i2 in range(nj):
                for i3 in range(nj):
                    for i4 in range(nj):
                        R_llll[i1,i2,i3,i4] = num.dot( r_g**2 * dr_g,
                                                       phi_jg[i1]*phi_jg[i2]*
                                                       phi_jg[i3]*phi_jg[i4] -
                                                       phit_jg[i1]*phit_jg[i2]*
                                                       phit_jg[i3]*phit_jg[i4])

        # prepare for angular parts
        L_i = []
        j_i = []
        for j,l1 in enumerate(self.l_j):
            for m1 in range(2 * l1 + 1):
                L_i.append(l1**2 + m1)
                j_i.append(j)
        ni = len(L_i)
        np = ni * (ni + 1) // 2 # length for packing
        # j_i is the list of j values
        # L_i is the list of L (=l**2+m for 0<=m<l) values
        # https://wiki.fysik.dtu.dk/gpaw/Overview#naming-convention-for-arrays

        # calculate the integrals
        I4_iip = num.empty((ni,ni,np),num.Float)
        I = num.empty((ni,ni),num.Float)
        for i1 in range(ni):
            L1 = L_i[i1]
            j1 = j_i[i1]
            for i2 in range(ni):
                L2 = L_i[i2]
                j2 = j_i[i2]
                for i3 in range(ni):
                    L3 = L_i[i3]
                    j3 = j_i[i3]
                    for i4 in range(ni):
                        L4 = L_i[i4]
                        j4 = j_i[i4]
                        I[i3,i4] = num.dot( G_LLL[L1,L2],
                                            G_LLL[L3,L4] ) *\
                                    R_llll[j1,j2,j3,j4]
                I4_iip[i1,i2,:] = pack2(I)

        self.I4_iip = I4_iip

def grr(phi_g, l, r_g):
    w_g = phi_g.copy()
    if l > 0:
        w_g[1:] /= r_g[1:]**l
        w1, w2 = w_g[1:3]
        r0, r1, r2 = r_g[0:3]
        w_g[0] = w2 + (w1 - w2) * (r0 - r2) / (r1 - r2)
    return w_g


if __name__ == '__main__':
    print """\
You are using the wrong setup.py script!  This setup.py defines a
Setup class used to hold the atomic data needed for a specific atom.
For building the GPAW code you must use the setup.py distutils script
at the root of the code tree.  Just do "cd .." and you will be at the
right place."""
