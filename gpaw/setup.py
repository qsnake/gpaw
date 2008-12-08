# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
from math import log, pi, sqrt
import sys

import numpy as npy
from ase.data import atomic_names

from gpaw.setup_data import SetupData
from gpaw.basis_data import Basis
from gpaw.gaunt import gaunt as G_LLL
from gpaw.spline import Spline
from gpaw.grid_descriptor import RadialGridDescriptor
from gpaw.utilities import unpack, erf, fac, hartree, pack2, divrl
from gpaw.xc_correction import XCCorrection
from gpaw.xc_functional import XCRadialGrid


def create_setup(symbol, xcfunc, lmax=0, nspins=1, type='paw', basis=None,
                 setupdata=None):
    if type == 'ae':
        from gpaw.ae import AllElectronSetup
        return AllElectronSetup(symbol, xcfunc, nspins)

    if type == 'hgh':
        from gpaw.hgh import HGHSetup
        return HGHSetup(symbol, xcfunc, nspins, basis=basis)

    if type == 'hgh.sc':
        from gpaw.hgh import HGHSetup
        return HGHSetup(symbol, xcfunc, nspins, semicore=True, basis=basis)

    return Setup(symbol, xcfunc, lmax, nspins, type, basis, setupdata)


class Setup:
    """Attributes:

    ========== =====================================================
    Name       Description
    ========== =====================================================
    ``Z``      Charge
    ``type``   Type-name of setup (eg. 'paw')
    ``symbol`` Chemical element label (eg. 'Mg')
    ``xcname`` Name of xc
    ``data``   Container class for information on the the atom, eg.
               Nc, Nv, n_j, l_j, f_j, eps_j, rcut_j.
               It defines the radial grid by ng and beta, from which
               r_g = beta * arange(ng) / (ng - arange(ng)).
               It stores pt_jg, phit_jg, phi_jg, vbar_g
    ========== =====================================================


    Attributes for making PAW corrections

    ============= ==========================================================
    Name          Description
    ============= ==========================================================
    ``Delta0``    Constant in compensation charge expansion coeff.
    ``Delta_Lii`` Linear term in compensation charge expansion coeff.
    ``O_ii``      Overlap metric
    ``B_ii``      Projector function overlaps B_ii = <pt_i | pt_i>
    ``C_ii``      Inverse overlap coefficients
    ``E``         Reference total energy of atom
    ``M``         Constant correction to Coulomb energy
    ``M_p``       Linear correction to Coulomb energy
    ``M_pp``      2nd order correction to Coulomb energy and Exx energy
    ``Kc``        Core kinetic energy
    ``K_p``       Linear correction to kinetic energy
    ``Xc``        Core Exx energy (Actually called ExxC ... rename)
    ``X_p``       Linear correction to Exx energy
    ``Mb``        Constant correction due to vbar potential
    ``Mb_p``      Linear correction due to vbar potential
    ``dEH0``      Constant correction due to average electrostatic potential
    ``dEH_p``     Linear correction due to average electrostatic potential
    ``I4_iip``    Correction to integrals over 4 all electron wave functions
    ============= ==========================================================

    It also has the attribute ``xc_correction`` which is an XCCorrection class
    instance capable of calculating the corrections due to the xc functional.


    Splines:

    ========== ============================================
    Name       Description
    ========== ============================================
    ``pt_j``   Projector functions
    ``phit_j`` Pseudo partial waves
    ``stepf``  Stepfunction
    ``vbar``   vbar potential
    ``nct``    Pseudo core density
    ``tauct``  Pseudo core kinetic energy density
    ========== ============================================
    """
    def __init__(self, symbol, xcfunc, lmax=0, nspins=1,
                 type='paw', basis=None, setupdata=None):
        actual_symbol = symbol
        self.type = type

        zero_reference = xcfunc.hybrid > 0
        self.HubU = None

        if type != 'paw':
            symbol += '.' + type
        self.symbol = symbol

        self.xcname = xcfunc.get_name()

        if setupdata is None:
            data = SetupData(actual_symbol, xcfunc.get_setup_name(),
                             type, zero_reference, True)
        else:
            data = setupdata
        self.data = data

        # Copy variables from data object to self.
        # Some variables should probably exist only on the data object, but
        # presently we copy them to keep existing code from failing.

        self.Nc = data.Nc
        self.Nv = data.Nv
        self.Z = data.Z
        l_j = self.l_j = data.l_j
        n_j = self.n_j = data.n_j
        self.f_j = data.f_j
        self.eps_j = data.eps_j
        nj = self.nj = len(l_j)
        ng = self.ng = data.ng
        beta = self.beta = data.beta
        self.softgauss = data.softgauss
        rcgauss = self.rcgauss = data.rcgauss
        rcut_j = self.rcut_j = data.rcut_j

        self.fcorehole = data.fcorehole

        self.ExxC = data.ExxC
        self.X_p = data.X_p

        pt_jg = data.pt_jg
        phit_jg = data.phit_jg
        phi_jg = data.phi_jg

        self.fingerprint = data.fingerprint
        self.filename = data.filename

        g = npy.arange(ng, dtype=float)
        r_g = beta * g / (ng - g)
        dr_g = beta * ng / (ng - g)**2
        d2gdr2 = -2 * ng * beta / (beta + r_g)**3

        self.lmax = lmax

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
        self.gcut2 = gcut2

        ni = 0
        i = 0
        j = 0
        jlL_i = []
        for l, n in zip(l_j, n_j):
            for m in range(2 * l + 1):
                jlL_i.append((j, l, l**2 + m))
                i += 1
            j += 1
        ni = i
        self.ni = ni

        np = ni * (ni + 1) // 2
        self.nq = nq = nj * (nj + 1) // 2

        self.lcut = lcut = max(l_j)
        if 2 * lcut < lmax:
            lcut = (lmax + 1) // 2

        if data.phicorehole_g is not None:
            print "ok"
            self.calculate_oscillator_strengths(r_g, dr_g, phi_jg)

        # Compute projector function overlaps B_ii = <pt_i | pt_i>
        B_jj = npy.zeros((nj, nj))
        for j1, pt1_g in enumerate(pt_jg):
            for j2, pt2_g in enumerate(pt_jg):
                B_jj[j1, j2] = npy.dot(r_g**2 * dr_g, pt1_g * pt2_g)
        self.B_ii = npy.zeros((ni, ni))
        i1 = 0
        for j1, l1 in enumerate(l_j):
            for m1 in range(2 * l1 + 1):
                i2 = 0
                for j2, l2 in enumerate(l_j):
                    for m2 in range(2 * l2 + 1):
                        if l1 == l2 and m1 == m2:
                            self.B_ii[i1, i2] = B_jj[j1, j2]
                        i2 += 1
                i1 += 1
        del B_jj

        # Construct splines:
        self.vbar = Spline(0, rcutfilter, data.vbar_g, r_g, beta)

        rcore, nc_g, nct_g, nct = self.construct_core_densities(r_g, dr_g,
                                                                beta, data)
        self.rcore = rcore
        self.nct = nct

        # Construct splines for core kinetic energy density:
        tauct_g = data.tauct_g
        if tauct_g is None:
            tauct_g = npy.zeros(ng)
        self.tauct = Spline(0, self.rcore, tauct_g, r_g, beta)

        # Step function:
        stepf = sqrt(4 * pi) * npy.ones(ng)
        stepf[gcutmax:] = 0.0
        self.stepf = Spline(0, rcutfilter, stepf, r_g, beta)

        self.pt_j = []
        for j in range(nj):
            l = l_j[j]
            self.pt_j.append(Spline(l, rcutfilter, pt_jg[j], r_g, beta))

        if basis is None:
            phit_j = self.create_basis_functions(phit_jg, beta, ng, rcut2,
                                                 gcut2, r_g)
        else:
            phit_j = self.read_basis_functions(basis)
        self.phit_j = phit_j

        self.niAO = 0
        for phit in self.phit_j:
            l = phit.get_angular_momentum_number()
            self.niAO += 2 * l + 1

        r_g = r_g[:gcut2].copy()
        dr_g = dr_g[:gcut2].copy()
        phi_jg = npy.array([phi_g[:gcut2].copy() for phi_g in phi_jg])
        phit_jg = npy.array([phit_g[:gcut2].copy() for phit_g in phit_jg])
        self.nc_g = nc_g = nc_g[:gcut2].copy()
        self.nct_g = nct_g = nct_g[:gcut2].copy()
        vbar_g = data.vbar_g[:gcut2].copy()
        tauc_g = data.tauc_g[:gcut2].copy()

        extra_xc_data = dict(data.extra_xc_data)
        # Cut down the GLLB related extra data
        for key, item in extra_xc_data.iteritems():
            if len(item) > 1:
                extra_xc_data[key] = item[:gcut2].copy()
        self.extra_xc_data = extra_xc_data

        self.phicorehole_g = data.phicorehole_g
        if self.phicorehole_g is not None:
            self.phicorehole_g = self.phicorehole_g[:gcut2].copy()

        Lcut = (2 * lcut + 1)**2
        T_Lqp = npy.zeros((Lcut, nq, np))
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

        (g_lg, n_qg, nt_qg,
         Delta_lq) = self.create_compensation_charges(r_g, dr_g, phi_jg,
                                                      phit_jg, np,
                                                      T_Lqp)

        self.g_lg = g_lg

        # Solves the radial poisson equation for density n_g
        def H(n_g, l):
            yrrdr_g = npy.zeros(gcut2)
            nrdr_g = n_g * r_g * dr_g
            hartree(l, nrdr_g, beta, ng, yrrdr_g)
            yrrdr_g *= r_g * dr_g
            return yrrdr_g

        wnc_g = H(nc_g, l=0)
        wnct_g = H(nct_g, l=0)


        self.wg_lg = wg_lg = [H(g_lg[l], l) for l in range(lmax + 1)]

        wn_lqg = [npy.array([H(n_qg[q], l) for q in range(nq)])
                  for l in range(2 * lcut + 1)]
        wnt_lqg = [npy.array([H(nt_qg[q], l) for q in range(nq)])
                   for l in range(2 * lcut + 1)]

        Delta0 = self.Delta0

        rdr_g = r_g * dr_g
        dv_g = r_g * rdr_g
        A = 0.5 * npy.dot(nc_g, wnc_g)
        A -= sqrt(4 * pi) * self.Z * npy.dot(rdr_g, nc_g)
        mct_g = nct_g + Delta0 * g_lg[0]
        wmct_g = wnct_g + Delta0 * wg_lg[0]
        A -= 0.5 * npy.dot(mct_g, wmct_g)
        self.M = A
        self.MB = -npy.dot(dv_g * nct_g, vbar_g)

        # Correction for average electrostatic potential:
        #
        #   dEH = dEH0 + dot(D_p, dEH_p)
        #
        self.dEH0 = sqrt(4 * pi) * (wnc_g - wmct_g -
                                    sqrt(4 * pi) * self.Z * r_g * dr_g).sum()
        dEh_q = (wn_lqg[0].sum(1) - wnt_lqg[0].sum(1) -
                 Delta_lq[0] * wg_lg[0].sum())
        self.dEH_p = npy.dot(dEh_q, T_Lqp[0]) * sqrt(4 * pi)

        A_q = 0.5 * (npy.dot(wn_lqg[0], nc_g)
                     + npy.dot(n_qg, wnc_g))
        A_q -= sqrt(4 * pi) * self.Z * npy.dot(n_qg, rdr_g)
        A_q -= 0.5 * (npy.dot(wnt_lqg[0], mct_g)
                     + npy.dot(nt_qg, wmct_g))
        A_q -= 0.5 * (npy.dot(mct_g, wg_lg[0])
                      + npy.dot(g_lg[0], wmct_g)) * Delta_lq[0]
        self.M_p = npy.dot(A_q, T_Lqp[0])

        AB_q = -npy.dot(nt_qg, dv_g * vbar_g)
        self.MB_p = npy.dot(AB_q, T_Lqp[0])

        A_lqq = []
        for l in range(2 * lcut + 1):
            A_qq = 0.5 * npy.dot(n_qg, npy.transpose(wn_lqg[l]))
            A_qq -= 0.5 * npy.dot(nt_qg, npy.transpose(wnt_lqg[l]))
            if l <= lmax:
                A_qq -= 0.5 * npy.outer(Delta_lq[l],
                                               npy.dot(wnt_lqg[l], g_lg[l]))
                A_qq -= 0.5 * npy.outer(npy.dot(nt_qg, wg_lg[l]),
                                               Delta_lq[l])
                A_qq -= 0.5 * npy.dot(g_lg[l], wg_lg[l]) * \
                        npy.outer(Delta_lq[l], Delta_lq[l])
            A_lqq.append(A_qq)

        self.M_pp = npy.zeros((np, np))
        L = 0
        for l in range(2 * lcut + 1):
            for m in range(2 * l + 1):
                self.M_pp += npy.dot(npy.transpose(T_Lqp[L]),
                                     npy.dot(A_lqq[l], T_Lqp[L]))
                L += 1

        if xcfunc.is_gllb():
            if xcfunc.xc.relaxed_core_response:
                self.njcore = extra_xc_data['njcore']
                self.core_A_kp = npy.zeros((self.njcore, np))
                self.core_At_kp = npy.zeros((self.njcore, np))
                self.core_B = npy.dot(g_lg[0], wg_lg[0]) / sqrt(4*pi)
                self.core_C = npy.dot(nct_g, wg_lg[0]) / sqrt(4*pi)
                self.coreref_k = npy.zeros((self.njcore))
                for k in range(0, self.njcore):
                    # Put the density of core orbital into radial
                    # representation
                    rho_g = extra_xc_data['core_orbital_density_'
                                          + str(k)] * sqrt(4*pi)

                    # Calculate the D_p dependent correction for E^a
                    self.core_A_kp[k] = npy.dot(npy.dot(n_qg, H(rho_g, 0)),
                                                T_Lqp[0])

                    # Calculate the D_P dependent correction for \tilde{E}^a
                    self.core_At_kp[k] = npy.dot(npy.dot(nt_qg, wg_lg[0]),
                                                 T_Lqp[0]) / sqrt(4 * pi)

                    # All other contributions are already included in
                    # reference from setup
                    self.coreref_k[k] = extra_xc_data['core_ref_' + str(k)]

        # Make a radial grid descriptor:
        rgd = RadialGridDescriptor(r_g, dr_g)

        xc = XCRadialGrid(xcfunc, rgd, nspins)

        self.xc_correction = XCCorrection(
            xc,
            [divrl(phi_g, l, r_g) for l, phi_g in zip(l_j, phi_jg)],
            [divrl(phit_g, l, r_g) for l, phit_g in zip(l_j, phit_jg)],
            nc_g / sqrt(4 * pi),
            nct_g / sqrt(4 * pi),
            rgd,
            [(j, l_j[j]) for j in range(nj)],
            2 * lcut,
            data.e_xc,
            self.phicorehole_g,
            data.fcorehole,
            nspins,
            tauc_g)

        # softgauss controls the use of particularly smooth compensation
        # charges, which are not required to approach zero at the
        # augmentation region boundary.
        #
        # Presently disabled
        if self.softgauss:
            rcutsoft = rcut2####### + 1.4
        else:
            rcutsoft = rcut2

        self.rcutsoft = rcutsoft

        # XXX reinstate the following check with correct names
        #if xcfunc.get_name() != self.xcname:
        #    raise RuntimeError('Not the correct XC-functional!')

        # Use atomic all-electron energies as reference:
        self.Kc = data.e_kinetic_core - data.e_kinetic
        self.M -= data.e_electrostatic
        self.E = data.e_total

        self.O_ii = sqrt(4.0 * pi) * unpack(self.Delta_pL[:, 0].copy())

        self.Delta_Lii = npy.zeros((ni, ni, self.Lmax))
        for L in range(self.Lmax):
            self.Delta_Lii[:,:,L] = unpack(self.Delta_pL[:, L].copy())

        K_q = []
        for j1 in range(nj):
            for j2 in range(j1, nj):
                K_q.append(data.e_kin_jj[j1, j2])
        self.K_p = sqrt(4 * pi) * npy.dot(K_q, T_Lqp[0])

        r = 0.02 * rcutsoft * npy.arange(51, dtype=float)
##        r = 0.04 * rcutsoft * npy.arange(26, dtype=float)
        alpha = rcgauss**-2
        self.alpha = alpha
        if self.softgauss:
            assert lmax <= 2
            alpha2 = 22.0 / rcutsoft**2
            alpha2 = 15.0 / rcutsoft**2 # What the heck

            vt0 = 4 * pi * (erf(sqrt(alpha) * r) - erf(sqrt(alpha2) * r))
            vt0[0] = 8 * sqrt(pi) * (sqrt(alpha) - sqrt(alpha2))
            vt0[1:] /= r[1:]
            vt_l = [vt0]
            if lmax >= 1:
                arg = npy.clip(alpha2 * r**2, 0.0, 700.0)
                e2 = npy.exp(-arg)
                arg = npy.clip(alpha * r**2, 0.0, 700.0)
                e = npy.exp(-arg)
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
        g = alpha2**1.5 * npy.exp(-alpha2 * r**2)
        g[-1] = 0.0
        self.ghat_l = [Spline(l, rcutsoft, d_l[l] * alpha2**l * g)
                       for l in range(lmax + 1)]

        self.rcutcomp = sqrt(10) * rcgauss

        # Compute inverse overlap coefficients C_ii
        self.C_ii = -npy.dot(self.O_ii, npy.linalg.inv(
            npy.identity(ni) + npy.dot(self.B_ii, self.O_ii)))

    def create_compensation_charges(self, r_g, dr_g, phi_jg, phit_jg, np,
                                    T_Lqp):
        lmax = self.lmax
        rcgauss = self.rcgauss
        #r_g = self.r_g
        #dr_g = self.dr_g
        gcut2 = self.gcut2
        nq = self.nq

        # Create gaussians used to expand compensation charges
        g_lg = npy.zeros((lmax + 1, gcut2))
        g_lg[0] = 4 / rcgauss**3 / sqrt(pi) * npy.exp(-(r_g / rcgauss)**2)
        for l in range(1, lmax + 1):
            g_lg[l] = 2.0 / (2 * l + 1) / rcgauss**2 * r_g * g_lg[l - 1]

        for l in range(lmax + 1):
            g_lg[l] /= npy.dot(r_g**(l + 2) * dr_g, g_lg[l])

        n_qg = npy.zeros((nq, gcut2))
        nt_qg = npy.zeros((nq, gcut2))
        q = 0 # q: common index for j1, j2
        for j1 in range(self.nj):
            for j2 in range(j1, self.nj):
                n_qg[q] = phi_jg[j1] * phi_jg[j2]
                nt_qg[q] = phit_jg[j1] * phit_jg[j2]
                q += 1

        Delta_lq = npy.zeros((lmax + 1, nq))
        for l in range(lmax + 1):
            Delta_lq[l] = npy.dot(n_qg - nt_qg, r_g**(2 + l) * dr_g)

        self.Lmax = (lmax + 1)**2
        self.Delta_pL = npy.zeros((np, self.Lmax))
        for l in range(lmax + 1):
            L = l**2
            for m in range(2 * l + 1):
                delta_p = npy.dot(Delta_lq[l], T_Lqp[L + m])
                self.Delta_pL[:, L + m] = delta_p

        Delta0 = npy.dot(self.nc_g - self.nct_g,
                        r_g**2 * dr_g) - self.Z / sqrt(4 * pi)
        self.Delta0 = Delta0
        return g_lg, n_qg, nt_qg, Delta_lq

    def construct_core_densities(self, r_g, dr_g, beta, setupdata):
        rcore = self.find_core_density_cutoff(r_g, dr_g, setupdata.nc_g)
        nct = Spline(0, rcore, setupdata.nct_g, r_g, beta)
        return rcore, setupdata.nc_g, setupdata.nct_g, nct

    def find_core_density_cutoff(self, r_g, dr_g, nc_g):
        # Find cutoff for core density:
        if self.Nc == 0:
            return 0.5
        else:
            N = 0.0
            g = self.ng - 1
            while N < 1e-7:
                N += sqrt(4 * pi) * nc_g[g] * r_g[g]**2 * dr_g[g]
                g -= 1
            return r_g[g]

    def set_hubbard_u(self, U, l):
        """Set Hubbard parameter.

        U in atomic units
        """
        self.HubU = U;
        self.Hubl = l;
        self.Hubi = 0;
        for ll in self.l_j:
            if ll == self.Hubl:
                break
            self.Hubi = self.Hubi+2*ll+1
        #print self.Hubi

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

        phit_j = []
        for j, phit_g in enumerate(phit_jg):
            if self.n_j[j] > 0:
                l = self.l_j[j]
                phit = phit_g[gcut3]
                dphitdr = ((phit - phit_g[gcut3 - 1]) /
                           (r_g[gcut3] - r_g[gcut3 - 1]))
                phit_g[gcut2:gcut3] -= phit * a_g + dphitdr * b_g
                phit_g[gcut3:] = 0.0
                phit_j.append(Spline(l, rcut3, phit_g, r_g, beta,
                                     points=100))
        return phit_j

    def read_basis_functions(self, basis):
        if isinstance(basis, str):
            basis = Basis(self.symbol, basis)

        rc = basis.d * (basis.ng - 1)
        r_g = npy.linspace(0., rc, basis.ng)

        # enable if-statement to revert to 'inefficient' equal-range basis
        # functions.  Left for testing purposes
        if 0:
            for j, bf in enumerate(basis.bf_j):
                phit_g = npy.zeros(r_g.shape)
                phit_g[:bf.ng] = bf.phit_g
                bf.phit_g = phit_g
                bf.ng = basis.ng
                bf.rc = rc

        phit_j = [Spline(bf.l, bf.rc, divrl(bf.phit_g, bf.l, r_g[:bf.ng]))
                       for bf in basis.bf_j]
        return phit_j

    def print_info(self, text):
        if self.phicorehole_g is None:
            text(self.symbol + '-setup:')
        else:
            text('%s-setup (%.1f core hole):' % (self.symbol, self.fcorehole))
        text('  name   :', atomic_names[self.Z])
        text('  id     :', self.fingerprint)
        text('  Z      :', self.Z)
        text('  valence:', self.Nv)
        if self.phicorehole_g is None:
            text('  core   : %d' % self.Nc)
        else:
            text('  core   : %.1f' % self.Nc)
        text('  charge :', self.Z - self.Nv - self.Nc)
        text('  file   :', self.data.filename)
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
        self.R_sii = npy.zeros((nsym, self.ni, self.ni))
        i1 = 0
        for l in self.l_j:
            i2 = i1 + 2 * l + 1
            for s, R_lmm in enumerate(R_slmm):
                self.R_sii[s, i1:i2, i1:i2] = R_lmm[l]
            i1 = i2

    def symmetrize(self, a, D_aii, map_sa):
        D_ii = npy.zeros((self.ni, self.ni))
        for s, R_ii in enumerate(self.R_sii):
            D_ii += npy.dot(R_ii, npy.dot(D_aii[map_sa[s][a]],
                                              npy.transpose(R_ii)))
        return D_ii / len(map_sa)

    def get_partial_waves(self):
        """Return spline representation of partial waves and densities."""

        l_j = self.l_j
        nj = len(l_j)
        beta = self.beta

        # cutoffs
        rcut2 = 2 * max(self.rcut_j)
        gcut2 = 1 + int(rcut2 * self.ng / (rcut2 + beta))

        # radial grid
        g = npy.arange(self.ng, dtype=float)
        r_g = beta * g / (self.ng - g)

        data = self.data

        # Construct splines:
        nc_g = data.nc_g.copy()
        nct_g = data.nct_g.copy()
        tauc_g = data.tauc_g
        nc_g[gcut2:] = nct_g[gcut2:] = 0.0
        nc = Spline(0, rcut2, data.nc_g, r_g, beta, points=1000)
        nct = Spline(0, rcut2, data.nct_g, r_g, beta, points=1000)
        if tauc_g is None:
            tauc_g = npy.zeros(nct_g.shape)
            tauct_g = tauc_g
        tauc = Spline(0, rcut2, data.tauc_g, r_g, beta, points=1000)
        tauct = Spline(0, rcut2, data.tauct_g, r_g, beta, points=1000)
        phi_j = []
        phit_j = []
        for j, (phi_g, phit_g) in enumerate(zip(data.phi_jg, data.phit_jg)):
            l = l_j[j]
            phi_g = phi_g.copy()
            phit_g = phit_g.copy()
            phi_g[gcut2:] = phit_g[gcut2:] = 0.0
            phi_j.append(Spline(l, rcut2, phi_g, r_g, beta, points=100))
            phit_j.append(Spline(l, rcut2, phit_g, r_g, beta, points=100))
        return phi_j, phit_j, nc, nct, tauc, tauct

    def calculate_oscillator_strengths(self, r_g, dr_g, phi_jg):
        self.A_ci = npy.zeros((3, self.ni))
        nj = len(phi_jg)
        i = 0
        for j in range(nj):
            l = self.l_j[j]
            if l == 1:
                a = npy.dot(r_g**3 * dr_g, phi_jg[j] * self.data.phicorehole_g)

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

        where phi_i1 is an all electron function and phit_i1 is its
        smooth partner.
        """

        if hasattr(self, 'I4_iip'):
##            print "already done"
            return # job already done

        # radial grid
        ng = self.ng
        g = npy.arange(ng, dtype=float)
        r_g = self.beta * g / (ng - g)
        dr_g = self.beta * ng / (ng - g)**2

        phi_jg = self.data.phi_jg
        phit_jg = self.data.phit_jg

        # compute radial parts
        nj = len(self.l_j)
        R_llll = npy.zeros((nj,nj,nj,nj))
        for i1 in range(nj):
            for i2 in range(nj):
                for i3 in range(nj):
                    for i4 in range(nj):
                        R_llll[i1,i2,i3,i4] = npy.dot( r_g**2 * dr_g,
                                                       phi_jg[i1]*phi_jg[i2]*
                                                       phi_jg[i3]*phi_jg[i4] -
                                                       phit_jg[i1]*phit_jg[i2]*
                                                       phit_jg[i3]*phit_jg[i4])

        # prepare for angular parts
        L_i = []
        j_i = []
        for j, l1 in enumerate(self.l_j):
            for m1 in range(2 * l1 + 1):
                L_i.append(l1**2 + m1)
                j_i.append(j)
        ni = len(L_i)
        np = ni * (ni + 1) // 2 # length for packing
        # j_i is the list of j values
        # L_i is the list of L (=l**2+m for 0<=m<l) values
        # https://wiki.fysik.dtu.dk/gpaw/devel/overview.html

        # calculate the integrals
        I4_iip = npy.empty((ni, ni, np))
        I = npy.empty((ni, ni))
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
                        I[i3,i4] = npy.dot( G_LLL[L1,L2],
                                            G_LLL[L3,L4] ) *\
                                    R_llll[j1,j2,j3,j4]
                I4_iip[i1,i2,:] = pack2(I)

        self.I4_iip = I4_iip



if __name__ == '__main__':
    print """\
You are using the wrong setup.py script!  This setup.py defines a
Setup class used to hold the atomic data needed for a specific atom.
For building the GPAW code you must use the setup.py distutils script
at the root of the code tree.  Just do "cd .." and you will be at the
right place."""
