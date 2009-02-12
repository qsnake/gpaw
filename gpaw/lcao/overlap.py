from math import sqrt, pi

import numpy as np
from numpy.fft import ifft

from gpaw.spherical_harmonics import Yl, nablaYL
from ase import Atoms
from ase.calculators.neighborlist import NeighborList

from gpaw.spline import Spline
from gpaw.gaunt import gaunt
from gpaw.utilities import fac


# Generate the coefficients for the Fourier-Bessel transform
C = []
a = 0.0
n = 5
for n in range(n):
    c = np.zeros(n+1, complex)
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
    g = np.zeros(m)
    for n in range(l + 1):
        g += (dr * 2 * m * k**(l - n) *
              ifft(C[l][n] * f * r**(1 + l - n), 2 * m)[:m].real)
    return g


class TwoCenterIntegralSplines:
    """ Two-center integrals class.

    This class implements a Fourier-space calculation of two-center
    integrals.
    """

    def __init__(self, setups, ng=2**12):
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
        self.r_g = np.arange(self.ng) * self.dr
        self.Q = 4 * self.ng
        self.dk = 2 * pi / self.Q / self.dr
        self.k = np.arange(self.Q // 2) * self.dk

        phit_g = np.zeros(self.ng) 
        phit_jq = {}
        for setup in setups:
            for j, phit in enumerate(setup.phit_j):
                l = phit.get_angular_momentum_number()
                id = (setup.symbol, j)
                phit_g[0:self.ng] = [phit(r) for r in self.r_g[0:self.ng]]
                phit_q = fbt(l, phit_g, self.r_g, self.k)
                phit_jq[id] = (l, phit_q)

        pt_g = np.zeros(self.ng) 
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
        S_g = np.zeros(2 * self.ng)
        self.lmax = l1 + l2
        splines = []
        R = np.arange(self.Q // 2) * self.dr
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
                a_g[0] = 8 * np.sum(a_q * k1**(-l1 - l2)) * self.dk
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

        p_mi = np.zeros((2 * la + 1, 2 * lb + 1))
        dPdR_cmi = None
        if derivative:
            dPdR_cmi = np.zeros((3, 2 * la + 1, 2 * lb + 1))

        l = (la + lb) % 2
        for p in self.P[(ida, idb)]:
            # Wouldn't that *actually* be G_mmi ?
            G_mmm = gaunt[la**2:(la + 1)**2,
                          lb**2:(lb + 1)**2,
                          l**2:(l + 1)**2].transpose((0, 2, 1))
            P, dPdr = p.get_value_and_derivative(r)

            GrlY_mi = np.dot(rlY_lm[l], G_mmm)
            
            p_mi += P * GrlY_mi

            # If basis function and projector are located on the same atom,
            # the overlap is translation invariant
            if derivative:
                if r < 1e-14:
                    dPdR_cmi[:] = 0.
                else:
                    Rhat = R / r
                    for c in range(3):
                        A_mi = P * np.dot(drlYdR_lmc[l][:, c], G_mmm)
                        B_mi = dPdr * GrlY_mi * Rhat[c]
                        dPdR_cmi[c, :, :] += A_mi + B_mi

            l += 2
        return p_mi, dPdR_cmi
        
    def st_overlap3(self, ida, idb, la, lb, r, R, rlY_lm, drlYdR_lmc):
        """ Returns the overlap and kinetic energy matrices. """
        
        s_mm = np.zeros((2 * lb + 1, 2 * la + 1))
        t_mm = np.zeros((2 * lb + 1, 2 * la + 1))
        dSdR_cmm = None
        dTdR_cmm = None
        derivative = bool(drlYdR_lmc)
        if derivative:
            dSdR_cmm = np.zeros((3, 2 * lb + 1, 2 * la + 1))
            dTdR_cmm = np.zeros((3, 2 * lb + 1, 2 * la + 1))
        ssplines = self.S[(ida, idb)]
        tsplines = self.T[(ida, idb)]
        l = (la + lb) % 2
        for s, t in zip(ssplines, tsplines):
            G_mmm = gaunt[lb**2:(lb + 1)**2,
                          la**2:(la + 1)**2,
                          l**2:(l + 1)**2].transpose((0, 2, 1))
            
            S, dSdr = s.get_value_and_derivative(r)
            T, dTdr = t.get_value_and_derivative(r)

            GrlY_mm = np.dot(rlY_lm[l], G_mmm)

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
                        GdrlYdR_mm = np.dot(drlYdR_lmc[l][:, c], G_mmm)
                        rlYRhat_mm = GrlY_mm * Rhat[c]

                        S1_mm = S * GdrlYdR_mm
                        S2_mm = dSdr * rlYRhat_mm
                        dSdR_cmm[c, :, :] += S1_mm + S2_mm

                        T1_mm = T * GdrlYdR_mm
                        T2_mm = dTdr * rlYRhat_mm
                        dTdR_cmm[c, :, :] += T1_mm + T2_mm
            
            l += 2
        return s_mm, t_mm, dSdR_cmm, dTdR_cmm


class TwoCenterIntegrals:
    """Hamiltonian class for LCAO-basis calculations."""

    def __init__(self, domain, setups, gamma=True, ibzk_qc=None):
        self.domain = domain
        self.setups = setups
        self.gamma = gamma
        self.ibzk_qc = ibzk_qc

        self.tci = None
        self.nl = None
        self.atoms = None
        self.M_a = None

        # Derivative overlaps should be evaluated lazily rather than
        # during initialization  to save memory/time. This is not implemented
        # yet, so presently we disable this.  Change behaviour by setting
        # this boolean.
        self.lcao_forces = False # XXX

    def set_positions(self, spos_ac):
        """Setting up S_kmm, T_kmm and P_kmi for LCAO calculations.

        ======    ==============================================
        S_kmm     Overlap between pairs of basis-functions
        T_kmm     Kinetic-Energy operator
        P_kmi     Overlap between basis-functions and projectors
        ======    ==============================================
        """

        if not self.tci:
            # First time:
            self.tci = TwoCenterIntegralSplines(self.setups.setups.values())
            natoms = len(spos_ac)
            cutoff_a = np.empty(natoms)
            self.M_a = np.empty(natoms, int)
            M = 0
            for a, setup in enumerate(self.setups):
                cutoff_a[a] = max([phit.get_cutoff()
                                   for phit in setup.phit_j])
                self.M_a[a] = M
                M += setup.niAO
            
            self.nl = NeighborList(cutoff_a, skin=0, sorted=True)
            self.atoms = Atoms(scaled_positions=spos_ac,
                               cell=self.domain.cell_cv,
                               pbc=self.domain.pbc_c)
        else:
            self.atoms.set_scaled_positions(spos_ac)
        
        self.nl.update(self.atoms)

    def calculate(self, spos_ac, S_qMM, T_qMM, P_aqMi, dtype):
        S_qMM[:] = 0.0
        T_qMM[:] = 0.0
        for P_qMi in P_aqMi.values():
            P_qMi[:] = 0.0
        
        if self.lcao_forces:
            nq = len(self.ibzk_qc)
            nao = self.setups.nao

            self.dPdR_akcmi = {}
            self.mask_amm = {}

            for a, M in enumerate(self.M_a):
                ni = self.setups[a].ni
                nM = self.setups[a].niAO

                dPdR_kcmi = np.zeros((nq, 3, nao, ni), dtype)
                self.dPdR_akcmi[a] = dPdR_kcmi
                # XXX Create "masks" on the nuclei which specify signs
                # and zeros for overlap derivatives.
                # This is inefficient and only a temporary hack!
                m1 = M
                m2 = m1 + nM
                mask_mm = np.zeros((nao, nao))
                mask_mm[:, m1:m2] = 1.
                mask_mm[m1:m2, :] = -1.
                mask_mm[m1:m2, m1:m2] = 0.
                self.mask_amm[a] = mask_mm

            self.dSdR_kcmm = np.zeros((nq, 3, nao, nao), dtype)
            self.dTdR_kcmm = np.zeros((nq, 3, nao, nao), dtype)
        cell_cv = self.domain.cell_cv

        if self.lcao_forces: # XXX
            dPdR_akcmi = self.dPdR_akcmi
        else:
            dPdR_akcmi = {}

        for a1, spos1_c in enumerate(spos_ac):
            P1_qMi = P_aqMi.get(a1)
            dPdR1_qvMi = dPdR_akcmi.get(a1) # XXX variable names
            M1 = self.M_a[a1]
            i, offsets = self.nl.get_neighbors(a1)
            for a2, offset in zip(i, offsets):
                P2_qMi = P_aqMi.get(a2)
                dPdR2_qvMi = dPdR_akcmi.get(a2)
                if P1_qMi is None and P2_qMi is None:
                    continue

                assert a2 >= a1
                selfinteraction = (a1 == a2 and offset.any())
                M2 = self.M_a[a2]
                spos2_c = spos_ac[a2] + offset

                d = -np.dot(spos2_c - spos1_c, cell_cv)
                r = sqrt(np.dot(d, d))
                rlY_lm = []
                drlYdR_lmc = []
                for l in range(5):
                    rlY_m = np.empty(2 * l + 1)
                    Yl(l, d, rlY_m)
                    rlY_lm.append(rlY_m)

                    if self.lcao_forces:
                        drlYdR_mc = np.empty((2 * l + 1, 3))
                        for m in range(2 * l + 1):
                            L = l**2 + m
                            drlYdR_mc[m, :] = nablaYL(L, d)
                        drlYdR_lmc.append(drlYdR_mc)

                phase_q = np.exp(-2j * pi * np.dot(self.ibzk_qc, offset))
                phase_q.shape = (-1, 1, 1)

                if P2_qMi is not None:
                    # Calculate basis-basis overlaps:
                    self.st(a1, a2, r, d, rlY_lm, drlYdR_lmc, phase_q,
                            selfinteraction, M1, M2, S_qMM, T_qMM)

                    # Calculate basis-projector function overlaps:
                    # So what's the reason for (-1)**l ?
                    # Better do the same thing with drlYdR
                    self.p(a1, a2, r, d,
                           [rlY_m * (-1)**l
                            for l, rlY_m in enumerate(rlY_lm)],
                           [drlYdR_mc * (-1)**l
                            for l, drlYdR_mc in enumerate(drlYdR_lmc)],
                           phase_q, P2_qMi, M1, dPdR2_qvMi)
                if P1_qMi is not None and (a1 != a2 or offset.any()):
                    self.p(a2, a1, r, d,
                           rlY_lm, drlYdR_lmc,
                           phase_q.conj(), P1_qMi, M2, dPdR1_qvMi)

        # Only lower triangle matrix elements of S and T have been calculated
        # so far.  Better fill out the rest
        if self.lcao_forces:
            nao = self.setups.nao
            tri1 = np.tri(nao)
            tri2 = np.tri(nao, None, -1)
            def tri2full(matrix, op=1):
                return tri1 * matrix + (op * tri2 * matrix).transpose().conj()

            for S_mm, T_mm, dSdR_cmm, dTdR_cmm in zip(S_qMM,
                                                      T_qMM,
                                                      self.dSdR_kcmm,
                                                      self.dTdR_kcmm):
                S_mm[:] = tri2full(S_mm)
                T_mm[:] = tri2full(T_mm)
                for c in range(3):
                    dSdR_cmm[c, :, :] = tri2full(dSdR_cmm[c], -1) # XXX
                    dTdR_cmm[c, :, :] = tri2full(dTdR_cmm[c], -1) # XXX

            # These will be the *unmodified* basis function overlaps
            # XXX We may be able to avoid remembering these.
            # The derivative dSdR which depends on the projectors is
            # presently found during force calculations, which means it is
            # not necessary here
            #self.Theta_kmm = self.S_kmm.copy()
            self.Theta_qMM = S_qMM.copy()
            self.dThetadR_kcmm = self.dSdR_kcmm.copy()

        # Add adjustment from O_ii, having already calculated <phi_m1|phi_m2>:
        #
        #                         -----
        #                          \            ~a   a   ~a
        # S    = <phi  | phi  > +   )   <phi  | p > O   <p | phi  >
        #  m1m2      m1     m2     /        m1   i   ij   j     m2
        #                         -----
        #                          aij
        #

        for a, P_qMi in P_aqMi.items():
            dO_ii = self.setups[a].O_ii
            for S_MM, P_Mi in zip(S_qMM, P_qMi):
                S_MM += np.dot(P_Mi, np.inner(dO_ii, P_Mi).conj())

        comm = self.domain.comm
        comm.sum(S_qMM)
        comm.sum(T_qMM)

        if self.lcao_forces:
            comm.sum(self.Theta_qMM)
            comm.sum(self.dThetadR_kcmm)
            comm.sum(self.dSdR_kcmm)
            comm.sum(self.dTdR_kcmm)

    def st(self, a1, a2, r, R, rlY_lm, drlYdR_lmc, phase_q, selfinteraction,
           M1a, M2, S_qMM, T_qMM):
        """Calculate overlaps and kinetic energy matrix elements for the
        (a,b) pair of atoms."""

        setup1 = self.setups[a1]
        setup2 = self.setups[a2]
        for j1, phit1 in enumerate(setup1.phit_j):
            id1 = (setup1.symbol, j1)
            l1 = phit1.get_angular_momentum_number()
            M1b = M1a + 2 * l1 + 1
            M2a = M2
            for j2, phit2 in enumerate(setup2.phit_j):
                id2 = (setup2.symbol, j2)
                l2 = phit2.get_angular_momentum_number()
                M2b = M2a + 2 * l2 + 1
                (s_mm, t_mm, dSdR_cmm, dTdR_cmm) = \
                    self.tci.st_overlap3(id1, id2, l1, l2, r, R, rlY_lm,
                                         drlYdR_lmc)

                if self.gamma:
                    if selfinteraction:
                        S_qMM[0, M1a:M1b, M2a:M2b] += s_mm.T
                        T_qMM[0, M1a:M1b, M2a:M2b] += t_mm.T
                    S_qMM[0, M2a:M2b, M1a:M1b] += s_mm
                    T_qMM[0, M2a:M2b, M1a:M1b] += t_mm
                else:
                    s_qmm = s_mm[None, :, :] * phase_q.conj()
                    t_qmm = t_mm[None, :, :] * phase_q.conj()
                    if selfinteraction:
                        s1_qmm = s_qmm.transpose(0, 2, 1).conj()
                        t1_qmm = t_qmm.transpose(0, 2, 1).conj()
                        S_qMM[:, M1a:M1b, M2a:M2b] += s1_qmm
                        T_qMM[:, M1a:M1b, M2a:M2b] += t1_qmm
                    S_qMM[:, M2a:M2b, M1a:M1b] += s_qmm
                    T_qMM[:, M2a:M2b, M1a:M1b] += t_qmm

                if self.lcao_forces:
                    # the below is more or less copy-paste of the above
                    # XXX do this in a less silly way
                    if self.gamma:
                        if selfinteraction:
                            dSdRT_cmm = np.transpose(dSdR_cmm, (0, 2, 1))
                            dTdRT_cmm = np.transpose(dTdR_cmm, (0, 2, 1))
                            self.dSdR_kcmm[0, :, M1a:M1b, M2a:M2b] += dSdRT_cmm
                            self.dTdR_kcmm[0, :, M1a:M1b, M2a:M2b] += dTdRT_cmm
                        self.dSdR_kcmm[0, :, M2a:M2b, M1a:M1b] += dSdR_cmm
                        self.dTdR_kcmm[0, :, M2a:M2b, M1a:M1b] += dTdR_cmm
                    else:
                        # XXX cumbersome
                        phase_qc = phase_q[:, None, :, :].repeat(3, axis=1)
                        dSdR_kcmm = dSdR_cmm[None, :, :, :] * phase_qc.conj()
                        dTdR_kcmm = dTdR_cmm[None, :, :, :] * phase_qc.conj()

                        if selfinteraction:
                            dSdR1_kcmm = dSdR_kcmm.transpose(0, 1, 3, 2).conj()
                            dTdR1_kcmm = dTdR_kcmm.transpose(0, 1, 3, 2).conj()
                            self.dSdR_kcmm[:, :, M1a:M1b, M2a:M2b] += dSdR1_kcmm
                            self.dTdR_kcmm[:, :, M1a:M1b, M2a:M2b] += dTdR1_kcmm
                        self.dSdR_kcmm[:, :, M2a:M2b, M1a:M1b] += dSdR_kcmm
                        self.dTdR_kcmm[:, :, M2a:M2b, M1a:M1b] += dTdR_kcmm

                M2a = M2b
            M1a = M1b

    def p(self, a1, a2, r, R, rlY_lm, drlYdR_lm, phase_q, P2_qMi, M1a,
          dPdR2_qvMi=None):
        """Calculate basis-projector functions overlaps for the (a,b) pair
        of atoms."""

        setup1 = self.setups[a1]
        setup2 = self.setups[a2]
        for j1, phit1 in enumerate(setup1.phit_j):
            id1 = (setup1.symbol, j1)
            l1 = phit1.get_angular_momentum_number()
            M1b = M1a + 2 * l1 + 1
            i2a = 0
            for j2, pt2 in enumerate(setup2.pt_j):
                id2 = (setup2.symbol, j2)
                l2 = pt2.get_angular_momentum_number()
                i2b = i2a + 2 * l2 + 1
                p_mi, dPdR_cmi = self.tci.p(id1, id2, l1, l2, r, R,
                                            rlY_lm, drlYdR_lm)
                if self.gamma:
                    P2_qMi[0, M1a:M1b, i2a:i2b] += p_mi
                else:
                    P2_qMi[:, M1a:M1b, i2a:i2b] += (p_mi[None, :, :] * phase_q)

                if self.lcao_forces:
                    if self.gamma:
                        #dPdR2_qvMi[0, :, M1a:M1b, ib:ib2] += dPdR_cmi
                        dPdR2_qvMi[0, :, M1a:M1b, i2a:i2b] += dPdR_cmi
                    else: # XXX phase_qc
                        phase_qc = phase_q[:, None, :, :].repeat(3, axis=1)
                        dPdR2_qvMi[:, :, M1a:M1b, i2a:i2b] += \
                                      dPdR_cmi[None, :, :, :] * phase_qc
                        #dPdR2_qvMi[:, :, M1a:M1b, ib:ib2] += \
                        #              dPdR_cmi[None, :, :, :] * phase_qc
                        
                i2a = i2b
            M1a = M1b
