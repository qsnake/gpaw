from math import sqrt, pi
from time import time

import numpy as npy

from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.utilities.lapack import diagonalize
from gpaw.spherical_harmonics import Yl, nablaYL
from ase import Atoms
from ase.calculators.neighborlist import NeighborList
from gpaw import debug
from _gpaw import overlap


class LCAOHamiltonian:
    """Hamiltonian class for LCAO-basis calculations."""

    def __init__(self, ng=2**12):
        self.tci = None  # two-center integrals
        self.lcao_initialized = False
        self.ng = ng
        if debug:
            self.eig_lcao_iteration = 0

        # Derivative overlaps should be evaluated lazily rather than
        # during initialization  to save memory/time. This is not implemented
        # yet, so presently we disable this.  Change behaviour by setting
        # this boolean.
        self.lcao_forces = False # XXX

    def initialize(self, paw):
        self.setups = paw.setups
        self.ibzk_kc = paw.ibzk_kc
        self.gamma = paw.gamma
        self.dtype = paw.dtype

    def initialize_lcao(self):
        """Setting up S_kmm, T_kmm and P_kmi for LCAO calculations.

        ======    ==============================================
        S_kmm     Overlap between pairs of basis-functions
        T_kmm     Kinetic-Energy operator
        P_kmi     Overlap between basis-functions and projectors
        ======    ==============================================
        """

        nkpts = len(self.ibzk_kc)

        self.nao = 0
        for nucleus in self.nuclei:
            nucleus.m = self.nao
            self.nao += nucleus.get_number_of_atomic_orbitals()

        for nucleus in self.my_nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nucleus.P_kmi = npy.zeros((nkpts, self.nao, ni), self.dtype)

        if self.lcao_forces:
            for nucleus in self.nuclei:
                ni = nucleus.get_number_of_partial_waves()
                nucleus.dPdR_kcmi = npy.zeros((nkpts, 3, self.nao, ni),
                                              self.dtype)
                # XXX Create "masks" on the nuclei which specify signs
                # and zeros for overlap derivatives.
                # This is inefficient and only a temporary hack!
                m1 = nucleus.m
                m2 = m1 + nucleus.get_number_of_atomic_orbitals()
                mask_mm = npy.zeros((self.nao, self.nao))
                mask_mm[:, m1:m2] = 1.
                mask_mm[m1:m2, :] = -1.
                mask_mm[m1:m2, m1:m2] = 0.
                nucleus.mask_mm = mask_mm

        if self.tci is None:
            self.tci = TwoCenterIntegrals(self.setups, self.ng)

        self.S_kmm = npy.zeros((nkpts, self.nao, self.nao), self.dtype)
        self.T_kmm = npy.zeros((nkpts, self.nao, self.nao), self.dtype)
        if self.lcao_forces:
            self.dSdR_kcmm = npy.zeros((nkpts, 3, self.nao, self.nao),
                                       self.dtype)
            self.dTdR_kcmm = npy.zeros((nkpts, 3, self.nao, self.nao),
                                       self.dtype)

        cell_c = self.gd.domain.cell_c

        atoms = Atoms(positions=[n.spos_c * cell_c for n in self.nuclei],
                      cell=cell_c,
                      pbc=self.gd.domain.pbc_c)

        nl = NeighborList([max([phit.get_cutoff()
                                for phit in n.setup.phit_j])
                           for n in self.nuclei], skin=0, sorted=True)
        nl.update(atoms)

        for a, nucleusa in enumerate(self.nuclei):
            sposa = nucleusa.spos_c
            i, offsets = nl.get_neighbors(a)
            for b, offset in zip(i, offsets):
                assert b >= a
                selfinteraction = (a == b and offset.any())
                ma = nucleusa.m
                nucleusb = self.nuclei[b]
                sposb = nucleusb.spos_c + offset

                d = -cell_c * (sposb - sposa)
                r = sqrt(npy.dot(d, d))
                rlY_lm = []
                drlYdR_lmc = []
                for l in range(5):
                    rlY_m = npy.empty(2 * l + 1)
                    Yl(l, d, rlY_m)
                    rlY_lm.append(rlY_m)

                    if self.lcao_forces:
                        drlYdR_mc = npy.empty((2 * l + 1, 3))
                        for m in range(2 * l + 1):
                            L = l**2 + m
                            drlYdR_mc[m, :] = nablaYL(L, d)
                        drlYdR_lmc.append(drlYdR_mc)

                phase_k = npy.exp(-2j * pi * npy.dot(self.ibzk_kc, offset))
                phase_k.shape = (-1, 1, 1)

                # Calculate basis-basis overlaps:
                self.st(a, b, r, d, rlY_lm, drlYdR_lmc, phase_k,
                        selfinteraction)

                # Calculate basis-projector function overlaps:
                # So what's the reason for (-1)**l ?
                # Better do the same thing with drlYdR
                self.p(a, b, r, d,
                       [rlY_m * (-1)**l
                        for l, rlY_m in enumerate(rlY_lm)],
                       [drlYdR_mc * (-1)**l
                        for l, drlYdR_mc in enumerate(drlYdR_lmc)],
                       phase_k)
                if a != b or offset.any():
                    self.p(b, a, r, d,
                           rlY_lm, drlYdR_lmc,
                           phase_k.conj())

        # Only lower triangle matrix elements of S and T have been calculated
        # so far.  Better fill out the rest
        if self.lcao_forces:
            tri1 = npy.tri(self.nao)
            tri2 = npy.tri(self.nao, None, -1)
            def tri2full(matrix, op=1):
                return tri1 * matrix + (op * tri2 * matrix).transpose().conj()

            for S_mm, T_mm, dSdR_cmm, dTdR_cmm in zip(self.S_kmm,
                                                      self.T_kmm,
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

        if self.gd.comm.size > 1:
            self.S_kmm /= self.gd.comm.size

        for nucleus in self.my_nuclei:
            dO_ii = nucleus.setup.O_ii
            for S_mm, P_mi in zip(self.S_kmm, nucleus.P_kmi):
                S_mm += npy.dot(P_mi, npy.inner(dO_ii, P_mi).conj())

        if self.gd.comm.size > 1:
            self.gd.comm.sum(self.S_kmm)

        # Near-linear dependence check. This is done by checking the
        # eigenvalues of the overlap matrix S_kmm. Eigenvalues close
        # to zero mean near-linear dependence in the basis-set.
        self.linear_kpts = {}
        for k in range(nkpts):
            P_mm = self.S_kmm[k].copy()
            p_m = npy.empty(self.nao)

            dsyev_zheev_string = 'LCAO: '+'diagonalize-test'

            self.timer.start(dsyev_zheev_string)
            if debug:
                self.timer.start(dsyev_zheev_string+' %03d' % self.eig_lcao_iteration)

            if self.gd.comm.rank == 0:
                p_m[0] = 42
                info = diagonalize(P_mm, p_m)
                assert p_m[0] != 42
                if info != 0:
                    raise RuntimeError('Failed to diagonalize: info=%d' % info)

            if debug:
                self.timer.stop(dsyev_zheev_string+' %03d' % self.eig_lcao_iteration)
                self.eig_lcao_iteration += 1
            self.timer.stop(dsyev_zheev_string)

            self.gd.comm.broadcast(P_mm, 0)
            self.gd.comm.broadcast(p_m, 0)

            self.thres = 1e-6
            if (p_m <= self.thres).any():
                self.linear_kpts[k] = (P_mm, p_m)

        # Debug stuff
        if 0:
            print 'Hamiltonian S_kmm[0] diag'
            print self.S_kmm[0].diagonal()
            print 'Hamiltonian S_kmm[0]'
            for row in self.S_kmm[0]:
                print ' '.join(['%02.03f' % f for f in row])
            print 'Eigenvalues:'
            print npy.linalg.eig(self.S_kmm[0])[0]

        self.lcao_initialized = True

    def st(self, a, b, r, R, rlY_lm, drlYdR_lmc, phase_k, selfinteraction):
        """Calculate overlaps and kinetic energy matrix elements for the
        (a,b) pair of atoms."""

        setupa = self.nuclei[a].setup
        ma = self.nuclei[a].m
        nucleusb = self.nuclei[b]
        setupb = nucleusb.setup
        for ja, phita in enumerate(setupa.phit_j):
            ida = (setupa.symbol, ja)
            la = phita.get_angular_momentum_number()
            ma2 = ma + 2 * la + 1
            mb = nucleusb.m
            for jb, phitb in enumerate(setupb.phit_j):
                idb = (setupb.symbol, jb)
                lb = phitb.get_angular_momentum_number()
                mb2 = mb + 2 * lb + 1
                (s_mm, t_mm, dSdR_cmm, dTdR_cmm) = \
                    self.tci.st_overlap3(ida, idb, la, lb, r, R, rlY_lm,
                                         drlYdR_lmc)

                if self.gamma:
                    if selfinteraction:
                        self.S_kmm[0, ma:ma2, mb:mb2] += s_mm.T
                        self.T_kmm[0, ma:ma2, mb:mb2] += t_mm.T
                    self.S_kmm[0, mb:mb2, ma:ma2] += s_mm
                    self.T_kmm[0, mb:mb2, ma:ma2] += t_mm
                else:
                    s_kmm = s_mm[None, :, :] * phase_k.conj()
                    t_kmm = t_mm[None, :, :] * phase_k.conj()
                    if selfinteraction:
                        s1_kmm = s_kmm.transpose(0, 2, 1).conj()
                        t1_kmm = t_kmm.transpose(0, 2, 1).conj()
                        self.S_kmm[:, ma:ma2, mb:mb2] += s1_kmm
                        self.T_kmm[:, ma:ma2, mb:mb2] += t1_kmm
                    self.S_kmm[:, mb:mb2, ma:ma2] += s_kmm
                    self.T_kmm[:, mb:mb2, ma:ma2] += t_kmm

                if self.lcao_forces:
                    # the below is more or less copy-paste of the above
                    # XXX do this in a less silly way
                    if self.gamma:
                        if selfinteraction:
                            dSdRT_cmm = npy.transpose(dSdR_cmm, (0, 2, 1))
                            dTdRT_cmm = npy.transpose(dTdR_cmm, (0, 2, 1))
                            self.dSdR_kcmm[0, :, ma:ma2, mb:mb2] += dSdRT_cmm
                            self.dTdR_kcmm[0, :, ma:ma2, mb:mb2] += dTdRT_cmm
                        self.dSdR_kcmm[0, :, mb:mb2, ma:ma2] += dSdR_cmm
                        self.dTdR_kcmm[0, :, mb:mb2, ma:ma2] += dTdR_cmm
                    else:
                        # XXX cumbersome
                        phase_kc = phase_k[:, None, :, :].repeat(3, axis=1)
                        dSdR_kcmm = dSdR_cmm[None, :, :, :] * phase_kc.conj()
                        dTdR_kcmm = dTdR_cmm[None, :, :, :] * phase_kc.conj()

                        if selfinteraction:
                            dSdR1_kcmm = dSdR_kcmm.transpose(0, 1, 3, 2).conj()
                            dTdR1_kcmm = dTdR_kcmm.transpose(0, 1, 3, 2).conj()
                            self.dSdR_kcmm[:, :, ma:ma2, mb:mb2] += dSdR1_kcmm
                            self.dTdR_kcmm[:, :, ma:ma2, mb:mb2] += dTdR1_kcmm
                        self.dSdR_kcmm[:, :, mb:mb2, ma:ma2] += dSdR_kcmm
                        self.dTdR_kcmm[:, :, mb:mb2, ma:ma2] += dTdR_kcmm

                mb = mb2
            ma = ma2


    def p(self, a, b, r, R, rlY_lm, drlYdR_lm, phase_k):
        """Calculate basis-projector functions overlaps for the (a,b) pair
        of atoms."""

        nucleusb = self.nuclei[b]

        if not (self.lcao_forces or nucleusb.in_this_domain):
            return

        setupa = self.nuclei[a].setup
        ma = self.nuclei[a].m
        setupb = nucleusb.setup
        for ja, phita in enumerate(setupa.phit_j):
            ida = (setupa.symbol, ja)
            la = phita.get_angular_momentum_number()
            ma2 = ma + 2 * la + 1
            ib = 0
            for jb, ptb in enumerate(setupb.pt_j):
                idb = (setupb.symbol, jb)
                lb = ptb.get_angular_momentum_number()
                ib2 = ib + 2 * lb + 1
                p_mi, dPdR_cmi = self.tci.p(ida, idb, la, lb, r, R,
                                            rlY_lm, drlYdR_lm)
                if self.gamma and nucleusb.in_this_domain:
                    nucleusb.P_kmi[0, ma:ma2, ib:ib2] += p_mi
                elif nucleusb.in_this_domain:
                    nucleusb.P_kmi[:, ma:ma2, ib:ib2] += (p_mi[None, :, :] *
                                                          phase_k)

                if self.lcao_forces:
                    if self.gamma:
                        nucleusb.dPdR_kcmi[0, :, ma:ma2, ib:ib2] += dPdR_cmi
                    else: # XXX phase_kc
                        phase_kc = phase_k[:, None, :, :].repeat(3, axis=1)
                        nucleusb.dPdR_kcmi[:, :, ma:ma2, ib:ib2] += \
                            dPdR_cmi[None, :, :, :] * phase_kc

                ib = ib2
            ma = ma2

    def calculate_effective_potential_matrix(self, Vt_skmm):
        Vt_skmm[:] = 0.0

        # Count number of boxes:
        nb = 0
        for nucleus in self.nuclei:
            if nucleus.phit_i is not None:
                for phit in nucleus.phit_i.lf_j:
                    nb += len(phit.box_b)

        # Array to hold basis set index:
        m_b = npy.empty(nb, int)

        nkpts = len(self.ibzk_kc)

        if self.gamma:
            phase_bk = npy.empty((0, 0), complex)
        else:
            phase_bk = npy.empty((nb, nkpts), complex) # XXX

        m = 0
        b1 = 0
        lfs_b = []
        for nucleus in self.nuclei:
            phit_i = nucleus.phit_i
            if phit_i is not None:
                for phit in phit_i.lf_j:
                    if debug:
                        box_b = [box.lfs for box in phit.box_b]
                    else:
                        box_b = phit.box_b
                    b2 = b1 + len(box_b)
                    m_b[b1:b2] = m
                    lfs_b.extend(box_b)
                    if not self.gamma:
                        phase_bk[b1:b2] = phit.phase_kb.T
                    b1 = b2
                    m += phit.ni
            else:
                m += nucleus.get_number_of_atomic_orbitals()

        assert b1 == nb

        overlap(lfs_b, m_b, phase_bk, self.vt_sG, Vt_skmm)

    # Methods not in use any more
    def p_overlap(self, R_c, i1, pos1, id1, l1, m1, P_mi):
        i2 = 0
        for nucleus2 in self.nuclei:
            pos2 = nucleus2.spos_c
            d = (R_c + pos1 - pos2) * self.gd.domain.cell_c
            setup2 = nucleus2.setup
            for j2, phit2 in enumerate(setup2.phit_j):
                id2 = (setup2.symbol, j2)
                l2 = phit2.get_angular_momentum_number()
                for m2 in range(2 * l2 + 1):
                    P = self.tci.p_overlap(id1, id2, l1, l2, m1, m2, d)
                    P_mi[i2, i1] = P
                    i2 += 1

    def st_overlap(self, R_c, i1, pos1, id1, l1, m1, S_mm, T_mm):
       i2 = 0
       for nucleus2 in self.nuclei:
           pos2 = nucleus2.spos_c
           d = (pos1 - pos2 + R_c) * self.gd.domain.cell_c
           setup2 = nucleus2.setup
           for j2, phit2 in enumerate(setup2.phit_j):
               id2 = (setup2.symbol, j2)
               l2 = phit2.get_angular_momentum_number()
               for m2 in range(2 * l2 + 1):
                   S, T = self.tci.st_overlap(id1, id2, l1, l2, m1, m2, d)
                   S_mm[i1, i2] = S
                   T_mm[i1, i2] = T
                   i2 += 1

    def calculate_displacements(self, rmax):
        """Calculate displacement vectors to be used for the relevant
        phase factors (phase_k)."""

        nn_c = npy.zeros(3, int)  # Number of neighboring cells
        for c in range(3):
            if self.gd.domain.pbc_c[c]:
                nn_c[c] = 1 + int(2 * rmax / self.gd.domain.cell_c[c])

        nd = (1 + 2 * nn_c).prod()
        R_dc = npy.empty((nd, 3))
        d = 0
        for d1 in range(-nn_c[0], nn_c[0] + 1):
            for d2 in range(-nn_c[1], nn_c[1] + 1):
                for d3 in range(-nn_c[2], nn_c[2] + 1):
                    R_dc[d, :] = d1, d2, d3
                    d += 1
        return R_dc

    def old_initialize(self):
        self.nao = 0
        for nucleus in self.nuclei:
            self.nao += nucleus.get_number_of_atomic_orbitals()
        self.phi_mG = self.gd.zeros(self.nao)

        m1 = 0
        for nucleus in self.nuclei:
            niao = nucleus.get_number_of_atomic_orbitals()
            m2 = m1 + niao
            nucleus.initialize_atomic_orbitals(self.gd, 42, None)
            nucleus.create_atomic_orbitals(self.phi_mG[m1:m2], 0)
            m1 = m2
        assert m2 == self.nao

        for nucleus in self.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nucleus.P_mi = npy.zeros((self.nao, ni))
            nucleus.pt_i.integrate(self.phi_mG, nucleus.P_mi)

        self.S_mm = npy.zeros((self.nao, self.nao))
        rk(self.gd.dv, self.phi_mG, 0.0, self.S_mm)

        # Filling up the upper triangle:
        for m in range(self.nao - 1):
            self.S_mm[m, m:] = self.S_mm[m:, m]

        for nucleus in self.nuclei:
            self.S_mm += npy.dot(npy.dot(nucleus.P_mi, nucleus.setup.O_ii),
                            npy.transpose(nucleus.P_mi))

        self.T_mm = npy.zeros((self.nao, self.nao))
        Tphi_mG = self.gd.zeros(self.nao)
        self.kin.apply(self.phi_mG, Tphi_mG)
        r2k(0.5 * self.gd.dv, self.phi_mG, Tphi_mG, 0.0, self.T_mm)

        # Filling up the upper triangle:
        for m in range(self.nao - 1):
            self.T_mm[m, m:] = self.T_mm[m:, m]
