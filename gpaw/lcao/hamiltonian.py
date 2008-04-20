from math import sqrt, pi
from time import time

import numpy as npy

from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw.utilities.lapack import diagonalize
from gpaw.spherical_harmonics import Yl
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

        for nucleus in self.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nucleus.P_kmi = npy.zeros((nkpts, self.nao, ni), self.dtype)

        if self.tci is None:
            self.tci = TwoCenterIntegrals(self.setups, self.ng)
                   
        self.S_kmm = npy.zeros((nkpts, self.nao, self.nao), self.dtype)
        self.T_kmm = npy.zeros((nkpts, self.nao, self.nao), self.dtype)
        S_mm = npy.zeros((self.nao, self.nao), self.dtype)
        T_mm = npy.zeros((self.nao, self.nao), self.dtype)

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
                for l in range(5):
                    rlY_m = npy.empty(2 * l + 1)
                    Yl(l, d, rlY_m)
                    rlY_lm.append(rlY_m)
                    
                phase_k = npy.exp(-2j * pi * npy.dot(self.ibzk_kc, offset))
                phase_k.shape = (-1, 1, 1)

                # Calculate basis-basis overlaps:
                self.st(a, b, r, rlY_lm, phase_k, selfinteraction)

                # Calculate basis-projector function overlaps:
                self.p(a, b, r,
                       [rlY_m * (-1)**l
                        for l, rlY_m in enumerate(rlY_lm)],
                       phase_k)
                if a != b or offset.any():
                    self.p(b, a, r,
                           rlY_lm,
                           phase_k.conj())
    
        for nucleus in self.nuclei:
            dO_ii = nucleus.setup.O_ii
            for S_mm, P_mi in zip(self.S_kmm, nucleus.P_kmi):
                S_mm += npy.dot(P_mi, npy.inner(dO_ii, P_mi).conj())
                
        # Check that the overlap matrix is positive definite        
        '''s_m = npy.empty(self.nao)
        for S_mm in self.S_kmm:
            assert diagonalize(S_mm.copy(), s_m) == 0
            if s_m[0] <= 0:
                print s_m[:10]
                raise RuntimeError('Overlap matrix not positive definite!')'''

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

    def st(self, a, b, r, rlY_lm, phase_k, selfinteraction):
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
                s_mm, t_mm = self.tci.st_overlap3(ida, idb, la, lb,
                                                              r, rlY_lm)
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
                        s1_kmm = s_kmm.transpose(0,2,1).conj()
                        t1_kmm = t_kmm.transpose(0,2,1).conj()
                        self.S_kmm[:, ma:ma2, mb:mb2] += s1_kmm
                        self.T_kmm[:, ma:ma2, mb:mb2] += t1_kmm
                    self.S_kmm[:, mb:mb2, ma:ma2] += s_kmm
                    self.T_kmm[:, mb:mb2, ma:ma2] += t_kmm
                mb = mb2
            ma = ma2

    def p(self, a, b, r, rlY_lm, phase_k):
        """Calculate basis-projector functions overlaps for the (a,b) pair
        of atoms."""

        setupa = self.nuclei[a].setup
        ma = self.nuclei[a].m
        nucleusb = self.nuclei[b]
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
                p_mi = self.tci.p(ida, idb, la, lb, r, rlY_lm)
                if self.gamma:
                    nucleusb.P_kmi[0, ma:ma2, ib:ib2] += p_mi
                else:
                    nucleusb.P_kmi[:, ma:ma2, ib:ib2] += (p_mi[None, :, :] *
                                                          phase_k)
                ib = ib2
            ma = ma2
        
    def calculate_effective_potential_matrix(self, Vt_skmm):
        Vt_skmm[:] = 0.0
        
        # Count number of boxes:
        nb = 0
        for nucleus in self.nuclei:
            if nucleus.phit_i is not None:
                nb += len(nucleus.phit_i.box_b)
        
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
                if debug:	
                    box_b = [box.lfs for box in phit_i.box_b]
                else:	
                    box_b = phit_i.box_b
                b2 = b1 + len(box_b)
                m_b[b1:b2] = m
                lfs_b.extend(box_b)
                if not self.gamma:
                    phase_bk[b1:b2] = phit_i.phase_kb.T
                b1 = b2
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
