from math import sqrt, pi

import numpy as npy

from gpaw.hamiltonian import Hamiltonian
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw import debug
from _gpaw import overlap


class LCAOHamiltonian(Hamiltonian):
    """Hamiltonian class for LCAO-basis calculations."""

    def __init__(self, paw):
        Hamiltonian.__init__(self, paw)
        self.setups = paw.setups
        self.ibzk_kc = paw.ibzk_kc
        self.gamma = paw.gamma
        self.dtype = paw.dtype
        self.initialized = False
        self.ng = 2**12

    def initialize(self):
        """Setting up S_kmm, T_kmm and P_kmi for LCAO calculations.

        ======    ==============================================
        S_kmm     Overlap between pairs of basis-functions
        T_kmm     Kinetic-Energy operator
        P_kmi     Overlap between basis-functions and projectors
        ======    ==============================================
        """
        
        self.nao = 0
        for nucleus in self.nuclei:
            nucleus.initialize_atomic_orbitals(self.gd, self.ibzk_kc, lfbc=None)
            self.nao += nucleus.get_number_of_atomic_orbitals()

        tci = TwoCenterIntegrals(self.setups, self.ng)

        R_dc = self.calculate_displacements(tci.rcmax)
        
        nkpts = len(self.ibzk_kc)

        for nucleus1 in self.nuclei:
            pos1 = nucleus1.spos_c
            setup1 = nucleus1.setup
            ni1 = nucleus1.get_number_of_partial_waves()
            nucleus1.P_kmi = npy.zeros((nkpts, self.nao, ni1), self.dtype)
            P_mi = npy.zeros((self.nao, ni1), self.dtype)
            for R_c in R_dc:
                i1 = 0 
                for j1, pt1 in enumerate(setup1.pt_j):
                    id1 = (setup1.symbol, j1)
                    l1 = pt1.get_angular_momentum_number()
                    for m1 in range(2 * l1 + 1):
                        self.p_overlap(R_c, i1, pos1, id1, l1, m1, P_mi, tci)
                        i1 += 1
                if self.gamma:
                    nucleus1.P_kmi[0] += P_mi
                else:
                    phase_k = npy.exp(-2j * pi * npy.dot(self.ibzk_kc, R_c))
                    for k in range(nkpts):
                        nucleus1.P_kmi[k] += P_mi * phase_k[k]
                    
        self.S_kmm = npy.zeros((nkpts, self.nao, self.nao), self.dtype)
        self.T_kmm = npy.zeros((nkpts, self.nao, self.nao), self.dtype)
        S_mm = npy.zeros((self.nao, self.nao), self.dtype)
        T_mm = npy.zeros((self.nao, self.nao), self.dtype)

        for R_c in R_dc:
            i1 = 0
            for nucleus1 in self.nuclei:
                pos1 = nucleus1.spos_c
                setup1 = nucleus1.setup
                for j1, phit1 in enumerate(setup1.phit_j):
                    id1 = (setup1.symbol, j1)
                    l1 = phit1.get_angular_momentum_number()
                    for m1 in range(2 * l1 + 1):
                        self.st_overlap(R_c, i1, pos1, id1,
                                        l1, m1, S_mm, T_mm, tci)
                        i1 += 1
            if self.gamma:
                self.S_kmm[0] += S_mm
                self.T_kmm[0] += T_mm
            else:
                phase_k = npy.exp(2j * pi * npy.dot(self.ibzk_kc, R_c))
                for k in range(nkpts):            
                    self.S_kmm[k] += S_mm * phase_k[k]
                    self.T_kmm[k] += T_mm * phase_k[k]


        for nucleus in self.nuclei:
            dO_ii = nucleus.setup.O_ii
            for S_mm, P_mi in zip(self.S_kmm, nucleus.P_kmi):
                S_mm += npy.dot(P_mi, npy.inner(dO_ii, P_mi).conj())

        # Debug stuff        
        if 0:
            print 'Hamiltonian S_kmm[0] diag'
            print self.S_kmm[0].diagonal()
            print 'Hamiltonian S_kmm[0]'
            for row in self.S_kmm[0]:
                print ' '.join(['%02.03f' % f for f in row])
            print 'Eigenvalues:'    
            print npy.linalg.eig(self.S_kmm[0])[0]

        self.initialized = True

    def p_overlap(self, R_c, i1, pos1, id1, l1, m1, P_mi, tci):
        i2 = 0
        for nucleus2 in self.nuclei:
            pos2 = nucleus2.spos_c
            d = (R_c + pos1 - pos2) * self.gd.domain.cell_c
            setup2 = nucleus2.setup
            for j2, phit2 in enumerate(setup2.phit_j):
                id2 = (setup2.symbol, j2)
                l2 = phit2.get_angular_momentum_number()
                for m2 in range(2 * l2 + 1):
                    P = tci.p_overlap(id1, id2, l1, l2, m1, m2, d)
                    P_mi[i2, i1] = P
                    i2 += 1

    def st_overlap(self, R_c, i1, pos1, id1, l1, m1, S_mm, T_mm, tci):
       i2 = 0
       for nucleus2 in self.nuclei:
           pos2 = nucleus2.spos_c
           d = (pos1 - pos2 + R_c) * self.gd.domain.cell_c
           setup2 = nucleus2.setup
           for j2, phit2 in enumerate(setup2.phit_j):
               id2 = (setup2.symbol, j2)
               l2 = phit2.get_angular_momentum_number()
               for m2 in range(2 * l2 + 1):
                   S, T = tci.st_overlap(id1, id2, l1, l2, m1, m2, d)
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
        
    def calculate_effective_potential_matrix(self, Vt_skmm):
        Vt_skmm[:] = 0.0
        
        # Count number of boxes:
        nb = 0
        for nucleus in self.nuclei:
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
            if debug:	
                box_b = [box.lfs for box in phit_i.box_b]
            else:	
                box_b = phit_i.box_b
            b2 = b1 + len(box_b)
            m_b[b1:b2] = m
            lfs_b.extend(box_b)
            if not self.gamma:
                phase_bk[b1:b2] = phit_i.phase_kb.T
            m += phit_i.ni
            b1 = b2

        overlap(lfs_b, m_b, phase_bk, self.vt_sG, Vt_skmm)

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
