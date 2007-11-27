from math import sqrt

import Numeric as num
from gpaw.hamiltonian import Hamiltonian
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack
from gpaw.lcao.overlap import TwoCenterIntegrals
from gpaw import debug


class LCAOHamiltonian(Hamiltonian):
    """Hamiltonian class for LCAO-basis calculations"""

    def __init__(self, paw):
        Hamiltonian.__init__(self, paw)
        self.setups = paw.setups
        
    def initialize(self):
        self.nao = 0
        for nucleus in self.nuclei:
            nucleus.initialize_atomic_orbitals(self.gd, 42, None)
            self.nao += nucleus.get_number_of_atomic_orbitals()

        tci = TwoCenterIntegrals(self.setups)

        for nucleus1 in self.nuclei:
            i1 = 0
            setup1 = nucleus1.setup
            ni1 = nucleus1.get_number_of_partial_waves()
            nucleus1.P_mi = num.zeros((self.nao, ni1), num.Float)
            for j1, pt1 in enumerate(setup1.pt_j):
                id1 = (setup1.symbol, j1)
                l1 = pt1.get_angular_momentum_number()
                for m1 in range(2 * l1 + 1):
                    i2 = 0
                    for nucleus2 in self.nuclei:
                        pos1 = nucleus1.spos_c
                        pos2 = nucleus2.spos_c
                        R = (pos1 - pos2) * self.gd.domain.cell_c
                        setup2 = nucleus2.setup
                        for j2, phit2 in enumerate(setup2.phit_j):
                            id2 = (setup2.symbol, j2)
                            l2 = phit2.get_angular_momentum_number()
                            for m2 in range(2 * l2 + 1):
                                P = tci.p_overlap(id1, id2, l1, l2, m1, m2, R)
                                nucleus1.P_mi[i2, i1] = P
                                i2 += 1
                    i1 += 1

        T_mm = num.zeros((self.nao, self.nao), num.Float)
        S_mm = num.zeros((self.nao, self.nao), num.Float)
        i1 = 0
        for nucleus1 in self.nuclei:
            setup1 = nucleus1.setup
            for j1, phit1 in enumerate(setup1.phit_j):
                id1 = (setup1.symbol, j1)
                l1 = phit1.get_angular_momentum_number()
                for m1 in range(2 * l1 + 1):
                    i2 = 0
                    for nucleus2 in self.nuclei:
                        pos1 = nucleus1.spos_c
                        pos2 = nucleus2.spos_c
                        R = (pos1 - pos2) * self.gd.domain.cell_c
                        setup2 = nucleus2.setup
                        for j2, phit2 in enumerate(setup2.phit_j):
                            id2 = (setup2.symbol, j2)
                            l2 = phit2.get_angular_momentum_number()
                            for m2 in range(2 * l2 + 1):
                                S, T = tci.st_overlap(id1, id2, l1, l2,
                                                   m1, m2, R)
                                S_mm[i1, i2] = S
                                T_mm[i1, i2] = T
                                i2 += 1
                    i1 += 1

        for nucleus in self.nuclei:
            S_mm += num.dot(num.dot(nucleus.P_mi, nucleus.setup.O_ii),
                            num.transpose(nucleus.P_mi))

        self.S_mm = S_mm
        self.T_mm = T_mm

        #self.old_initialize()
        
    def calculate_effective_potential_matrix(self, V_mm):
        box_b = []
        for nucleus in self.nuclei:
            if debug:
                box_b.append(nucleus.phit_i.box_b[0].lfs)
            else:
                box_b.extend(nucleus.phit_i.box_b)
        assert len(box_b) == len(self.nuclei)
        from _gpaw import overlap
        overlap(box_b, self.vt_sG[0], V_mm)

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

        phi_mG = self.phi_mG
        H_mm = num.zeros((self.nao, self.nao), num.Float)
        r2k(0.5 * self.gd.dv, phi_mG, self.vt_sG[0] * phi_mG, 0.0, H_mm)
        print H_mm
        H0_mm = num.zeros((self.nao, self.nao), num.Float)
        self.calculate_effective_potential_matrix(H0_mm)
        print H0_mm
        sdfkgh
        for nucleus in self.nuclei:
            ni = nucleus.get_number_of_partial_waves()
            nucleus.P_mi = num.zeros((self.nao, ni), num.Float)
            nucleus.pt_i.integrate(self.phi_mG, nucleus.P_mi)

        self.S_mm = num.zeros((self.nao, self.nao), num.Float)
        rk(self.gd.dv, self.phi_mG, 0.0, self.S_mm)
        
        # Filling up the upper triangle:
        for m in range(self.nao - 1):
            self.S_mm[m, m:] = self.S_mm[m:, m]

        for nucleus in self.nuclei:
            self.S_mm += num.dot(num.dot(nucleus.P_mi, nucleus.setup.O_ii),
                            num.transpose(nucleus.P_mi))
        self.T_mm = num.zeros((self.nao, self.nao), num.Float)
        Tphi_mG = self.gd.zeros(self.nao)
        self.kin.apply(self.phi_mG, Tphi_mG)
        r2k(0.5 * self.gd.dv, self.phi_mG, Tphi_mG, 0.0, self.T_mm)

        # Filling up the upper triangle:
        for m in range(self.nao - 1):
            self.T_mm[m, m:] = self.T_mm[m:, m]
