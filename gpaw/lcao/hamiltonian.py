import Numeric as num
from gpaw.hamiltonian import Hamiltonian
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack


class LCAOHamiltonian(Hamiltonian):
    """Hamiltonian class for LCAO-basis calculations"""
    
    def initialize(self):
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

    def calculate_effective_potential_matrix(self, V_mm):
        box_b = []
        for nucleus in self.nuclei:
            box_b.extend(nucleus.phit_i.box_b)
        assert len(box_b) == len(self.nuclei)
        from _gpaw import overlap
        from time import time as t
        t0 = t()
        overlap(box_b, self.vt_sG[0], V_mm)
        t1 = t()
        print t1 - t0
