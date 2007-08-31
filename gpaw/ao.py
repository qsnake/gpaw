import Numeric as num
from gpaw.utilities.blas import rk, r2k
from gpaw.utilities import unpack

class AtomicBasisSolver:
    def __init__(self, paw):
        self.gd = paw.gd
        self.nuclei = paw.nuclei
        self.initialized = False

    def initialize(self, hamiltonian):
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

        for nucleus in self.nuclei:
            self.S_mm += num.dot(num.dot(nucleus.P_mi, nucleus.setup.O_ii),
                            num.transpose(nucleus.P_mi))

        self.T_mm = num.zeros((self.nao, self.nao), num.Float)
        Tphi_mG = self.gd.zeros(self.nao)
        hamiltonian.kin.apply(self.phi_mG, Tphi_mG)
        r2k(0.5 * self.gd.dv, self.phi_mG, Tphi_mG, 0.0, self.T_mm)
        
        print self.S_mm
        print self.T_mm

        
        self.error = 1.0e-12
        self.initialized = True
    
    def iterate(self, hamiltonian, kpt_u):
        if not self.initialized:
            self.initialize(hamiltonian)

        vt_G = hamiltonian.vt_sG[0]
        H_mm = num.zeros((self.nao, self.nao), num.Float)
        r2k(0.5 * self.gd.dv, self.phi_mG, vt_G * self.phi_mG, 0.0, H_mm)
        
        for nucleus in self.nuclei:
            dH_ii = unpack(nucleus.H_sp[0])
            H_mm += num.dot(num.dot(nucleus.P_mi, dH_ii),
                            num.transpose(nucleus.P_mi))

        H_mm += self.T_mm

        print H_mm
