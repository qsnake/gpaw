class XCFunctional:
    orbital_dependent = False
    def __init__(self, name):
        self.name = name
        self.gd = None
        self.ekin = 0.0
        
    def get_setup_name(self):
        return self.name
    
    def initialize(self, density, hamiltonian, wfs, occupations):
        pass

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def set_positions(self, spos_ac):
        pass
    
    def summary(self, fd):
        pass

    def estimate_memory(self, mem):
        pass
    
    # Orbital dependent stuff:
    def apply_orbital_dependent_hamiltonian(self, kpt, psit_nG,
                                            Htpsit_nG, dH_asp):
        pass
    
    def correct_hamiltonian_matrix(self, kpt, H_nn):
        pass

    def add_correction(self, kpt, psit_xG, R_xG, P_axi, c_axi, n_x=None,
                       calculate_change=False):
        pass
    
    def rotate(self, kpt, U_nn):
        pass

    def get_kinetic_energy_correction(self):
        return self.ekin

    def add_forces(self, F_av):
        pass
