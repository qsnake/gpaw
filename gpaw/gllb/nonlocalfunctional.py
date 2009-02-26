from gpaw.xc_functional import ZeroFunctional

class NonLocalFunctional(ZeroFunctional):

    def __init__(self):
        self.contributions = []
        self.xcs = {}
        
    def pass_stuff(self, paw):
        self.gd = paw.density.gd # smooth grid describtor
        self.finegd = paw.density.finegd # fine grid describtor
        self.nt_sg = paw.density.nt_sg # smooth density
        self.setups = paw.wfs.setups # All the setups 
        self.nspins = paw.wfs.nspins # number of spins
        self.wfs = paw.wfs
        self.occupations = paw.occupations
        self.density = paw.density
        self.atoms = paw.atoms
        self.hamiltonian = paw.hamiltonian

        #self.vt_sg = paw.vt_sg # smooth potential
        #self.kpt_u = kpt_u # kpoints object       
        #self.interpolate = interpolate # interpolation function
        #self.nuclei = nuclei

        # Is this OK place?
        self.initialize()
        
    def pass_stuff_1d(self, ae):
        self.ae = ae

    def initialize(self):
        for contribution in self.contributions:
            contribution.initialize()

    def initialize_1d(self):
        for contribution in self.contributions:
            contribution.initialize_1d()
    
    def calculate_spinpaired(self, e_g, n_g, v_g):
        e_g[:] = 0.0
        for contribution in self.contributions:
            contribution.calculate_spinpaired(e_g, n_g, v_g)

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        e_g[:] = 0.0
        for contribution in self.contributions:
            contribution.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g)
            
    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        Exc = 0.0
        H_sp[:] = 0.0
        for contribution in self.contributions:
            Exc += contribution.calculate_energy_and_derivatives(D_sp, H_sp, a)
        Exc -= self.setups[a].xc_correction.Exc0
        return Exc

    def get_xc_potential_and_energy_1d(self, v_g):
        Exc = 0.0
        for contribution in self.contributions:
            Exc += contribution.add_xc_potential_and_energy_1d(v_g)
        return Exc
    
    def get_smooth_xc_potential_and_energy_1d(self, vt_g):
        Exc = 0.0
        for contribution in self.contributions:
            Exc += contribution.add_smooth_xc_potential_and_energy_1d(vt_g)
        return Exc

    def initialize_from_atomic_orbitals(self, basis_functions):
        for contribution in self.contributions:
            contribution.initialize_from_atomic_orbitals(basis_functions)

    def get_extra_setup_data(self, dict):
        for contribution in self.contributions:
            contribution.add_extra_setup_data(dict)

    def add_contribution(self, contribution):
        self.contributions.append(contribution)
        self.xcs[contribution.get_name()] = contribution

    def print_functional(self):
        print
        print "Functional being used consists of"
        print "---------------------------------------------------"
        print "| Weight    | Module           | Description      |"
        print "---------------------------------------------------"
        for contribution in self.contributions:
            print "|%9.3f  | %-17s| %-17s|" % (contribution.weight, contribution.get_name(), contribution.get_desc())
        print "---------------------------------------------------"
        print

    def read(self, reader):
        for contribution in self.contributions:
            contribution.read(reader)

    def write(self, writer):
        for contribution in self.contributions:
            contribution.write(writer)     
