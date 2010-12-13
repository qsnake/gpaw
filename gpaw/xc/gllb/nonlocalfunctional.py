from gpaw.xc.functional import XCFunctional
from gpaw.mpi import world

class NonLocalFunctional(XCFunctional):
    type = 'GLLB'
    def __init__(self, xcname):
        self.contributions = []
        self.xcs = {}
        XCFunctional.__init__(self, xcname)
    
    def initialize(self, density, hamiltonian, wfs, occupations):
        self.gd = density.gd # smooth grid describtor
        self.finegd = density.finegd # fine grid describtor
        self.nt_sg = density.nt_sg # smooth density
        self.setups = wfs.setups # All the setups 
        self.nspins = wfs.nspins # number of spins
        self.wfs = wfs
        self.occupations = occupations
        self.density = density
        self.hamiltonian = hamiltonian
        self.nvalence = wfs.nvalence

        #self.vt_sg = paw.vt_sg # smooth potential
        #self.kpt_u = kpt_u # kpoints object       
        #self.interpolate = interpolate # interpolation function
        #self.nuclei = nuclei

        # Is this OK place?
        self.initialize0()
        
    def pass_stuff_1d(self, ae):
        self.ae = ae

    def initialize0(self):
        for contribution in self.contributions:
            contribution.initialize()

    def initialize_1d(self):
        for contribution in self.contributions:
            contribution.initialize_1d()

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        #if gd is not self.gd:
        #    self.set_grid_descriptor(gd)
        if e_g is None:
            e_g = gd.empty()
        if v_sg is None:
            v_sg = np.zeros_like(n_sg)
        if self.nspins == 1:
            self.calculate_spinpaired(e_g, n_sg[0], v_sg[0])
        else:
            dsfsdfg
        return gd.integrate(e_g)
    
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
        if world.rank is not 0:
            return
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

    def write(self, writer, natoms):
        for contribution in self.contributions:
            contribution.write(writer, natoms)     
