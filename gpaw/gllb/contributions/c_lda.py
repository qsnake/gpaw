from gpaw.gllb.contributions.contribution import Contribution
from gpaw.xc_functional import XCRadialGrid, XCFunctional
import numpy as npy
from numpy import dot as dot3  # Avoid dotblas bug!
from math import pi, sqrt

class C_LDA(Contribution):
    def __init__(self, nlfunc, weight, functional = 'LDA'):
        Contribution.__init__(self, nlfunc, weight)
        self.functional = functional
    def get_name(self):
        return 'LDA'
        
    def initialize(self):
        self.xc = XCFunctional(self.functional)
        self.vt_sg = self.nlfunc.finegd.empty(self.nlfunc.nspins)
        self.e_g = self.nlfunc.finegd.empty()#.ravel()

    def initialize_1d(self):
        self.ae = self.nlfunc.ae
        self.xc = XCRadialGrid(self.functional, self.ae.rgd) 
        self.v_g = npy.zeros(self.ae.N)

    def calculate_spinpaired(self, e_g, n_g, v_g):
        self.e_g[:] = 0.0
        self.vt_sg[:] = 0.0
        self.xc.calculate_spinpaired(self.e_g, n_g, self.vt_sg[0])
        v_g += self.weight * self.vt_sg[0]
        e_g += (self.weight * self.e_g).ravel()

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        self.e_g[:] = 0.0
        self.vt_sg[:] = 0.0
        self.xc.calculate_spinpolarized(self.e_g, na_g, self.vt_sg[0], nb_g, self.vt_sg[0])
        va_g += self.weight * self.vt_sg[0]
        vb_g += self.weight * self.vt_sg[1]
        e_g += (self.weight * self.e_g).ravel()

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        # Get the XC-correction instance
        print a
        c = self.nlfunc.setups[a].xc_correction

        assert self.nlfunc.nspins == 1

        D_p = D_sp[0]
        dEdD_p = H_sp[0][:]
        D_Lq = dot3(c.B_Lqp, D_p)
        n_Lg = npy.dot(D_Lq, c.n_qg)
        n_Lg[0] += c.nc_g * sqrt(4 * pi)
        nt_Lg = npy.dot(D_Lq, c.nt_qg)
        nt_Lg[0] += c.nct_g * sqrt(4 * pi)
        E = 0
        vt_g = npy.zeros(c.ng)
        v_g = npy.zeros(c.ng)
        e_g = npy.zeros(c.ng)
        for w, Y_L in zip(c.weights, c.Y_yL):
            nt_g = npy.dot(Y_L, nt_Lg)
            vt_g[:] = 0.0
            e_g[:] = 0.0
            self.xc.calculate_spinpaired(e_g, nt_g, vt_g)
            dEdD_p -= self.weight * w * npy.dot(dot3(c.B_pqL, Y_L),
                                  npy.dot(c.nt_qg, vt_g * c.rgd.dv_g))

            E -= w * npy.dot(e_g, c.rgd.dv_g)
            n_g = npy.dot(Y_L, n_Lg)
            v_g[:] = 0.0
            e_g[:] = 0.0
            self.xc.calculate_spinpaired(e_g, n_g, v_g)
            dEdD_p += self.weight * w * npy.dot(dot3(c.B_pqL, Y_L),
                                  npy.dot(c.n_qg, v_g * c.rgd.dv_g))
            E += w * npy.dot(e_g, c.rgd.dv_g)
            
        return (E - c.Exc0) * self.weight

    def add_xc_potential_and_energy_1d(self, v_g):
        self.v_g[:] = 0.0
        Exc = self.xc.get_energy_and_potential(self.ae.n, self.v_g)
        v_g += self.weight * self.v_g
        return self.weight * Exc

    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        self.v_g[:] = 0.0
        Exc = self.xc.get_energy_and_potential(self.ae.nt, self.v_g)
        vt_g += self.weight * self.v_g
        return self.weight * Exc

    def initialize_from_atomic_orbitals(self, basis_functions):
        # LDA needs only density, which is already initialized
        pass

    def add_extra_setup_data(self, dict):
        # LDA has not any special data
        pass

    def write(self, writer):
        # LDA has not any special data to be written
        pass

    def read(self, reader):
        # LDA has not any special data to be read
        pass
        
