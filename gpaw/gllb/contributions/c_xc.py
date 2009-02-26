from gpaw.gllb.contributions.contribution import Contribution
from gpaw.xc_functional import XCRadialGrid, XC3DGrid, XCFunctional
from gpaw.xc_correction import A_Liy
import numpy as npy
from numpy import dot as dot3  # Avoid dotblas bug!
from math import pi, sqrt

class C_XC(Contribution):
    def __init__(self, nlfunc, weight, functional = 'LDA'):
        Contribution.__init__(self, nlfunc, weight)
        self.functional = functional

    def get_name(self):
        return 'XC'

    def get_desc(self):
        return "("+self.functional+")"
        
    def initialize(self):
        self.xc = XC3DGrid(XCFunctional(self.functional), self.nlfunc.finegd, self.nlfunc.nspins)
        self.vt_sg = self.nlfunc.finegd.empty(self.nlfunc.nspins)
        self.e_g = self.nlfunc.finegd.empty()

    def initialize_1d(self):
        self.ae = self.nlfunc.ae
        self.xc = XCRadialGrid(self.functional, self.ae.rgd) 
        self.v_g = npy.zeros(self.ae.N)

    def calculate_spinpaired(self, e_g, n_g, v_g):
        self.e_g[:] = 0.0
        self.vt_sg[:] = 0.0
        self.xc.get_energy_and_potential_spinpaired(n_g, self.vt_sg[0], e_g=self.e_g)
        v_g += self.weight * self.vt_sg[0]
        e_g += (self.weight * self.e_g).ravel()

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        self.e_g[:] = 0.0
        self.vt_sg[:] = 0.0
        self.xc.get_energy_and_potential(na_g, self.vt_sg[0], nb_g, self.vt_sg[1], e_g=self.e_g)
        va_g += self.weight * self.vt_sg[0]
        vb_g += self.weight * self.vt_sg[1]
        e_g += (self.weight * self.e_g).ravel()

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        # Get the XC-correction instance
        c = self.nlfunc.setups[a].xc_correction

        assert self.nlfunc.nspins == 1
        xcfunc = self.xc.xcfunc
        D_p = D_sp[0]
        dEdD_p = H_sp[0][:]
        D_Lq = dot3(c.B_Lqp, D_p)
        n_Lg = npy.dot(D_Lq, c.n_qg)
        n_Lg[0] += c.nc_g * sqrt(4 * pi)
        nt_Lg = npy.dot(D_Lq, c.nt_qg)
        nt_Lg[0] += c.nct_g * sqrt(4 * pi)
        dndr_Lg = npy.zeros((c.Lmax, c.ng))
        dntdr_Lg = npy.zeros((c.Lmax, c.ng))
        for L in range(c.Lmax):
            c.rgd.derivative(n_Lg[L], dndr_Lg[L])
            c.rgd.derivative(nt_Lg[L], dntdr_Lg[L])
                                                            
        E = 0
        vt_g = npy.zeros(c.ng)
        v_g = npy.zeros(c.ng)
        e_g = npy.zeros(c.ng)
        y = 0
        for w, Y_L in zip(c.weights, c.Y_yL):
            A_Li = A_Liy[:c.Lmax, :, y]
            a1x_g = npy.dot(A_Li[:, 0], n_Lg)
            a1y_g = npy.dot(A_Li[:, 1], n_Lg)
            a1z_g = npy.dot(A_Li[:, 2], n_Lg)
            a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
            a2_g[1:] /= c.rgd.r_g[1:]**2
            a2_g[0] = a2_g[1]
            a1_g = npy.dot(Y_L, dndr_Lg)
            a2_g += a1_g**2
            deda2_g = npy.zeros(c.ng)  

            v_g[:] = 0.0
            e_g[:] = 0.0
            n_g = npy.dot(Y_L, n_Lg)
            xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
            
            E += w * npy.dot(e_g, c.rgd.dv_g)
            x_g = -2.0 * deda2_g * c.rgd.dv_g * a1_g
            c.rgd.derivative2(x_g, x_g)
            x_g += v_g * c.rgd.dv_g
            dEdD_p += self.weight * w * npy.dot(dot3(c.B_pqL, Y_L),
                                  npy.dot(c.n_qg, x_g))
            x_g = 8.0 * pi * deda2_g * c.rgd.dr_g
            dEdD_p += w * npy.dot(dot3(c.B_pqL,
                                       A_Li[:, 0]),
                                  npy.dot(c.n_qg, x_g * a1x_g))
            dEdD_p += w * npy.dot(dot3(c.B_pqL,
                                       A_Li[:, 1]),
                                  npy.dot(c.n_qg, x_g * a1y_g))
            dEdD_p += w * npy.dot(dot3(c.B_pqL,
                                       A_Li[:, 2]),
                                  npy.dot(c.n_qg, x_g * a1z_g))

            n_g = npy.dot(Y_L, nt_Lg)
            a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
            a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
            a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
            a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
            a2_g[1:] /= c.rgd.r_g[1:]**2
            a2_g[0] = a2_g[1]
            a1_g = npy.dot(Y_L, dntdr_Lg)
            a2_g += a1_g**2
            v_g = npy.zeros(c.ng)
            e_g = npy.zeros(c.ng)
            deda2_g = npy.zeros(c.ng)

            v_g[:] = 0.0
            e_g[:] = 0.0
            xcfunc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)

            E -= w * npy.dot(e_g, c.dv_g)
            x_g = -2.0 * deda2_g * c.dv_g * a1_g
            c.rgd.derivative2(x_g, x_g)
            x_g += v_g * c.dv_g
                                                                                                                                                            
            B_Lqp = c.B_Lqp
            dEdD_p -= w * npy.dot(dot3(c.B_pqL, Y_L),
                                  npy.dot(c.nt_qg, x_g))
            x_g = 8.0 * pi * deda2_g * c.rgd.dr_g
            dEdD_p -= w * npy.dot(dot3(c.B_pqL,
                                       A_Li[:, 0]),
                                  npy.dot(c.nt_qg, x_g * a1x_g))
            dEdD_p -= w * npy.dot(dot3(c.B_pqL,
                                       A_Li[:, 1]),
                                  npy.dot(c.nt_qg, x_g * a1y_g))
            
            dEdD_p -= w * npy.dot(dot3(c.B_pqL,
                                       A_Li[:, 2]),
                                  npy.dot(c.nt_qg, x_g * a1z_g))
            
            y += 1
        
        return (E) * self.weight

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
        
