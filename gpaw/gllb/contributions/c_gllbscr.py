from gpaw.gllb.contributions.contribution import Contribution
from gpaw.xc_functional import XCRadialGrid, XCFunctional, XC3DGrid
from gpaw.xc_correction import A_Liy
from gpaw.gllb import safe_sqr
from math import sqrt, pi
from gpaw.io.tar import TarFileReference
import numpy as npy

K_G = 0.382106112167171

class C_GLLBScr(Contribution):
    def __init__(self, nlfunc, weight, functional = 'X_B88-None'):
        Contribution.__init__(self, nlfunc, weight)
        self.functional = functional
        self.old_coeffs = None
        self.iter = 0
        
    def get_name(self):
        return "SCREENING"

    def get_desc(self):
        return "("+self.functional+")"
        
    # Initialize GLLBScr functional
    def initialize_1d(self):
        self.ae = self.nlfunc.ae
        self.xc = XCRadialGrid(self.functional, self.ae.rgd) 
        self.v_g = npy.zeros(self.ae.N)
        self.e_g = npy.zeros(self.ae.N)

    # Calcualte the GLLB potential and energy 1d
    def add_xc_potential_and_energy_1d(self, v_g):
        self.v_g[:] = 0.0
        self.e_g[:] = 0.0
        self.xc.get_energy_and_potential_spinpaired(self.ae.n, self.v_g, e_g=self.e_g)
        v_g += 2 * self.weight * self.e_g / (self.ae.n + 1e-10)
        Exc = self.weight * npy.sum(self.e_g * self.ae.rgd.dv_g)
        return Exc

    def initialize(self):
        self.occupations = self.nlfunc.occupations
        self.xc = XCFunctional(self.functional)
        self.xc_grid3d = XC3DGrid(self.xc, self.nlfunc.finegd, self.nlfunc.nspins)
        self.xc_grid3d.allocate()
        self.vt_sg = self.nlfunc.finegd.empty(self.nlfunc.nspins)
        self.e_g = self.nlfunc.finegd.empty()#.ravel()

    def get_coefficient_calculator(self):
        return self

    def f(self, f):
        return sqrt(f)
    
    def get_coefficients_1d(self, smooth=False, lumo_perturbation = False):
        homo_e = max( [ npy.where(f>1e-3, e, -1000) for f,e in zip(self.ae.f_j, self.ae.e_j)]) 
        if not smooth:
            if lumo_perturbation:
                lumo_e = min( [ npy.where(f<1e-3, e, 1000) for f,e in zip(self.ae.f_j, self.ae.e_j)])
                return npy.array([ f * K_G * (self.f( max(0, lumo_e - e)) - self.f(max(0, homo_e -e)))
                                        for e,f in zip(self.ae.e_j, self.ae.f_j) ])
            else:
                return npy.array([ f * K_G * (self.f( max(0, homo_e - e)))
                                   for e,f in zip(self.ae.e_j, self.ae.f_j) ])
        else:
            return [ [ f * K_G * self.f( max(0, homo_e - e))
                    for e,f in zip(e_n, f_n) ]
                     for e_n, f_n in zip(self.ae.e_ln, self.ae.f_ln) ]
        

    def get_coefficients_by_kpt(self, kpt_u, lumo_perturbation = False):
        if kpt_u[0].psit_nG is None or isinstance(kpt_u[0].psit_nG, TarFileReference): 
            return None

        e_ref = self.occupations.get_zero_kelvin_homo_eigenvalue(kpt_u)

        # The parameter ee might sometimes be set to small thereshold value to
        # achieve convergence on systems with degenerate HOMO.
        ee = 0.0

        if lumo_perturbation:
            e_ref_lumo = self.occupations.get_zero_kelvin_lumo_eigenvalue(kpt_u)
            return [ npy.array([
                f * K_G * (self.f( npy.where(e_ref_lumo - e>ee, e_ref_lumo-e,0))
                         -self.f( npy.where(e_ref      - e>ee, e_ref-e,0)))
                     for e, f in zip(kpt.eps_n, kpt.f_n) ])
                     for kpt in kpt_u ]
            
            
        else:
            # Mix the coefficients with 25%
            coeff = [ npy.array([ f * K_G * self.f( npy.where(e_ref - e>ee, e_ref-e,0))
                     for e, f in zip(kpt.eps_n, kpt.f_n) ])
                     for kpt in kpt_u ]
            if self.old_coeffs is None:
                self.old_coeffs = coeff
            else:
                mix = 1.0
                #if self.iter > 7:
                self.old_coeffs = [ (1-mix) * old + mix * new for old, new in zip(coeff, self.old_coeffs) ]
                    
            return self.old_coeffs
        

    def calculate_spinpaired(self, e_g, n_g, v_g):
        self.e_g[:] = 0.0
        self.vt_sg[:] = 0.0
        self.xc_grid3d.get_energy_and_potential_spinpaired(n_g, self.vt_sg[0], e_g = self.e_g)
        v_g += self.weight * 2 * self.e_g / (n_g + 1e-10)
        e_g += self.weight * self.e_g.ravel()

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g, 
                                a2_g=None, aa2_g=None, ab2_g=None, deda2_g=None,
                                dedaa2_g=None, dedab2_g=None):
        raise NotImplementedError

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        # Get the XC-correction instance
        c = self.nlfunc.setups[a].xc_correction

        assert self.nlfunc.nspins == 1

        D_p = D_sp[0]
        dEdD_p = H_sp[0][:]
        D_Lq = npy.dot(c.B_Lqp, D_p)
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
        deda2_g = npy.zeros(c.ng)
        for y, (w, Y_L) in enumerate(zip(c.weights, c.Y_yL)):
            # Cut gradient releated coefficient to match the setup's Lmax
            A_Li = A_Liy[:c.Lmax, :, y]

            # Expand pseudo density
            nt_g = npy.dot(Y_L, nt_Lg)

            # Expand pseudo density gradient
            a1x_g = npy.dot(A_Li[:, 0], nt_Lg)
            a1y_g = npy.dot(A_Li[:, 1], nt_Lg)
            a1z_g = npy.dot(A_Li[:, 2], nt_Lg)
            a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
            a2_g[1:] /= c.rgd.r_g[1:]**2
            a2_g[0] = a2_g[1]
            a1_g = npy.dot(Y_L, dntdr_Lg)
            a2_g += a1_g**2
            
            vt_g[:] = 0.0
            e_g[:] = 0.0
            # Calculate pseudo GGA energy density (potential is discarded)
            self.xc.calculate_spinpaired(e_g, nt_g, vt_g, a2_g, deda2_g)


            # Calculate pseudo GLLB-potential from GGA-energy density
            vt_g[:] = 2 * e_g / (nt_g + 1e-10)

            
            dEdD_p -= self.weight * w * npy.dot(npy.dot(c.B_pqL, Y_L),
                                  npy.dot(c.nt_qg, vt_g * c.rgd.dv_g))

            E -= w * npy.dot(e_g, c.rgd.dv_g)
            
            # Expand density
            n_g = npy.dot(Y_L, n_Lg)

            # Expand density gradient
            a1x_g = npy.dot(A_Li[:, 0], n_Lg)
            a1y_g = npy.dot(A_Li[:, 1], n_Lg)
            a1z_g = npy.dot(A_Li[:, 2], n_Lg)
            a2_g = a1x_g**2 + a1y_g**2 + a1z_g**2
            a2_g[1:] /= c.rgd.r_g[1:]**2
            a2_g[0] = a2_g[1]
            a1_g = npy.dot(Y_L, dndr_Lg)
            a2_g += a1_g**2
            
            v_g[:] = 0.0
            e_g[:] = 0.0
            # Calculate GGA energy density (potential is discarded)
            self.xc.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)

            # Calculate GLLB-potential from GGA-energy density
            v_g[:] = 2 * e_g / (n_g + 1e-10)
            
            dEdD_p += self.weight * w * npy.dot(npy.dot(c.B_pqL, Y_L),
                                  npy.dot(c.n_qg, v_g * c.rgd.dv_g))
            E += w * npy.dot(e_g, c.rgd.dv_g)
            
        return (E) * self.weight

    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        self.v_g[:] = 0.0
        self.e_g[:] = 0.0
        self.xc.get_energy_and_potential_spinpaired(self.ae.nt, self.v_g, e_g=self.e_g)
        vt_g += 2 * self.weight * self.e_g / (self.ae.nt + 1e-10)
        return self.weight * npy.sum(self.e_g * self.ae.rgd.dv_g)

    def initialize_from_atomic_orbitals(self, basis_functions):
        # GLLBScr needs only density which is already initialized
        pass
        
    def add_extra_setup_data(self, dict):
        # GLLBScr has not any special data
        pass

    def read(self, reader):
        # GLLBScr has no special data to be read
        pass

    def write(self, writer):
        # GLLBScr has no special data to be written
        pass
        


