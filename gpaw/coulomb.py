from math import pi

import numpy as npy
from numpy.fft import fftn

from ase.units import Hartree
from pair_density import PairDensity2 as PairDensity
from gpaw.poisson import PoissonSolver
from gpaw.utilities import pack, unpack
from gpaw.utilities.tools import pick, construct_reciprocal, dagger
from gpaw.utilities.complex import real
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.blas import r2k


def get_vxc(paw, spin):
    """Calculate matrix elements of the xc-potential."""
    psit_nG = paw.kpt_u[spin].psit_nG[:]
    nt_g = paw.density.nt_sg[spin]
    vxct_g = paw.finegd.empty()
    paw.hamiltonian.xc.get_energy_and_potential(nt_g, vxct_g)
    vxct_G = paw.gd.empty()
    paw.hamiltonian.restrict(vxct_g, vxct_G)
    Vxc_nn = npy.zeros((paw.nbands, paw.nbands))
    r2k(0.5 * paw.gd.dv, psit_nG, vxct_G * psit_nG, 0.0, Vxc_nn)
    for nucleus in paw.my_nuclei:
        D_sp = nucleus.D_sp
        H_sp = 0.0 * D_sp
        nucleus.setup.xc_correction.calculate_energy_and_derivatives(
            D_sp, H_sp)
        H_ii = unpack(H_sp[spin])
        P_ni = nucleus.P_uni[spin]
        Vxc_nn += npy.dot(P_ni, npy.dot(H_ii, P_ni.T))
    return Vxc_nn * paw.Ha


class Coulomb:
    """Class used to evaluate coulomb integrals"""
    def __init__(self, gd, poisson=None):
        """Class should be initialized with a grid_descriptor 'gd' from
           the gpaw module.
        """        
        self.gd = gd
        self.poisson = poisson

    def load(self, method):
        """Make sure all necessary attributes have been initialized"""
        
        # ensure that requested method is valid
        assert method in ('real', 'recip_gauss', 'recip_ewald'),\
            str(method) + ' is an invalid method name,\n' +\
            'use either real, recip_gauss, or recip_ewald'

        if method.startswith('recip'):
            if self.gd.comm.size > 1:
                raise RuntimeError('Cannot do parallel FFT, ' +\
                                   'use method=\'real\'')
            if not hasattr(self, 'k2'):
                self.k2, self.N3 = construct_reciprocal(self.gd)
                
            if method.endswith('ewald') and not hasattr(self, 'ewald'):
                # cutoff radius
                rc = 0.5 * npy.average(self.gd.domain.cell_c)
                # ewald potential: 1 - cos(k rc)
                self.ewald = (npy.ones(self.gd.n_c) - 
                              npy.cos(npy.sqrt(self.k2) * rc))
                # lim k -> 0 ewald / k2 
                self.ewald[0, 0, 0] = 0.5 * rc**2

            if method.endswith('gauss') and not hasattr(self, 'ng'):
                gauss = Gaussian(self.gd)
                self.ng = gauss.get_gauss(0) / npy.sqrt(4 * pi)
                self.vg = gauss.get_gauss_pot(0) / npy.sqrt(4 * pi)
        
        else: # method == 'real'
            if not hasattr(self, 'solve'):
                if self.poisson is not None:
                    self.solve = self.poisson.solve
                else:
                    solver = PoissonSolver(nn=2)
                    solver.initialize(self.gd, load_gauss=True)
                    self.solve = solver.solve


    def coulomb(self, n1, n2=None, Z1=None, Z2=None, method='recip_gauss'):
        """Evaluates the coulomb integral of n1 and n2

        The coulomb integral is defined by::

                                      *
                      /    /      n1(r)  n2(r')
          (n1 | n2) = | dr | dr'  -------------,
                      /    /         |r - r'|
                      
        where n1 and n2 could be complex.

        real:
           Evaluate directly in real space using gaussians to neutralize
           density n2, such that the potential can be generated by standard
           procedures
              
        recip_ewald:
           Evaluate by Fourier transform.
           Divergence at division by k^2 is avoided by utilizing the Ewald /
           Tuckermann trick, which formaly requires the densities to be
           localized within half of the unit cell.

        recip_gauss:
           Evaluate by Fourier transform.
           Divergence at division by k^2 is avoided by removing total charge
           of n1 and n2 with gaussian density ng::
           
                                                   *          *    *
            (n1|n2) = (n1 - Z1 ng|n2 - Z2 ng) + (Z2 n1 + Z1 n2 - Z1 Z2 ng | ng)

           The evaluation of the integral (n1 - Z1 ng|n2 - Z2 ng) is done in
           k-space using FFT techniques.
        """
        self.load(method)
        # determine integrand using specified method
        if method == 'real':
            I = self.gd.zeros()
            if n2 == None: n2 = n1; Z2 = Z1
            self.solve(I, n2, charge=Z2, eps=1e-12, zero_initial_phi=True)
            I *= npy.conjugate(n1)           
        elif method == 'recip_ewald':
            n1k = fftn(n1)
            if n2 == None: n2k = n1k
            else: n2k = fftn(n2)
            I = npy.conjugate(n1k) * n2k * \
                self.ewald * 4 * pi / (self.k2 * self.N3)
        elif method == 'recip_gauss':
            # Determine total charges
            if Z1 == None: Z1 = self.gd.integrate(n1)
            if Z2 == None and n2 != None: Z2 = self.gd.integrate(n2)

            # Determine the integrand of the neutral system
            # (n1 - Z1 ng)* int dr'  (n2 - Z2 ng) / |r - r'|
            nk1 = fftn(n1 - Z1 * self.ng)
            if n2 == None:
                I = npy.absolute(nk1)**2 * 4 * pi / (self.k2 * self.N3)
            else:
                nk2 = fftn(n2 - Z2 * self.ng)
                I = npy.conjugate(nk1) * nk2 * 4 * pi / (self.k2 * self.N3)

            # add the corrections to the integrand due to neutralization
            if n2 == None:
                I += (2 * real(npy.conjugate(Z1) * n1) - abs(Z1)**2 * self.ng)\
                     * self.vg
            else:
                I += (npy.conjugate(Z1) * n2 + Z2 * npy.conjugate(n1) -
                      npy.conjugate(Z1) * Z2 * self.ng) * self.vg
        else:
             raise RuntimeError, 'Method %s unknown' % method
         
        if n1.dtype.char == float and (n2 == None or
                                           n2.dtype.char == float):
            return real(self.gd.integrate(I))
        else:
            return self.gd.integrate(I)


class Coulomb4:
    """Determine four-index Coulomb integrals::

                                             *
                            /    /      rho12(r) rho34(r')
          (n1 n2 | n3 n4) = | dr | dr'  ------------------,
                            /    /            |r - r'|
                                                *     *
                            /    /      w1(r) w2(r) w3(r') w4(r')
                          = | dr | dr'  -------------------------,
                            /    /               |r - r'|

    where::

                       *
          rho12(r) = w1(r) w2(r)
    """
    def __init__(self, paw, spin, method='real'):
        self.kpt = paw.kpt_u[spin]
        self.pd = PairDensity(paw, finegrid=True)
        self.nt12_G = paw.gd.empty()
        self.nt34_G = paw.gd.empty()
        self.rhot12_g = paw.finegd.empty()
        self.rhot34_g = paw.finegd.empty()
        self.psum = paw.gd.comm.sum
        self.my_nuclei = paw.my_nuclei
        self.u = spin

        coulomb = Coulomb(paw.finegd, poisson=paw.hamiltonian.poisson)
        coulomb.load(method)
        self.method = method
        self.coulomb = coulomb.coulomb
        
    def get_integral(self, n1, n2, n3, n4):
        rhot12_g = self.rhot12_g
        rhot12_g[:] = 0.0
        self.pd.initialize(self.kpt, n1, n2)
        self.pd.get_coarse(self.nt12_G)
        self.pd.add_compensation_charges(self.nt12_G, rhot12_g)
        
        if npy.all(n3 == n1) and npy.all(n4 == n2):
            rhot34_g = None
        else:
            rhot34_g = self.rhot34_g
            rhot34_g[:] = 0.0
            self.pd.initialize(self.kpt, n3, n4)
            self.pd.get_coarse(self.nt34_G)
            self.pd.add_compensation_charges(self.nt34_G, rhot34_g)

        # smooth part
        Z12 = float(npy.all(n1 == n2))
        Z34 = float(npy.all(n3 == n4))
        I = self.coulomb(rhot12_g, rhot34_g, Z12, Z34, self.method)
        
        # Add atomic corrections
        Ia = 0.0
        for nucleus in self.my_nuclei:
            #   ----
            # 2 >     P   P  C    P  P
            #   ----   1i  2j ijkl 3k 4l
            #   ijkl 
            P_ni = nucleus.P_uni[self.u]
            D12_p = pack(npy.outer(pick(P_ni, n1), pick(P_ni, n2)), 1e3)
            D34_p = pack(npy.outer(pick(P_ni, n3), pick(P_ni, n4)), 1e3)
            Ia += 2 * npy.dot(D12_p, npy.dot(nucleus.setup.M_pp, D34_p))
        #print I, Ia
        I += self.psum(Ia)

        return I


def wannier_coulomb_integrals(paw, U_nj, spin,
                              types=['xc',   # Local xc functional
                                     'ijij', # Direct
                                     'ijji', # Exchange
                                     'iijj', # iijj
                                     'iiij', # Semiexchange
                                     'ikjk', # Extra
                                     ]):
    # Returns some of the Coulomb integrals
    # V_{ijkl} = \iint drdr' / |r-r'| i*(r) j*(r') k(r) l(r')
    # using coulomb4, which determines
    # C4(ijkl) = \iint drdr' / |r-r'| i(r) j*(r) k*(r') l(r')
    # i.e. V_ijkl = C4(jkil)

    coulomb4 = Coulomb4(paw, spin).get_integral
    nwannier = U_nj.shape[1]
    if paw.dtype is complex or U_nj.dtype is complex:
        dtype = complex
    else:
        dtype = float
                          
    if 'xc' in types:
        V_xc = npy.dot(dagger(U_nj), npy.dot(get_vxc(paw, spin), U_nj))
    V_ijij = npy.zeros([nwannier, nwannier], dtype)
    V_ijji = npy.zeros([nwannier, nwannier], dtype)
    V_iijj = npy.zeros([nwannier, nwannier], dtype)
    V_iiij = npy.zeros([nwannier, nwannier], dtype)
    if 'ikjk' in types:
        V_ikjk = npy.zeros([nwannier, nwannier, nwannier], dtype)
    
    for i in range(nwannier):
        ni = U_nj[:, i]
        for j in range(i, nwannier):
            nj = U_nj[:, j]
            print "Doing Coulomb integrals for orbitals", i, j

            if 'ijij' in types:
                # V_{ij, ij} = C4(jiij)
                V_ijij[i, j] = coulomb4(nj, ni, ni, nj) * Hartree
                
            if 'ijji' in types:
                # V_{ij, ji} = C4(jjii)
                V_ijji[i, j] = coulomb4(nj, nj, ni, ni) * Hartree

            if 'iijj' in types:
                # V_{ii, jj} = C4(ijij)
                V_iijj[i, j] = coulomb4(ni, nj, ni, nj) * Hartree

            if 'iiij' in types:
                # V_{ii, ij} = C4(iiij)
                V_iiij[i, j] = coulomb4(ni, ni, ni, nj) * Hartree

                # V_{jj, ji} = C4(jjji)
                V_iiij[j, i] = coulomb4(nj, nj, nj, ni) * Hartree

            if 'ikjk' in types:
                for k in range(nwannier):
                    nk = U_nj[:, k]
                    # V_{ik, jk} = C4(kjik)
                    V_ikjk[i, j, k] = coulomb4(nk, nj, ni, nk) * Hartree

    # Fill out lower triangle of direct, exchange, and iijj elements
    for i in range(nwannier):
        for j in range(i):
            V_ijij[i, j] = V_ijij[j, i]
            V_ijji[i, j] = V_ijji[j, i]
            V_iijj[i, j] = V_iijj[j, i].conj()
            if 'ikjk' in types:
                V_ikjk[i,j,:] = V_ikjk[j,i,:].conj()

    result = ()
    for type in types:
        result += (eval('V_' + type), )
    return result

