from math import pi, sqrt
from os.path import isfile

import numpy as np
#from scipy.special import sph_jn
from ase.units import Hartree, Bohr
from ase.data import chemical_symbols

from gpaw.gaunt import gaunt as G_LLL
from gpaw.spherical_harmonics import Y
from gpaw.setup_data import SetupData
from gpaw.setup import Setup
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import pack, unpack
from gpaw.utilities.blas import gemmdot
from gpaw.response import CHI


class PeriodicSys(CHI):
    def __init__(self):
        CHI.__init__(self)


    def get_optical_spectrum(self, calc, q, wcut, wmin, wmax, dw, eta=0.2, sigma=2*1e-5, OpticalLimit=True):
        """Calculate Optical absorption spectrum.

        The optical absorption spectra is defined as::

            ABS = Im \epsilon_M (q=0, w)
        """       
        
        epsilon0, epsilonRPA, epsilonLDA = self.get_dielectric_function(calc, q, wcut, wmin,
                                           wmax, dw, eta, sigma, OpticalLimit)

        return np.imag(epsilon0), np.imag(epsilonRPA), np.imag(epsilonLDA)


    def get_EELS_spectrum(self, calc, q, wcut, wmin, wmax, dw, eta=0.2, sigma=2*1e-5, OpticalLimit=False):
        """Calculate Electron Energy Loss Spectrum.

        Calculate EELS of a periodic system for a particular q. The 
        Loss function is related to::

                         -1            4 pi                              1        
            - Im \epsilon (q, w) = - -------  Im  chi (q, w)  = - Im ----------
                        G=0,G'=0      |q|**2        G=0,G'=0         \epsilon_M
        """

        epsilon0, epsilonRPA, epsilonLDA = self.get_dielectric_function(calc, q, wcut, wmin, 
                                           wmax, dw, eta, sigma, OpticalLimit)    

        return -np.imag(1./epsilon0),  -np.imag(1./epsilonRPA),  -np.imag(1./epsilonLDA)



    def get_dielectric_function(self, calc, q, wcut, wmin, wmax, dw, eta, sigma, OpticalLimit):
        """Calculate Macroscopic dielectric function.

        The macroscopic dielectric function is defined as::

                                        1
            \epsilon_M (q, w) = ----------------
                                        -1
                                \epsilon  (q, w)
                                        00
        
        while::

                    -1                  4pi
            \epsilon  (q, w)  =  1  + ------- chi (q, w)
                    00                 |q|**2    00
        """

        self.OpticalLimit = OpticalLimit

        chi0G0_w, chiG0RPA_w, chiG0LDA_w = self.calculate_chiGG(
                                         calc, q, wcut, wmin, wmax, dw, eta, sigma)

        # Transform q from reduced coordinate to cartesian coordinate
        
        qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)]) 
        
        assert qq.any() != 0
        tmp =  4. * pi / (qq[0]*qq[0]+qq[1]*qq[1]+qq[2]*qq[2]) 

        print 'Macroscopic dielectric function obtained! '

        epsilon0 = 1./(1. + tmp * chi0G0_w)
        epsilonRPA = 1./(1. + tmp * chiG0RPA_w)
        epsilonLDA = 1./(1. + tmp * chiG0LDA_w)

        return epsilon0, epsilonRPA, epsilonLDA



    def calculate_chiGG(self, calc, q, wcut, wmin, wmax, dw, eta, sigma):
        """Calculate chi_GG for a certain q and a series of omega at G=G'=0"""

        # Initialize, common stuff
        print 'Initializing:'
        e_kn, f_kn, C_knM, orb_MG, P_aMi, spos_ac, nt_G, tmp = (
           self.initialize(calc, q, wcut, wmin, wmax, dw, eta))

        if self.HilbertTrans:
            assert tmp.shape == (self.Nw, self.nS, self.nS) and tmp.dtype == complex
            chi0_wSS = tmp
        else:
            assert tmp.shape == (self.nkpt, 3)
            bzkpt_kG = tmp

        # Get pair-orbitals in Gspace
        print 'Calculating pair-orbital in G-space'
        Gvec = self.get_Gvectors()
        n_SG = self.pair_orbital_Gspace(orb_MG, calc.wfs.gd, calc.wfs.setups, 
                                        P_aMi, Gvec)

        # Get kernel
        print 'Calculating kernel'
        if isfile('kernel.npz'):
            foo = np.load('kernel.npz')
            KRPA_SS = foo['KRPA']
            KLDA_SS = foo['KLDA']
        else:
            KRPA_SS, KLDA_SS = self.kernel_extended_sys(n_SG, Gvec, nt_G,
                                orb_MG, calc.wfs.gd, calc.density.D_asp, 
                                P_aMi, calc.wfs.setups)
            np.savez('kernel.npz', KRPA=KRPA_SS, KLDA=KLDA_SS)

        # Solve Dyson's equation
        print 'Solving Dyson equation and transfrom chi_SS to G-space'
        chi0G0_w = np.zeros(self.Nw, dtype=complex)
        chiG0RPA_w = np.zeros_like(chi0G0_w)
        chiG0LDA_w = np.zeros_like(chi0G0_w)


        for iw in range(self.Nw):
            if not self.HilbertTrans:
                chi0_SS = self.calculate_chi0(bzkpt_kG, e_kn, f_kn, C_knM, q, iw*self.dw, eta=eta/Hartree)
            else:
                chi0_SS = chi0_wSS[iw]
            chi0_SS /= self.vol

            # Non-interacting
            chi0G0_w[iw] = self.chi_to_Gspace(chi0_SS, n_SG[:,0])
    
            # RPA
            chi_SS = self.solve_Dyson(chi0_SS, KRPA_SS)
            chiG0RPA_w[iw] = self.chi_to_Gspace(chi_SS, n_SG[:,0])
        
            # LDA
            chi_SS = self.solve_Dyson(chi0_SS, KLDA_SS)
            chiG0LDA_w[iw] = self.chi_to_Gspace(chi_SS, n_SG[:,0])

        return chi0G0_w, chiG0RPA_w, chiG0LDA_w


    def kernel_extended_sys(self, n_SG, Gvec, nt_G, orb_MG, gd, D_asp, P_aMi, setups):
        """Calculate the Kernel of a specific q for an extended system.

        The kernel is expressed as::

                          ----   *
            K      (q) =  \     n (G1) K  (q)  n (G2), 
             S1,S2        /___   S1     G1,G2   S2
                          G1,G2

        while the Coulomb part is::

             Coul        1     /  3  3  -i(q+G1).r   1    i(q+G2).r'
            K  (q)  =  -----  | dr dr' e          ------ e
             G1,G2      vol  /                    |r-r'|

                         4 pi
                    =  --------- delta(G1,G2), 
                       |q+G1|**2

        and the exchange-correlation part is::

             xc         1     /  3  3  -i(q+G1).r                   i(q+G2).r'
            K  (q)  = -----  | dr dr'  e         f (r) delta(r-r') e
             G1,G2     vol  /                     xc

                        1     /  3  -i(G1-G2).r
                    = -----  | dr  e            f (r)
                       vol  /                    xc

        The exchange-correlation kernel is more efficient if evaluating in real space::

             xc               /  3  *
            K      (q) = vol | dr  n (r) f (r) n (r)
             S1,S2          /       S1    xc    S2

        while:: 

                      1   ----         iG.r
            n (r) = ----- \     n (G) e     
             S       vol  /___   S
                            G
        """

        # Coulomb Kernel is diagonal 
        Kcoul_G = np.zeros(self.nG0)

        assert (self.q).any() != 0
        Kcoul_G[0] = 4. * pi / (self.q[0]**2 + self.q[1]**2 + self.q[2]**2)
        
        # Calculate G = 0 term separately
        for i in range(1,self.nG0):
            # get q+G vector 
            xx = np.array([np.inner(np.float64((Gvec[i]) + self.q), self.bcell[:,j]) for j in range(3)])
            Kcoul_G[i] = 1. / ( np.sqrt(np.inner(xx, xx)) )
        Kcoul_G *= 4. * pi 
        
        Kcoul_SS = gemmdot( (n_SG.conj() * Kcoul_G), (n_SG.T).copy(), beta = 0. )
        Kxc_SS = self.get_Kxc(nt_G, D_asp, orb_MG, P_aMi, gd, setups) * self.vol

        return Kcoul_SS, Kcoul_SS + Kxc_SS
 

    def chi_to_Gspace(self, chi_SS, nG0_S):
        """Transformation from chi_SS' to chi_GG'(G=G'=0) at a certain q and omega

        The transformation is defined as::

                            ----                        *
            chi    (q,w)  = \    n (G=0) * chi (q,w) * n (G=0)
               GG'=0        /___  S         SS'         S'
                             SS'
        """

        chiG0 = np.inner(np.inner(nG0_S, chi_SS), nG0_S.conj())

        return chiG0 


    def pair_orbital_Gspace(self, orb_MG, gd, setups, P_aMi, Gvec):
        """Calculate pair LCAO orbital in reciprocal space.

        The pair density is defined as::

                               -iG.r             ----           ~a     ~a
            n (G) = < phi   | e     | phi  >  +  \    < phi   | p  > < p     | phi  >
             S           mu              nu      /___      mu    ik     jk+q      nu
                                                 a,ij

                     iq.R_a (      a    -i(q+G).r     a        ~a    -i(q+G).r    ~a    )
                    e       | < phi  | e         | phi  > - < phi  | e         | phi  > |
                            (      i                  j          i                  j   )
            
        where the k-dependent projector is defined as::

              ~a      -ik.(r-R_a)   ~a
            | p  > = e            | p  >
               ik                    i
        """       
        
        n_SG = np.zeros((self.nS, self.nG0), dtype=complex)

        # The soft part
        for iS, (mu, nu) in enumerate(self.Sindex):
            n_SG[iS] = (np.fft.fftn(orb_MG[mu].conj() * orb_MG[nu])).ravel()
        # To check whether n_SG is correct, just look at the G=0 component
        # tmp = orb_MG[mu].conj() * orb_MG[nu]
        # calc.wfs.gd.integrate(tmp) should == n_SG[nLCAO*mu+nu, 0]

        n_SG = n_SG * self.vol / self.nG0
                               
        if self.OpticalLimit:
            print 'Optical limit calculation'
            qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
            for iS, (mu, nu) in enumerate(self.Sindex):
                    n_SG[iS, 0] = (-1j) * np.dot(qq, 
                         gd.calculate_dipole_moment( orb_MG[mu].conj() * orb_MG[nu]))

        # The augmentation part
        phi_aiiG = {}
        for iS, (mu, nu) in enumerate(self.Sindex):
            for a, id in enumerate(setups.id_a):
                Z, type, basis = id
                if not phi_aiiG.has_key(Z):
                    phi_aiiG[Z] = self.two_phi_planewave_integrals(Z, Gvec)
                assert phi_aiiG[Z] is not None
                tmp_ii = np.outer(P_aMi[a][mu].conj(), P_aMi[a][nu])
                for iG in range(self.nG0):
                    n_SG[iS, iG] += (tmp_ii * phi_aiiG[Z][:,:,iG]).sum()
                        
        return n_SG


    def get_Gvectors(self):
        """Calculate G-vectors.

        The G-vectors are defined as::

            G = m b  + m b  + m b  ,
                 1 1    2 2    3 3

        while b are lattice vectors, and m are integers

        By Fourier Tranform, the G-vectors are ordered as::
    
            0, 1, 2, ...., Gmax, Gmin, ... , -2, -1 

        The number of G-vectors == the number of grid points in the same direction (x,y,z)

        when the number of grid points is odd::
            
            Gmax = - Gmin = int (number of grid points / 2), Eg: 0,1,2,3,-3,-2,-1

        when the number of grid points is even::

            Gmax = - Gmin + 1 = number of grid points / 2, Eg: 0,1,2,3,-2,-1

        Note, only m vectors (the integer coefficients) are returned ! 
        """
        
        m = {}
        for dim in range(3):
            m[dim] = np.zeros(self.nG[dim],dtype=int)
            for i in range(self.nG[dim]):
                m[dim][i] = i
                if m[dim][i] > np.int(self.nG[dim]/2):
                    m[dim][i] = i- self.nG[dim]       

        G = np.zeros((self.nG0, 3), dtype=int)

        n = 0
        for i in range(self.nG[0]):
            for j in range(self.nG[1]):
                for k in range(self.nG[2]):
                    G[n, 0] = m[0][i]
                    G[n, 1] = m[1][j]
                    G[n, 2] = m[2][k]
                    n += 1

        return G


    def two_phi_planewave_integrals(self, Z, Gvec):
        """Calculate integral of two partial waves with a planewave for a certain specie.
    
        The expression is::
    
               a          a     -ik.(r-R_a)      a
            phi    = < phi  | e             | phi  > 
               ij         i                      j
        
                         ~ a     -ik.(r-R_a)    ~ a
                   -  < phi  | e             | phi  >
                           i                      j

        The planewave is expanded using real spherical harmonics by::
    
                          inf  l
             ik.r        ---- ----  l           ^     ^
            e     = 4 pi \    \    i  j (kr) Y (k) Y (r)
                         /___ /___     l      lm    lm
                          l=0 m=-l

        where \hat{v} is the polar angles of vector v and 
        j_l(kr) is the spherical bessel function. 

        The partial waves are also written as radial part times a spherical harmonics, 
        and the final expression for the integration is::
                            
                           ----     l    ^
            phi (G) = 4 pi \    (-i)  Y (k) 
               ij          /___        lm
                            lm

                       /  2                             /    ^            ^
                    *  | r  j (kr) phi (r) phi (r) dr * | Y (r) Y    Y   dr
                      /      l        j1      j2       /   lm    L1   L2

        where j is the combined index for (nl), L is the index for (l**2 + m).
        """
    
        # Create setup for a certain specie
        xcfunc = XCFunctional('LDA',nspins=1)
        symbol = chemical_symbols[Z]
        data = SetupData(symbol,'LDA')
        s = Setup(data,xcfunc,lmax=2)

        # radial grid stuff
        ng = s.ng
        g = np.arange(ng, dtype=float)
        r_g = s.beta * g / (ng - g) 
        dr_g = s.beta * ng / (ng - g)**2
        r2dr_g = r_g **2 * dr_g
        gcut2 = s.gcut2
            
        # Obtain the phi_j and phit_j
        phi_jg = []
        phit_jg = []
        
        for (phi_g, phit_g) in zip(s.data.phi_jg, s.data.phit_jg):
            phi_g = phi_g.copy()
            phit_g = phit_g.copy()
            phi_g[gcut2:] = phit_g[gcut2:] = 0.
            phi_jg.append(phi_g)
            phit_jg.append(phit_g)

        # Construct L (l**2 + m) and j (nl) index
        L_i = []
        j_i = []
        lmax = 0 
        for j, l in enumerate(s.l_j):
            for m in range(2 * l + 1):
                L_i.append(l**2 + m)
                j_i.append(j)
                if l > lmax:
                    lmax = l
        ni = len(L_i)
        lmax = 2 * lmax + 1

        # Initialize        
        R_jj = np.zeros((s.nj, s.nj))
        R_ii = np.zeros((ni, ni))
        phi_iiG = np.zeros((ni, ni, self.nG0), dtype=complex)
        j_lg = np.zeros((lmax, ng))
   
        # Store (phi_j1 * phi_j2 - phit_j1 * phit_j2 ) for further use
        tmp_jjg = np.zeros((s.nj, s.nj, ng))
        for j1 in range(s.nj):
            for j2 in range(s.nj): 
                tmp_jjg[j1, j2] = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]

        # Loop over G vectors 
        for iG in range(self.nG0):
            kk = np.array([np.inner(self.q + Gvec[iG], self.bcell[:,i]) for i in range(3)])
            k = np.sqrt(np.inner(kk, kk)) # calculate length of q+G
            
            # Calculating spherical bessel function
            for ri in range(ng):
                j_lg[:,ri] = self.sph_jn(lmax - 1,  k*r_g[ri])

            for li in range(lmax):
                # Radial part 
                for j1 in range(s.nj):
                    for j2 in range(s.nj): 
                        R_jj[j1, j2] = np.dot(r2dr_g, tmp_jjg[j1, j2] * j_lg[li])

                for mi in range(2 * li + 1):
                    # Angular part
                    for i1 in range(ni):
                        L1 = L_i[i1]
                        j1 = j_i[i1]
                        for i2 in range(ni):
                            L2 = L_i[i2]
                            j2 = j_i[i2]
                            R_ii[i1, i2] =  G_LLL[L1, L2, li**2+mi]  * R_jj[j1, j2]

                    phi_iiG[:, :, iG] += R_ii * Y(li**2 + mi, kk[0], kk[1], kk[2]) * (-1j)**li
            if iG % 10000 == 0:
                print '    Finished G vectors: ', iG, '(total: ', self.nG0, ')'
        phi_iiG *= 4 * pi

        if self.OpticalLimit:
            # Change the G = 0 component (not correct yet)
            # Can call Delta_pL
            qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
            Li = np.array([3, 1, 2])
            phi_p3 = np.array([s.Delta_pL[:,Li[ix]].copy() for ix in range(3)])
            phi_iiG[:, :, 0] = unpack(np.dot(qq, phi_p3)) * (-1j) * sqrt(4. * pi / 3.)
                          

#            li = 1
#            index = np.array([1,2,0]) # y, z, x
#            qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
#            R_ii3 = np.zeros((ni, ni, 3))
#            for j1 in range(s.nj):
#                for j2 in range(s.nj):
#                    R_jj[j1, j2] = np.dot(r2dr_g, tmp_jjg[j1, j2])
#            for mi in range(2 * li + 1):
#                for i1 in range(ni):
#                    for i2 in range(ni):
#                        R_ii3[i1, i2, index[mi-1]] = G_LLL[L_i[i1], L_i[i2], li**2+mi] * R_jj[j_i[i1],j_i[i2]]
#            phi_iiG[:, :, 0] = np.dot(R_ii3, qq) * (-1j) 
#
        
        return phi_iiG


    def sph_jn(self, n, z):
        """Calcuate spherical Bessel function.
   
        The spehrical Bessel function for the first three orders are::

                    sinz               3       sinz   3cosz
            j (z) = ---- ,   j (z) = (--- -1 ) ---- - -----
             0       z        2       z^2       z      z^2  
                            
                    sinz   cosz           15     6  sinz    15      cosz
            j (z) = ---- - ----, j (z) = (--- - ---)----  -(--- - 1)----
             1       z^2    z     3       z^3    z   z      z^2      z  
        """

        if n > 3:
            raise ValueError(
         'Spherical bessel function with n > 3 not implemented yet!')
        sph_n = np.zeros(4)
        if z == 0.:
            sph_n[0] = 1.
            sph_n[1:] = 0.
        else:
            tmp1 = np.sin(z) / z
            tmp2 = np.cos(z) / z
            sph_n[0] = tmp1
            sph_n[1] = tmp1 / z - tmp2
            sph_n[2] = (3./z**2 -1.) * tmp1 - 3./z * tmp2
            if n == 3:
                sph_n[3] = (15./z**3 - 6./z) * tmp1 - (15./z**2 -1.) * tmp2
                
        return sph_n[0:n+1]
