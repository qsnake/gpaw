from math import pi
from os.path import isfile

import numpy as np
from ase.units import Hartree

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
        e_kn, f_kn, C_knM, orb_MG, spos_ac, nt_G, tmp = (
           self.initialize(calc, q, wcut, wmin, wmax, dw, eta))

        if self.HilbertTrans:
            assert tmp.shape == (self.Nw, self.nS, self.nS) and tmp.dtype == complex
            chi0_wSS = tmp
        else:
            assert tmp.shape == (self.nkpt, 3)
            bzkpt_kG = tmp

        assert calc.atoms.pbc.all()
        self.get_primitive_cell()

        print 'Periodic system calculations.'
        print 'Reciprocal primitive cell (1 / a.u.)'
        print self.bcell
        print 'Cell volume (a.u.**3):', self.vol
        print 'BZ volume (1/a.u.**3):', self.BZvol

        # Get pair-orbitals in Gspace
        print 'Calculating pair-orbital in G-space'
        n_SG = self.pair_orbital_Gspace(orb_MG, calc.wfs.gd)

        # Get kernel
        print 'Calculating kernel'
        if isfile('kernel.npz'):
            foo = np.load('kernel.npz')
            KRPA_SS = foo['KRPA']
            KLDA_SS = foo['KLDA']

        else:
            Gvec = self.get_Gvectors()
            KRPA_SS, KLDA_SS = self.kernel_extended_sys(n_SG, Gvec, nt_G,
                                orb_MG, calc.wfs.gd, calc.density.D_asp, 
                                calc.wfs.kpt_u[0], calc.wfs.setups)
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


    def kernel_extended_sys(self, n_SG, Gvec, nt_G, orb_MG, gd, D_asp, kpt, setups):
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
            Kcoul_G[i] = 1. / ( xx[0]*xx[0] + xx[1]*xx[1] + xx[2]*xx[2] )
        Kcoul_G *= 4. * pi 
        

        Kcoul_SS = gemmdot( (n_SG.conj() * Kcoul_G), (n_SG.T).copy(), beta = 0. )

        Kxc_SS = self.get_Kxc(nt_G, D_asp, orb_MG, kpt, gd, setups) * self.vol

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


    def pair_orbital_Gspace(self, orb_MG, gd):
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

        for mu in range(self.nLCAO):
            for nu in range(self.nLCAO):
                # The last dimension runs fastest when using ravel()
                # soft part
                n_SG[self.nLCAO*mu + nu] = (np.fft.fftn(orb_MG[mu].conj() * orb_MG[nu])).ravel()

        # To check whether n_SG is correct, just look at the G=0 component
        # tmp = orb_MG[mu].conj() * orb_MG[nu]
        # calc.wfs.gd.integrate(tmp) should == n_SG[nLCAO*mu+nu, 0]

        n_SG = n_SG * self.vol / self.nG0

        if self.OpticalLimit:
            print 'Optical limit calculation'
            N_gd = orb_MG.shape[1:4]
            r = np.zeros(N_gd, dtype=complex)
    
            qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
    
            for i in range(N_gd[0]):
                for j in range(N_gd[1]):
                    for k in range(N_gd[2]):
                        tmp = np.array([i*self.h_c[0], j*self.h_c[1], k*self.h_c[2]])
                        r[i,j,k] = np.dot(qq, tmp)
      
            for mu in range(self.nLCAO):
                for nu in range(self.nLCAO):
                    n_SG[self.nLCAO*mu + nu, 0] = np.complex128(gd.integrate(orb_MG[mu] * orb_MG[nu] * r))

        return n_SG


    def get_primitive_cell(self):
        """Calculate the reciprocal lattice vectors and the volume of primitive and BZ cell.

        The volume of the primitive cell is calculated by::

            vol = | a1 . (a2 x a3) |

        The primitive lattice vectors are calculated by::

                       a2 x a3
            b1 = 2 pi ---------, and so on
                         vol

        Parameters:

        a: ndarray
            Primitive cell lattice vectors, calc.get_atoms().cell(), (3, 3)
        b: ndarray
            Reciprocal lattice vectors, (3, 3)
        vol: float
            Volume of the primitive cell
        BZvol: float
            Volume of the BZ, BZvol = (2pi)**3/vol for 3-dimensional crystal
        """

        a = self.acell

        self.vol = np.dot(a[0],np.cross(a[1],a[2]))
        self.BZvol = (2. * pi)**3 / self.vol

        b = np.zeros_like(a)        
        b[0] = np.cross(a[1], a[2])
        b[1] = np.cross(a[2], a[0])
        b[2] = np.cross(a[0], a[1])
        self.bcell = 2. * pi * b / self.vol

        return


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


