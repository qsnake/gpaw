from math import pi

import numpy as np
from ase.units import Hartree

from gpaw.response import CHI

class PeriodicSys(CHI):
    def __init__(self):
        pass


    def get_EELS_spectrum(self, calc, q, wcut, wmin, wmax, dw, eta=0.2, sigma=2*1e-5):
        """Calculate Electron Energy Loss Spectrum.

        Calculate EELS of a periodic system for a particular q. The 
        Loss function is related to::

                         -1            4 pi
            - Im \epsilon (q, w) = - -------  Im  chi (q, w)
                        G=0,G'=0      |q|**2        G=0,G'=0
        """

        # Calculate chi_G=0,G'=0 (q, w)
        chi0G0_w, chiG0_w = self.calculate_chiGG(calc, q, wcut, wmin, wmax, dw, eta, sigma)

        # Transform q from reduced coordinate to cartesian coordinate
        qq = np.array([np.inner(q, self.bcell[:,i]) for i in range(3)]) 
        
        LossFunc0_w = - 4. * pi / (qq[0]*qq[0]+qq[1]*qq[1]+qq[2]*qq[2]) * np.imag(chi0G0_w)
        LossFunc_w = - 4. * pi / (qq[0]*qq[0]+qq[1]*qq[1]+qq[2]*qq[2]) * np.imag(chiG0_w)
        print 'EELS spectrum obtained! '

        return LossFunc0_w, LossFunc_w


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
        print 'Primitive cell (1 / a.u.)'
        print self.bcell

        # Get pair-orbitals in Gspace
        print 'Calculating pair-orbital in G-space'
        n_SG = self.pair_orbital_Gspace(orb_MG, calc.gd)

        # Get kernel
        print 'Calculating kernel'
        Gvec = self.get_Gvectors()
        # q are expressed in terms of the primitive lattice vectors
        KRPA_SS = self.kernel_extended_sys(n_SG, q, Gvec)

        # Solve Dyson's equation
        print 'Solving Dyson equation and transfrom chi_SS to G-space'
        chi0G0_w = np.zeros(self.Nw, dtype=complex)
        chiG0_w = np.zeros(self.Nw, dtype=complex)

        for iw in range(self.Nw):
            if not self.HilbertTrans:
                chi0_SS = self.calculate_chi0(bzkpt_kG, e_kn, f_kn, C_knM, q, iw*self.dw, eta=eta/Hartree)
            else:
                chi0_SS = chi0_wSS[iw]
            chi0_SS /= self.vol

            chi_SS = self.solve_Dyson(chi0_SS, KRPA_SS)
            
            chi0G0_w[iw] = self.chi_to_Gspace(chi0_SS, n_SG[:,0])
            chiG0_w[iw] = self.chi_to_Gspace(chi_SS, n_SG[:,0])

        return chi0G0_w, chiG0_w


    def kernel_extended_sys(self, n_SG, q, Gvec):
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
        """

        # Coulomb Kernel is diagonal 
        Kcoul_G = np.zeros(self.nG0)

        for i in range(self.nG0):
            # get q+G vector 
            xx = np.array([np.inner((Gvec[i] + q), self.bcell[:,j]) for j in range(3)])
            Kcoul_G[i] = 1. / ( xx[0]*xx[0] +xx[1]*xx[1] + xx[2]*xx[2] )
        Kcoul_G *= 4. * pi 

        Kcoul_SS = np.dot((n_SG.conj() * Kcoul_G), n_SG.T)
        
        return Kcoul_SS


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
        # calc.gd.integrate(tmp) should == n_SG[nLCAO*mu+nu, 0]
        return n_SG * self.vol / self.nG0 


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
            m[dim] = np.zeros(self.nG[dim])
            for i in range(self.nG[dim]):
                m[dim][i] = i
                if m[dim][i] > np.int(self.nG[dim]/2):
                    m[dim][i] = i- self.nG[dim]       

        G = np.zeros((self.nG0, 3))

        n = 0
        for i in range(self.nG[0]):
            for j in range(self.nG[1]):
                for k in range(self.nG[2]):
                    G[n, 0] = m[0][i]
                    G[n, 1] = m[1][j]
                    G[n, 2] = m[2][k]
                    n += 1

        return G


