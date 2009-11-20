from math import pi, sqrt

import numpy as np
from ase.units import Hartree, Bohr

from gpaw.coulomb import CoulombNEW
from gpaw.xc_functional import XCFunctional
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.xc_correction import XCCorrection
from gpaw.utilities import pack, unpack, unpack2 
from gpaw.utilities.lapack import diagonalize

class CHI:
    def __init__(self):
        self.xc = 'LDA'
        self.nspin = 1


    def initialize(self, calc, q, wcut, wmin, wmax, dw, eta):
        """Common stuff for all calculations (finite and extended systems) 

        Parameters: 

        bzkpt_kG: ndarray
            The coordinate of kpoints in the whole BZ, (nkpt,3)
        nband: integer
            The number of bands
        nkpt: integer
            The number of kpoints
        e_kn: ndarray
            Eigenvalues, (nkpt, nband)
        f_kn: ndarray
            Occupations, (nkpt,nband)
        C_knM: ndarray
            LCAO coefficient, (nkpt,nband,nLCAO)
        orb_MG: ndarray
            LCAO orbitals (nLCAO, ngrid, ngrid, ngrid)          
        q: float
            The wavevection in chi(q, w)
        wcut: float
            Cut-off energy for spectral function, 1D
        wmin (or wmax): float
            Energy cutoff for the dipole strength spectra
        dw: float
            Energy intervals for both the spectral function and the spectra
        eta: float
            The imaginary part in the non-interacting response function
        Nw: integer
            The number of frequency points on the spectra
        NwS: integer
            The number of frequency points for the spectral function
        nLCAO: integer
            The nubmer of LCAO orbitals used
        nS: integer
            The combined mu, nu index for the matrix chi_SS', nS = nLCAO**2

        """
        bzkpt_kG = calc.get_bz_k_points()
        self.nband = calc.get_number_of_bands()
        self.nkpt = bzkpt_kG.shape[0]
        self.acell = calc.atoms.cell / Bohr
        if calc.atoms.pbc.all() == True:
            self.get_primitive_cell()
        else:
            self.vol = 1.

        # obtain eigenvalues, occupations, LCAO coefficients and wavefunctions
        e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(self.nkpt)])
        f_kn = np.array([calc.get_occupation_numbers(kpt=k)
                          for k in range(self.nkpt)])
        C_knM = np.array([kpt.C_nM.copy() for kpt in calc.wfs.kpt_u])

        wrapper = LCAOwrap(calc)
        orb_MG = wrapper.get_orbitals()
        spos_ac = calc.atoms.get_scaled_positions()
        nt_G = calc.density.nt_sG[0]

        # Unit conversion
        e_kn = e_kn / Hartree
        wcut = wcut / Hartree
        wmin = wmin / Hartree
        wmax = wmax / Hartree
        self.dw = dw / Hartree
        eta = eta / Hartree

        self.Nw = int((wmax - wmin) / self.dw) + 1
        self.NwS = int(wcut/self.dw) + 1
        self.nLCAO = C_knM.shape[2]
        self.nS = self.nLCAO **2
        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]

        print 
        print 'Parameters used:'
        print
        print 'Number of bands:', self.nband
        print 'Number of kpints:', self.nkpt
        print 'Unit cell:'
        print self.acell
        if calc.atoms.pbc.all() == True: 
            print 'Primitive cell:'
            print self.bcell
        print 
        print 'Number of frequency points:', self.Nw
        print 'Number of frequency points for spectral function:', self.NwS
        print 'Number of LCAO orbitals:', self.nLCAO
        print 'Number of pair orbitals:', self.nS
        print 'Number of Grid points / G-vectors, and in total:', self.nG, self.nG0
        print 

        # Get spectral function
        print 'Calculating spectral function'
        specfunc_SSw = self.calculate_spectral_function(bzkpt_kG, e_kn,
                              f_kn, C_knM, q, wcut, self.dw, sigma=2*1e-5)

        # Get chi0_SS' by hilbert transform
        print 'Performing hilbert transform'
        chi0_SSw = self.hilbert_transform(specfunc_SSw, wmin, wmax, self.dw, eta)

        return e_kn, f_kn, C_knM, orb_MG, spos_ac, nt_G, chi0_SSw


    def get_dipole_strength(self, calc, q, wcut, wmin, wmax, dw, eta=0.2, sigma=2*1e-5):
        """Obtain the dipole strength spectra for a finite system.

        Parameters: 

        n_S: ndarray 
            Pair-orbitals in real space, (1, nS)
        specfunc_SSw: ndarray
            Spectral function, (nS, nS, NwS, dtype = C_knM.type), can be complex128 or float64
        chi0_SSw: ndarray
            The non-interacting density response function, (nS, nS, Nw, dtype=complex)
        kernelRPA_SS (or kernelLDA_SS): ndarray
            Kernel for the finite sys, (nS, nS), it is float64, but can be complex for periodic sys.
        SNonInter_w (or SRPA_w, SLDA_w): ndarray
            Dipole strength function, (Nw)
        """
 
        e_kn, f_kn, C_knM, orb_MG, spos_ac, nt_G, chi0_SSw = (
           self.initialize(calc, q, wcut, wmin, wmax, dw, eta))

        # Get pair-orbitals in real space
        n_S = self.pair_orbital_Rspace(orb_MG, calc.gd.h_c, calc.wfs.setups, 
                                         calc.wfs.kpt_u[0])

        # Get kernel
        kernelRPA_SS, kernelLDA_SS = self.kernel_finite_sys(nt_G, calc.density.D_asp, orb_MG, 
                        calc.wfs.kpt_u[0], calc.gd, calc.wfs.setups, spos_ac)

        # Solve Dyson's equation and Get dipole strength function
        SNonInter_w = np.zeros((self.Nw,3))
        SRPA_w = np.zeros((self.Nw,3))
        SLDA_w = np.zeros((self.Nw,3))
        for iw in range(self.Nw):
            SNonInter_w[iw,:] = self.calculate_dipole_strength(chi0_SSw[:,:,iw], n_S, iw*self.dw)
            chi_SS = self.solve_Dyson(chi0_SSw[:,:,iw], kernelRPA_SS)
            SRPA_w[iw,:] = self.calculate_dipole_strength(chi_SS, n_S, iw*self.dw)
            chi_SS = self.solve_Dyson(chi0_SSw[:,:,iw], kernelLDA_SS)
            SLDA_w[iw,:] = self.calculate_dipole_strength(chi_SS, n_S, iw*self.dw)

        # Solve Casida's equation to get the excitation energies in eV
        eCasidaRPA_s, sCasidaRPA_s = self.solve_casida(e_kn[0], f_kn[0], C_knM[0], kernelRPA_SS, n_S)
        eCasidaLDA_s, sCasidaLDA_s = self.solve_casida(e_kn[0], f_kn[0], C_knM[0], kernelLDA_SS, n_S)

        return SNonInter_w, SRPA_w, SLDA_w, eCasidaRPA_s, eCasidaLDA_s, sCasidaRPA_s, sCasidaLDA_s


    def get_EELS_spectrum(self, calc, q, wcut, wmin, wmax, dw, eta=0.2, sigma=2*1e-5):
        """Calculate Electron Energy Loss Spectrum of a periodic system for a particular q. 
            
        The Loss function is related to: 

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
        e_kn, f_kn, C_knM, orb_MG, spos_ac, nt_G, chi0_SSw = (
           self.initialize(calc, q, wcut, wmin, wmax, dw, eta))

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
            chi_SS = self.solve_Dyson(chi0_SSw[:,:,iw], KRPA_SS)
            
            chi0G0_w[iw] = self.chi_to_Gspace(chi0_SSw[:,:,iw], n_SG[:,0])
            chiG0_w[iw] = self.chi_to_Gspace(chi_SS, n_SG[:,0])

        return chi0G0_w, chiG0_w


    def calculate_chi0(self, bzkpt_kG, e_kn, f_kn, C_knM, q, omega, eta=0.2):
        """Calculate non-interacting response function (LCAO basis) for a certain q and omega. 

        The chi0_SS' matrix is calculated by::

                              ---- ----
               0          2   \    \          f_nk - f_n'k+q
            chi (q, w) = ---   )    )    ------------------------
               SS'       N_k  /    /     w + e_nk -e_n'k+q + i*eta
                              ---- ----
                               kbz  nn'
                            *                   *
                       *  C     C       C     C
                           nkM   n'k+qN  nkM'  n'k+qN'

        Parameters:

        e_kn: ndarray
            Eigen energies, (nkpt, nband)
        f_kn: ndarray
            Occupations, (nkpt, nband)
        C_knM: ndarray
            LCAO coefficients, (nkpt,nband,nLCAO)
        """

        chi0_SS = np.zeros((self.nS, self.nS), dtype=complex)

        if self.nkpt > 1:
            kq = self.find_kq(bzkpt_kG, q)
        else:
            kq = np.zeros((1))

        for k in range(self.nkpt):
            for n in range(self.nband):
                for m in range(self.nband):
                    focc = f_kn[k,n] - f_kn[kq[k],m]
                    if focc > 1e-5:
                        # pair C
                        tmp = (np.outer(C_knM[k,n,:].conj(), C_knM[kq[k],m,:])).ravel()
                       # tmp = self.pair_C(C_knM[k,n,:], C_knM[kq[k],m,:])
                        # transpose and conjugate, C*C*C*C
                        tmp = np.outer(tmp, tmp.conj()) 
                        chi0_SS += tmp * focc / (omega + e_kn[k,n] - e_kn[kq[k],m] + 1j*eta)

        return chi0_SS / self.vol


    def calculate_spectral_function(self, bzkpt_kG, e_kn, f_kn, C_knM, q, wcut, dw, sigma=1e-5):
        """Calculate spectral function for a certain q and a series of omega.

        The spectral function A_SS' is defined as::

                            ---- ----
             0          2   \    \         
            A (q, w) = ---   )    )   (f_nk - f_n'k+q) * delta(w + e_nk -e_n'k+q)
             SS'       N_k  /    /    
                            ---- ----
                             kbz  nn'
                          *                   *
                     *  C     C       C     C
                         nkM   n'k+qN  nkM'  n'k+qN'
             
        Use Gaussian wave for delta function
        """

        specfunc_SSw = np.zeros((self.nS, self.nS, self.NwS), dtype=C_knM.dtype)

        if self.nkpt > 1:
            kq = self.find_kq(bzkpt_kG, q)
        else:
            kq = np.zeros((1))

        for k in range(self.nkpt):
            for n in range(self.nband):
                for m in range(self.nband):
                    focc = f_kn[k,n] - f_kn[kq[k],m]
                    if focc > 1e-5:
                        w0 = e_kn[kq[k],m] - e_kn[k,n]
                        # pair C
                        tmp = (np.outer(C_knM[k,n,:].conj(), C_knM[kq[k],m,:])).ravel()
                        # tmp = self.pair_C(C_knM[k,n,:], C_knM[kq[k],m,:]) # tmp[nS,1]
                        # C C C C
                        tmp = focc * np.outer(tmp, tmp.conj()) # tmp[nS,nS]
                        # calculate delta function
                        deltaw = self.delta_function(w0, dw,self.NwS, sigma)
                        for wi in range(self.NwS):
                            if deltaw[wi] > 1e-5:
                                specfunc_SSw[:,:,wi] += tmp * deltaw[wi]
        return specfunc_SSw *dw / (self.vol) 


    def hilbert_transform(self, specfunc_SSw, wmin, wmax, dw, eta):
        """Obtain chi0_SS' by hilbert transform with the spectral function A_SS'.

        The hilbert tranform is performed as::

                            inf
               0            /   0                1             1
            chi (q, w) =   |  A (q, w') * (___________ - ___________)  dw'
               SS'         /   SS'          w-w'+ieta      w+w'+ieta
                           0

        Note, The dw' is reduced above in the delta function
        """

        chi0_SSw = np.zeros((self.nS, self.nS, self.Nw), dtype=complex)

        for iw in range(self.Nw):
            w = wmin + iw * dw
            for jw in range(self.NwS):
                ww = jw * dw # = w' above 
                chi0_SSw[:,:,iw] += (1. / (w - ww + 1j*eta) 
                               - 1. / (w + ww + 1j*eta)) * specfunc_SSw[:,:,jw]
        return chi0_SSw


    def kernel_finite_sys(self, nt_G, D_asp, orb_MG, kpt, gd, setups, spos_ac):
        """Calculate the Kernel for a finite system. 
    
        The kernel is expressed as, refer to report 4/11/2009, Eq. (18) - (22)::
                                                                            
                     //                 (      1                )
            K      = || dr1 dr2 n (r1 ) | --------  + f  (r1,r2)|  n (r2)
             S1,S2   //          S1     ( |r1 - r2|    xc       )   S2  
                                                                   
        while::

                     ~        ----  a       ~ a
            n (r)  = n (r)  + \    n (r)  - n (r)
             S        S       /___  S        S
                                a
            ~
            n (r)  = phi (r) phi (r)
             S          mu      nu
    
             a       ----          ~ a    ~ a
            n (r)  = \    < phi  | p  > < p  | phi  >  phi (r) phi (r)
             S       /___      mu   i      j      nu      i       j
                      ij
            ~a       ----          ~ a    ~ a           ~       ~  
            n (r)  = \    < phi  | p  > < p  | phi  >  phi (r) phi (r)
             S       /___      mu   i      j      nu      i       j
                      ij

        Note, phi_mu is LCAO orbital, while phi_i or phi_j are partial waves

        Coulomb Kernel: use coulomb.calculate (note it returns the E_coul in eV)

        XC kernel is obtained by::
 
             xc
            K       = < n   | f [n] | n   >  (note, n is the total density)
             S1,S2       S1    xc      S2
                        ~        ~    ~
                    = < n   | f [n] | n   > 
                         S1    xc      S2
                         ----     a        a     a         ~a       ~a    ~a
                      +  \     < n   | f [n ] | n   >  - < n   | f [n ] | n   >
                         /___     S1    xc       S2         S1    xc       S2
                           a

        The second term of the XC kernel can be further evaluated by::

            ---- ----           ~ a     ~ a                     ~ a     ~ a
            \    \     < phi  | p   > < p   | phi  >   < phi  | p   > < p   | phi  >
            /___ /___       mu   i1      i2      nu         mu   i3      i4      nu
              a  i1,i2        1                    1          2                    2
                 i3,i4

                    (  /      a       a         a     a       a 
                  * | | dr phi (r) phi (r)  f [n ] phi (r) phi (r) 
                    ( /       i1      i2     xc       i3      i4

                       /    ~ a     ~ a        ~a   ~ a     ~ a     )
                    - | dr phi (r) phi (r)  f [n ] phi (r) phi (r)  |
                      /       i1      i2     xc       i3      i4    )

        The method four_phi_integrals calculate the () term in the above equation
        """

        Kcoul_SS = np.zeros((self.nS, self.nS))
        Kxc_SS = np.zeros_like(Kcoul_SS)
        P1_ap = {}
        P2_ap = {}
        J_II = {}

        fxc_G = self.fxc(nt_G)  # nt_G contains core density
        for a, D_sp in D_asp.items():
            J_pp = setups[a].xc_correction.four_phi_integrals(D_sp, self.fxc)
            ni = setups[a].ni
            J_II[a] = np.zeros((ni*ni, ni*ni))   
            nii = J_pp.shape[0]
            J_pI = np.zeros((nii, ni*ni))
            for ip, J_p in enumerate(J_pp):
                J_pI[ip] = unpack2(J_p).ravel() # D_sp uses pack
            for ii in range(ni*ni):
                J_II[a][:, ii] = unpack2(J_pI[:, ii].copy()).ravel()

        coulomb = CoulombNEW(gd, setups, spos_ac)
        for n in range(self.nLCAO):
            for m in range(self.nLCAO):
                nt1_G = orb_MG[n] * orb_MG[m] 
                for a, P_Mi in kpt.P_aMi.items():
                    D_ii = np.outer(P_Mi[n].conj(), P_Mi[m])
                    P1_ap[a] = pack(D_ii, tolerance=1e30)
                for p in range(self.nLCAO):
                    for q in range(self.nLCAO):
                        nt2_G = orb_MG[p] * orb_MG[q]
                        # Coulomb Kernel
                        for a, P_Mi in kpt.P_aMi.items():
                            D_ii = np.outer(P_Mi[p].conj(), P_Mi[q])
                            P2_ap[a] = pack(D_ii, tolerance=1e30)
                        Kcoul_SS[self.nLCAO*n+m, self.nLCAO*p+q] = coulomb.calculate(
                                    nt1_G, nt2_G, P1_ap, P2_ap)
                        # XC Kernel
                        Kxc_SS[self.nLCAO*n+m, self.nLCAO*p+q] = gd.integrate(nt1_G*fxc_G*nt2_G)

                        for a, P_Mi in kpt.P_aMi.items():
                            P1_I = np.outer(P_Mi[n].conj(), P_Mi[m]).ravel()                            
                            P2_I = np.outer(P_Mi[p].conj(), P_Mi[q]).ravel() 
                            Kxc_SS[self.nLCAO*n+m, self.nLCAO*p+q] += (
                                    np.inner(np.inner(P1_I, J_II[a]), P2_I) )

            print 'finished', n, 'cycle', ' (max: nLCAO = ', self.nLCAO, ')'
        tmp = Kcoul_SS / Hartree

        return tmp, tmp + Kxc_SS


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

#        Kcoul_GG = np.eye(self.nG0)
        Kcoul_G = np.zeros(self.nG0)
        Kcoul_SS = np.zeros((self.nS, self.nS), dtype= complex)

        for i in range(self.nG0):
            # get q+G vector 
            xx = np.array([np.inner((Gvec[i] + q), self.bcell[:,j]) for j in range(3)])
            #Kcoul_GG[i, i] = 1. / ( xx[0]*xx[0] +xx[1]*xx[1] + xx[2]*xx[2] )
            Kcoul_G[i] = 1. / ( xx[0]*xx[0] +xx[1]*xx[1] + xx[2]*xx[2] )
        #Kcoul_GG *= 4. * pi 
        Kcoul_G *= 4. * pi 

        for i in range(self.nS):
            for j in range(self.nS):
                #Kcoul_SS[i, j] = np.inner(np.inner(n_SG[i].conj(), Kcoul_GG), n_SG[j])
                Kcoul_SS[i, j] = (n_SG[i].conj() * Kcoul_G * n_SG[j]).sum()
        
        return Kcoul_SS


    def solve_Dyson(self, chi0_SS, kernel_SS):
        """Solve Dyson's equation for a certain q and w. 

        The Dyson's equation is written as::
                                                                 
                            0         ----   0
            chi (q, w) = chi (q, w) + \    chi (q, w) K     chi (q, w)
              SS'          SS'        /___   SS        S S    S S'
                                       S S     1        1 2    2
                                        1 2
                      
        Input: chi_0 (q, w) at a given q and omega (for finite system, q = 0)
        Output: chi(q, w)

        It corresponds to solve::

            ---- (           ----   0             )                   0 
            \    |delta   - \    chi (q, w) K     |  chi (q, w)  = chi (q, w)
            /___ |    SS    /___   SS        S S  |    S S'          SS'
              S  (      2     S      1        1 2 )     2
               2               1 

        which is the form: Ax = B with known matrix A and B
        """

        A_SS = np.eye(self.nS, self.nS, dtype=complex) - np.dot(chi0_SS, kernel_SS)
        chi_SS = np.dot(np.linalg.inv(A_SS), chi0_SS)
        # or equivalently, 
        # chi = np.linalg.solve(A_SS, chi0)

        return chi_SS


    def calculate_dipole_strength(self, chi_SS, n_S, omega):
        """Calculate dipole strength for a particular omega.

        The dipole strength is obtained by (atomic unit)::

                    2w
            S(w) = ---- Im alpha(w) , 
                    pi

        while alpha is the dynamical polarizability defined as::

                         //
            alpha(w) = - || dr dr' r chi(r,r',w) r'
                        //
                         //          ----               
                     = - || dr dr' r \    n (r) chi (w) n (r') r'
                        //           /___  S       SS'   S'
                                      SS'

        The pair density is defined as::

            n (r) = phi (r) * phi (r) 
             S         mu        nu
     
                      ----          ~ a    ~ a          (                    ~       ~      )
                    + \    < phi  | p  > < p  | phi  >  | phi (r) phi (r) - phi (r) phi (r) |
                      /___      mu   i      j      nu   (    i       j         i       j    )
                       ij       

        As a result::

                         ----   /                    /
            alpha(w) = - \     | dr r n (r) chi (w) | dr' r' n (r')
                         /___ /        S       SS'  /         S'
                          SS'
                         ----
                     = - \    n  chi (w) n  
                         /___  S    SS'   S'
                          SS'
 
        where n_S is defined in pair_orbital_Rspace
        """

        alpha = np.zeros(3, dtype=complex)
        for i in range(3):
            alpha[i] = - np.dot( np.dot(n_S[:,i], chi_SS), n_S[:,i]) 

        S = 2. * omega / pi * np.imag(alpha)

        return S


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


    def find_kq(self, bzkpt_kG, q):
        """Find the index of k+q for all kpoints in BZ."""

        found = False
        kq = np.zeros(self.nkpt)
 
        for k in range(self.nkpt):
            # bzkpt(k,:) + q = bzkpt(kq,:)
            for kk in range(self.nkpt):
                tmp = sum(bzkpt_kG[k] - bzkpt_kG[kk] - q)
                if (np.abs(tmp) < 1e-8 or np.abs(tmp - 0.5) < 1e-8 or 
                    np.abs(tmp + 0.5) < 1e-8):
                    kq[k] = k
                    found = True
                    break
            if not found: 
                raise ValueError('k+q not found')
        return kq


    def pair_orbital_G0(self, orb_MG):
        """Calculate pair atomic orbital at G=0. 

        The pair orbital is defined as::

                       /        *                -iGr 
            n (G=0) = | dr ( phi (r)  phi (r) ) e
             S       /         mu       nu

        where::

                    ----
            phi   = \    Phi   ( r - R ), with Phi_mu the LCAO orbital
               mu   /---    mu        I
                      I

        But note that, all the LCAO orbitals are real, 
        which means Phi_mu and phi_mu are all real 
        """

        pairphiG0_MM = np.zeros((self.nLCAO, self.nLCAO))

        for mu in range(self.nLCAO):
            for nu in range(self.nLCAO):
                pairphiG0_MM[mu, nu] = np.sum(orb_MG[mu] * orb_MG[nu])
        pairphiG0_S = np.reshape(pairphiG0_MM, (1, self.nS)) 

        return pairphiG0_S


    def pair_orbital_Rspace(self, orb_MG, h_c, setups, kpt):
        """Calculate pair LCAO orbital in real space. 

        The pair density is defined as::
             
                   /
            n   =  | dr  r  phi (r) phi (r) 
             S    /           mu      nu
                      ----          ~ a    ~ a          /      (                    ~       ~      )
                    + \    < phi  | p  > < p  | phi  >  | dr r | phi (r) phi (r) - phi (r) phi (r) |
                      /___      mu   i      j      nu  /       (    i       j         i       j    )
                       ij       

        Parameters:

        orb: ndarray
            LCAO orbital on the grid, (nband, Nx, Ny, Nz)
        Delta_pL: ndarray
            L = 1, 2, 3 corresponds to y, z, x, refer to c/bmgs/sharmonic.c
        """

        N_gd = orb_MG.shape[1:4] # number of grid points
        r = np.zeros((N_gd))        
        n_MM = np.zeros((self.nLCAO, self.nLCAO))
        n_S = np.zeros((self.nS, 3))
        tmp =  sqrt(4. * pi / 3.)    
        Li = np.array([3, 1, 2])

        for ix in range(3): # loop over x, y, z axis
            phi_I={}
            for a in range(len(setups)):
                phi_p = setups[a].Delta_pL[:,Li[ix]].copy()
                phi_I[a] = unpack(phi_p).ravel()
       
            for i in range(N_gd[0]):
                for j in range(N_gd[1]):
                    for k in range(N_gd[2]):
                        if ix == 0:
                            r[i,j,k] = i*h_c[0]
                        elif ix == 1:
                            r[i,j,k] = j*h_c[1] 
                        else:
                            r[i,j,k] = k*h_c[2]
    
            for mu in range(self.nLCAO):
                for nu in range(self.nLCAO):  
                    n_MM[mu,nu] = np.sum(orb_MG[mu] * orb_MG[nu] * r)
                    for a, P_Mi in kpt.P_aMi.items():
                        P_I = np.outer(P_Mi[mu], P_Mi[nu]).ravel()
                        n_MM[mu,nu] += np.sum(P_I * phi_I[a]) * tmp
            n_S[:,ix] = np.reshape(n_MM, self.nS)

        return n_S


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
                n_SG[self.nLCAO*mu + nu] = np.fft.fftn((orb_MG[mu].conj() * orb_MG[nu]).ravel())

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


    def delta_function(self, x0, dx, Nx, sigma):
        """Approximate delta funcion using Gaussian wave.

        The Gaussian wave expression for delta function is written as::

                                                 (x-x0)**2
                                   1          - ___________
            delta(x-x0) =  ---------------   e    4*sigma
                          2 sqrt(pi*sigma)

        Parameters:

        sigma: float
            Broadening factor
        x0: float
            The center of the delta function
        """        

        deltax = np.zeros(Nx)
        for i in range(Nx):
            deltax[i] = np.exp(-(i * dx - x0)**2/(4. * sigma))
        return deltax / (2. * sqrt(pi * sigma))


    def fxc(self, n):
        """Return fxc[n(r)] for a given density array."""
        
        name = self.xc
        nspins = self.nspin

        libxc = XCFunctional(name, nspins)
       
        N = n.shape
        n = np.ravel(n)
        fxc = np.zeros_like(n)

        libxc.calculate_fxc_spinpaired(n, fxc)
        return np.reshape(fxc, N)


    def solve_casida(self, e_n, f_n, C_nM, kernel_SS, n_S):
        """Solve Casida's equation with input from LCAO calculations (nspin = 1).

        The Casida matrix equation is written as::

                            2 
            Omega F  = omega  F
                   I        I  I

        while the Omega matrix is defined as::

                                      2          ---------------  
            Omega   = delta  delta   e    + 2   / f   e  f   e   K  
                ss'        ik     jq  s       \/   s   s  s'  s'  ss'

        Note, s(s') is a combined index for ij (kq)

        The kernel is obtained from ::

                    ----
            K    =  \    C      C      C      C      K
             ss'    /___  i,mu   j,nu   k,mu   q,nu   S S
                     S S      1      1      2      2   1 2
                      1 2

        Parameters:

        i (or k): integer
            Index for occupied states
        j (or q): integer
            Index for unoccupied states
        s (or s'): integer
            Combined index for ij (or kq)
        S (or S'): integer
            Combined index for mu,nu (or mu',nu')
        C_nM: ndarray
            The LCAO coefficients at kpt=0, (nband, nLCAO)
        omega_I: ndarray
            Excitation energies
        """

        # Count number of occupied and unoccupied states pairs
        Nocc = 0
        Nunocc = 0
        for n in range(self.nband):
            if f_n[n] > 0:
                Nocc += 1
            else: 
                Nunocc +=1
        if Nocc + Nunocc != self.nband:
            raise ValueError('Nocc + Nunocc != nband')
        npair = Nocc * Nunocc

        # calculate the factor before the K_ij,kq matrix
        e_s = np.zeros(npair)
        f_s = np.zeros_like(e_s)

        ipair = 0
        for i in range(Nocc):
            for j in range(Nocc, self.nband):
                e_s[ipair] = e_n[j] - e_n[i] # s: ij pair
                f_s[ipair] = f_n[i] - f_n[j]
                ipair += 1

        fe_ss = np.outer(e_s * f_s, e_s * f_s).ravel()
        fe_ss = (np.array([2. * sqrt(fe_ss[i]) for i in range(npair**2)])).reshape(npair, npair)
 
        # calculate kernel K_ij,kq
        npair1 = 0
        npair2 = 0
        kernel_ss = np.zeros((npair, npair))

        for i in range(Nocc):
            for j in range(Nocc, self.nband):
                C1_S = np.outer(C_nM[i], C_nM[j]).ravel() # S: mu nu pair
                for k in range(Nocc):
                    for q in range(Nocc, self.nband):
                        C2_S = np.outer(C_nM[k], C_nM[q]).ravel() # S: mu nu pair
                        kernel_ss[npair1, npair2] = np.inner(np.inner(C1_S, kernel_SS), C2_S)
                        npair2 += 1
                npair1 += 1
                npair2 = 0

        kernel_ss *= fe_ss

        # add the delta matrix to obtain the Omega matrix
        delta_ss = np.eye(npair,npair)
        for i in range(npair):
            delta_ss[i,i] *= e_s[i]**2

        Omega_ss = kernel_ss + delta_ss

        # diagonalize the Omega matrix
        eExcitation_s = np.zeros(npair)
        diagonalize(Omega_ss, eExcitation_s)

        # get the excitation energies in Hartree
        eExcitation_s = np.array([sqrt(eExcitation_s[i]) for i in range(npair)])
        
        # get the dipole strength 
        ipair = 0
        mu_s = np.zeros((npair, 3))
        for i in range(Nocc):
            for j in range(Nocc, self.nband):
                for ix in range(3): # x,y,z three directions
                    mu_s[ipair, ix] = np.inner(np.outer(C_nM[i], C_nM[j]).ravel(), n_S[:, ix])
                ipair += 1

        fe_s = np.array([ sqrt(f_s[i] * e_s[i]) for i in range(npair)])

        DipoleStrength = np.zeros((npair, 3))
        for s1 in range(npair):
            FI_s = Omega_ss[s1]            
            DipoleStrength[s1] = np.array([ 2. * ((mu_s[:, ix] * fe_s * FI_s).sum())**2 for ix in range(3) ])

        return eExcitation_s * Hartree, DipoleStrength


