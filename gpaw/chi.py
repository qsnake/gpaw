import numpy as np
from math import pi, sqrt
from ase.units import Hartree
from gpaw.coulomb import CoulombNEW
from gpaw.utilities import pack
from gpaw.xc_functional import XCFunctional
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.xc_correction import XCCorrection
from gpaw.utilities import unpack, unpack2 
from gpaw.utilities.lapack import diagonalize

class CHI:
    def __init__(self):
        self.xc = 'LDA'
        self.nspin = 1

    def get_dipole_strength(self, calc, q, wcut, wmin, wmax, dw, eta=0.2, sigma=2*1e-5):
        """ Obtain the dipole strength spectra for a finite system

        bzkpt_kG : the coordinate of kpoints in the whole BZ, nkpt*3D
        nband : number of bands, 1D
        nkpt  : number of kpoints, 1D
        e_kn : eigenvalues, (nkpt, nband)
        f_kn   : occupations, (nkpt,nband)
        C_knM  : LCAO coefficient (nkpt,nband,nLCAO)
        orb_MG : LCAO orbitals (nLCAO, ngrid, ngrid, ngrid)
          
        q     : chi(q, w), 1D
        wcut : cut-off energy for spectral function, 1D
        wmin, wmax : energy for the dipole strength spectra, 1D
        dw    : energy intervals for both the spectral function and the spectra, 1D
        eta   : the imaginary part in the non-interacting response function, 1D
        Nw    : number of frequency points on the spectra, 1D
        NwS   : number of frequency points for the spectral function, 1D

        nLCAO : nubmer of LCAO orbitals used, 1D
        nS    : S index for the matrix chi_SS', nS = nLCAO**2, 1D

        Phi_S : pair-orbitals in real space, Phi_S, (1, nS)
        specfunc_SSw : spectral function, (nS, nS, NwS, dtype = C_knM.type), 
                       can be complex128 or float64
        chi0_SSw  : chi0_SS', (nS, nS, Nw, dtype=complex)
        kernelRPA/LDA_SS: kernel_SS' in finite sys, (nS, nS), 
             so its float64, but can be complex for periodic sys.
        SNonInter/RPA/LDA_w : dipole strength function, (Nw)

        """
 
        bzkpt_kG = calc.get_bz_k_points()
        self.nband = calc.get_number_of_bands()
        self.nkpt = bzkpt_kG.shape[0]
       
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
        dw = dw / Hartree
        eta = eta / Hartree

        self.Nw = int((wmax - wmin) / dw) + 1 
        self.NwS = int(wcut/dw) + 1         

        self.nLCAO = C_knM.shape[2]
        self.nS = self.nLCAO **2

        # Get pair-orbitals in real space
        n_S = self.pair_orbital_Rspace(orb_MG, calc.gd.h_c, calc.wfs.setups, 
                                         calc.wfs.kpt_u[0])

        # Get spectral function
        specfunc_SSw = self.calculate_spectral_function(bzkpt_kG, e_kn,
                              f_kn, C_knM, q, wcut, dw, sigma=2*1e-5)

        # Get chi0_SS' by hilbert transform
        chi0_SSw = self.hilbert_transform(specfunc_SSw, wmin, wmax, dw, eta=eta)

        # Get kernel
        kernelRPA_SS, kernelLDA_SS = self.kernel_finite_sys(nt_G, calc.density.D_asp, orb_MG, 
                        calc.wfs.kpt_u[0], calc.gd, calc.wfs.setups, spos_ac)

        # Solve Dyson's equation and Get dipole strength function
        SNonInter_w = np.zeros((self.Nw,3))
        SRPA_w = np.zeros((self.Nw,3))
        SLDA_w = np.zeros((self.Nw,3))
        for iw in range(self.Nw):
            SNonInter_w[iw,:] = self.calculate_dipole_strength(chi0_SSw[:,:,iw], n_S, iw*dw)
            chi_SS = self.solve_Dyson(chi0_SSw[:,:,iw], kernelRPA_SS)
            SRPA_w[iw,:] = self.calculate_dipole_strength(chi_SS, n_S, iw*dw)
            chi_SS = self.solve_Dyson(chi0_SSw[:,:,iw], kernelLDA_SS)
            SLDA_w[iw,:] = self.calculate_dipole_strength(chi_SS, n_S, iw*dw)

        # Solve Casida's equation to get the excitation energies in eV
        eCasida_s = self.solve_casida(e_kn[0], f_kn[0], C_knM[0], kernelLDA_SS)

        return SNonInter_w, SRPA_w, SLDA_w, eCasida_s

    def calculate_chi0(self, bzkpt_kG, e_kn, f_kn, C_knM, q, omega, eta=0.2):
        """ Calculate chi0_SS' for a certain q and omega::

                                 ---- ----
                  0          2   \    \          f_nk - f_n'k+q
               chi (q, w) = ---   )    )    ------------------------
                  SS'       N_k  /    /     w + e_nk -e_n'k+q + i*eta
                                 ---- ----
                                  kbz  nn'
                               *                   *
                          *  C     C       C     C
                              nkM   n'k+qN  nkM'  n'k+qN'
        e_kn(nkpt,nband)
        f_kn(nkpt,nband)
        C_knM(nkpt,nband,nLCAO) nLCAO=nband
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
                        tmp = self.pair_C(C_knM[k,n,:], C_knM[kq[k],m,:])
                        # transpose and conjugate, C*C*C*C
                        tmp = np.outer(tmp, tmp.conj()) 
                        chi0_SS += tmp * focc / (omega + e_kn[k,n] - e_kn[kq[k],m] + 1j*eta)

        return chi0_SS / self.nkpt

    def calculate_spectral_function(self, bzkpt, e_kn, f_kn, C_knM, q, wcut, dw, sigma=1e-5):
        """ Calculate spectral function A_SS' for a certain q and a series of omega::

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
                        tmp = self.pair_C(C_knM[k,n,:], C_knM[kq[k],m,:]) # tmp[nS,1]
                        # C C C C
                        tmp = focc * np.outer(tmp, tmp.conj()) # tmp[nS,nS]
                        # calculate delta function
                        deltaw = self.delta_function(w0, dw,self.NwS, sigma)
                        for wi in range(self.NwS):
                            if deltaw[wi] > 1e-5:
                                specfunc_SSw[:,:,wi] += tmp * deltaw[wi]
        return specfunc_SSw *dw / self.nkpt

    def hilbert_transform(self, specfunc_SSw, wmin, wmax, dw, eta=0.01 ):
        """ Obtain chi0_SS' by hilbert transform with the spectral function A_SS'::

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
        """  Calculate the Kernel for a finite system
    
        Refer to report 4/11/2009, Eq. (18) - (22)::

                     //                   /    1                   \
            K      = || dr  dr   n (r )  (  -------- + f  (r  , r)  ) n (r )
             S1,S2   //   1   2   S  1    \ |r - r |    xc  1    2 /   S  2 
                                   1          1   2                     2 
      
            while 
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

        XC kernel :: 
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
                            P1_I = np.outer(P_Mi[n], P_Mi[m]).ravel()                            
                            P2_I = np.outer(P_Mi[p], P_Mi[q]).ravel() 
                            P_II = np.outer(P1_I, P2_I)
                            Kxc_SS[self.nLCAO*n+m, self.nLCAO*p+q] += (np.dot(P_II, J_II[a])).sum()
 
            print 'finished', n, 'cycle', ' (max: nLCAO = ', self.nLCAO, ')'
        tmp = Kcoul_SS / Hartree
        return tmp, tmp + Kxc_SS

    def solve_Dyson(self, chi0_SS, kernel_SS):
        """ Solve Dyson's equation for a certain q and w::

                            0         ----   0
            chi (q, w) = chi (q, w) + \    chi (q, w) K     chi (q, w)
              SS'          SS'        /___   SS        S S    S S'
                                       S S     1        1 2    2
                                        1 2
                      
        Input: chi_0 (q, w) at a given q and omega, 
                          for finite system, q = 0
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
        """ Calculate dipole strength for a particular omega
        Atomic unit ::

                     2w
             S(w) = ---- Im alpha(w) , alpha is the dynamical polarizability 
                     pi
                          //
             alpha(w) = - || dr dr' r chi(r,r',w) r'
                         //
                          //          ----               
                      = - || dr dr' r \    pair_phi(r) chi (w) pair_phi(r') r'
                         //           /___               SS'
                                      SS'
             where pair_phi(r) = phi * phi
                                   mu    nu
                          ----   /                         /
                      = - \     | dr r pair_phi(r) chi (w) | dr' r' pair_phi(r')
                          /___ /                     SS'  /
                           SS'
                          ----
                      = - \    Phi  chi (w) Phi  
                          /___    S    SS'     S'
                           SS'
             where Phi_S is defined in pair_orbital_Rspace
        """

        alpha = np.zeros(3, dtype=complex)
        for i in range(3):
            alpha[i] = - np.dot( np.dot(n_S[:,i], chi_SS), n_S[:,i]) 

        S = 2. * omega / pi * np.imag(alpha)

        return S

    def chi_to_Gspace(self, chi_SS, pairphiG0_S):
        """ Transform from chi_SS' to chi_GG'    

        Input either chi0 or chi at a certain  q and omega, and pair_phi::

                            ----                                    *
            chi    (q,w)  = \    pair_phi(G=0) * chi (q,w) *pair_phi (G=0)
               GG'=0        /___         S         SS'              S'
                             SS'

        Note,
        chi0(nLCAO**2,nLCAO**2) dtype=complex 
        pair_phi(1, nLCAO**2)   dtype=float64

        """

        chiGspace = np.sum( np.dot( np.dot(pairphiG0_S, chi_SS),
                                     np.reshape(pairphiG0_S,(self.nS,1)) ))
        return chiGspace


    def find_kq(self, bzkpt_kG, q):
        """ Find the index of k+q for all kpoints in BZ """

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


    def pair_C(self, C1, C2):     
        """ Calculate pair C_nkM C_n',k+q,N for a certain k, q, n, n'

        C1 = C_knM(k, n, :), C2 = C_knM(k+q, m, :)
        C1.shape = (nLCAO,)

        """

        return np.ravel(np.outer(C1.conj(),C2)) # (nS,)


    def pair_orbital_G0(self, orb_MG):
        """ Calculate pair atomic orbital at G=0 :: 

                              /        *                -iGr 
            pair_phi (G=0) = | dr ( phi (r)  phi (r) ) e
                    S       /         mu       nu
            where
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
        """ Calculate pair LCAO orbital in real space::
             
                   /
            n   =  | dr  r  phi (r) phi (r) 
             S    /           mu      nu
                      ----          ~ a    ~ a          (                    ~       ~      )
                    + \    < phi  | p  > < p  | phi  >  | phi (r) phi (r) - phi (r) phi (r) |
                      /___      mu   i      j      nu   (    i       j         i       j    )
                       ij       
        orb(nband, Nx, Ny, Nz)
        Delta_pL : L = 1, 2, 3 corresponds to y, z, x, refer to c/bmgs/sharmonic.c
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


    def delta_function(self, x0, dx, Nx, sigma):
        """ Approximate delta funcion using Gaussian wave::

                                                 (x-x0)**2
                                   1          - ___________
            delta(x-x0) =  ---------------   e    4*sigma
                          2 sqrt(pi*sigma)
        """        

        deltax = np.zeros(Nx)
        for i in range(Nx):
            deltax[i] = np.exp(-(i * dx - x0)**2/(4. * sigma))
        return deltax / (2. * sqrt(pi * sigma))


    def fxc(self, n):
        """ Return fxc[n(r)] for a given density array 

        """
        
        name = self.xc
        nspins = self.nspin

        libxc = XCFunctional(name, nspins)
       
        N = n.shape
        n = np.ravel(n)
        fxc = np.zeros_like(n)

        libxc.calculate_fxc_spinpaired(n, fxc)
        return np.reshape(fxc, N)

    def solve_casida(self, e_n, f_n, C_nM, kernel_SS):
        """ Solve Casida's equation with input from LCAO calculations (nspin = 1) ::

                            2 
            Omega F  = omega  F
                   I        I  I
                                      2          ---------------  
            Omega   = delta  delta   e    + 2   / f   e  f   e   K  
                ss'        ik     jq  s       \/   s   s  s'  s'  ss'

        Note, s is a combined index for ij, 
              s'                        kq

        The kernel is obtained from ::
                    ----
            K    =  \    C      C      C      C      K
             ss'    /___  i,mu   j,nu   k,mu   q,nu   S S
                     S S      1      1      2      2   1 2
                      1 2
        while capital S is a combined index for mu, nu
        and C is the LCAO coefficients

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

        kernel_ss += delta_ss

        # diagonalize the Omega matrix to get the square of exciation energies in Hartree
        eExcitation_s = np.zeros(npair)
        diagonalize(kernel_ss, eExcitation_s)
        eExcitation_s = np.array([sqrt(eExcitation_s[i]) for i in range(npair)])
        
        return eExcitation_s * Hartree


