import numpy as np
from ase.units import Hartree
from gpaw.coulomb import CoulombNEW
from gpaw.utilities import pack
from gpaw.xc_functional import XCFunctional
from gpaw.lcao.pwf2 import LCAOwrap

class CHI:
    def __init__(self):
        self.xc = 'LDA'
        self.nspin = 1

    def get_dipole_strength(self, calc, q, wcut, wmin, wmax, dw, eta=0.2, sigma=2*1e-5):

        ''' Obtain the dipole strength spectra for a finite system

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

        '''
 
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
        Phi_S = self.pair_orbital_Rspace(orb_MG, calc.gd.h_c)

        # Get spectral function
        specfunc_SSw = self.calculate_spectral_function(bzkpt_kG, e_kn,
                              f_kn, C_knM, q, wcut, dw, sigma=2*1e-5)

        # Get chi0_SS' by hilbert transform
        chi0_SSw = self.hilbert_transform(specfunc_SSw, wmin, wmax, dw, eta=eta)

        # Get kernel
        kernelRPA_SS, kernelLDA_SS = self.kernel_finite_sys(nt_G, orb_MG, 
                        calc.wfs.kpt_u[0], calc.gd, calc.wfs.setups, spos_ac)

        # Solve Dyson's equation and Get dipole strength function
        SNonInter_w = np.zeros(self.Nw)
        SRPA_w = np.zeros(self.Nw)
        SLDA_w = np.zeros(self.Nw)
        for iw in range(self.Nw):
            SNonInter_w[iw] = self.calculate_dipole_strength(chi0_SSw[:,:,iw], Phi_S, iw*dw)
            chi_SS = self.solve_Dyson(chi0_SSw[:,:,iw], kernelRPA_SS)
            SRPA_w[iw] = self.calculate_dipole_strength(chi_SS, Phi_S, iw*dw)
            chi_SS = self.solve_Dyson(chi0_SSw[:,:,iw], kernelLDA_SS)
            SLDA_w[iw] = self.calculate_dipole_strength(chi_SS, Phi_S, iw*dw)

        return SNonInter_w, SRPA_w, SLDA_w

    def calculate_chi0(self, bzkpt_kG, e_kn, f_kn, C_knM, q, omega, eta=0.2):
        '''
          Calculate chi0_SS' for a certain q and omega

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
        '''

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
        '''
            Calculate spectral function A_SS' for a certain q and a series of omega
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
        '''

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
        '''
                        inf
           0            /   0                1             1
        chi (q, w) =   |  A (q, w') * (___________ - ___________)  dw'
           SS'         /   SS'          w-w'+ieta      w+w'+ieta
                       0

        # The dw' is deduced above in the delta function
        '''

        chi0_SSw = np.zeros((self.nS, self.nS, self.Nw), dtype=complex)

        for iw in range(self.Nw):
            w = wmin + iw * dw
            for jw in range(self.NwS):
                ww = jw * dw # = w' above 
                chi0_SSw[:,:,iw] += (1. / (w - ww + 1j*eta) 
                               - 1. / (w + ww + 1j*eta)) * specfunc_SSw[:,:,jw]
        return chi0_SSw

    def kernel_finite_sys(self, nt_G, orb_MG, kpt, gd, setups, spos_ac):

        '''        //                             /    1                   \
          K      = || dr  dr   phi (r ) phi (r ) (  -------- + f  (r  , r)  ) 
           S1,S2   //   1   2    mu  1    nu  1   \ |r - r |    xc  1    2 /
                                   1        1         1   2

                    *  phi (r )  phi (r )
                         mu  2     nu  2
                           2         2
          Especially note that , 
          coulomb.calculate returns the coulomb interaction in eV

        '''

        Kcoul_SS = np.zeros((self.nS, self.nS))
        Kxc_SS = np.zeros_like(Kcoul_SS)
        P1_ap = {}
        P2_ap = {}

        fxc_G = self.fxc(nt_G, self.xc, self.nspin)

        coulomb = CoulombNEW(gd, setups, spos_ac)
        for n in range(self.nLCAO):
            for m in range(self.nLCAO):
                nt1_G = orb_MG[n] * orb_MG[m] 
                for a, P_ni in kpt.P_ani.items():
                    D_ii = np.outer(P_ni[n].conj(), P_ni[m])
                    P1_ap[a] = pack(D_ii, tolerance=1e30)
                for p in range(self.nLCAO):
                    for q in range(self.nLCAO):
                        nt2_G = orb_MG[p] * orb_MG[q]
                        # Coulomb Kernel
                        for a, P_ni in kpt.P_ani.items():
                            D_ii = np.outer(P_ni[p].conj(), P_ni[q])
                            P2_ap[a] = pack(D_ii, tolerance=1e30)
                        # print n, m, p, q
                        Kcoul_SS[self.nLCAO*n+m, self.nLCAO*p+q] = coulomb.calculate(
                                    nt1_G, nt2_G, P1_ap, P2_ap)
                        # XC Kernel
                        Kxc_SS[self.nLCAO*n+m, self.nLCAO*p+q] = gd.integrate(nt1_G*fxc_G*nt2_G)
            print 'finished', n, ' (max: nLCAO)'
        tmp = Kcoul_SS / Hartree
        return tmp, tmp + Kxc_SS

    def solve_Dyson(self, chi0_SS, kernel_SS):

        ''' Solve Dyson's equation for a certain q and w
                        0         ----   0
        chi (q, w) = chi (q, w) + \    chi (q, w) K     chi (q, w)
          SS'          SS'        /___   SS        S S    S S'
                                   S S     1        1 2    2
                                    1 2
                  0
        Input: chi (q, w) at a given q and omega, 
                          for finite system, q = 0
        Output: chi(q, w)

        ---- (           ----   0             )                   0 
        \    |delta   - \    chi (q, w) K     |  chi (q, w)  = chi (q, w)
        /___ |    SS    /___   SS        S S  |    S S'          SS'
          S  (      2     S      1        1 2 )     2
           2               1 

        which is the form: Ax = B with known matrix A and B

        '''

        A_SS = np.eye(self.nS, self.nS, dtype=complex) - np.dot(chi0_SS, kernel_SS)
        chi_SS = np.dot(np.linalg.inv(A_SS), chi0_SS)
        # or equivalently, 
        # chi = np.linalg.solve(A_SS, chi0)

        return chi_SS


    def calculate_dipole_strength(self, chi_SS, Phi_S, omega):

        '''
         Calculate S(w) for a particular omega, atomic unit inside
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
        ''' 

        alpha = - np.sum( np.dot( np.dot(Phi_S, chi_SS), np.reshape(Phi_S,(Phi_S.shape[1],1)) ))

        S = 2. * omega / np.pi * np.imag(alpha)

        return S

    def chi_to_Gspace(self, chi_SS, pairphiG0_S):
        '''    
            Input either chi0 or chi at a certain  q and omega, and pair_phi       
                            ----                                    *
            chi    (q,w)  = \    pair_phi(G=0) * chi (q,w) *pair_phi (G=0)
               GG'=0        /___         S         SS'              S'
                             SS'

            Note,
            chi0(nLCAO**2,nLCAO**2) dtype=complex 
            pair_phi(1, nLCAO**2)   dtype=float64

        '''
        chiGspace = np.sum( np.dot( np.dot(pairphiG0_S, chi_SS),
                                     np.reshape(pairphiG0_S,(self.nS,1)) ))
        return chiGspace


    def find_kq(self, bzkpt_kG, q):
        ''' Find the index of k+q for all kpoints in BZ '''

        found = False
        kq = np.zeros(self.nkpt)

        for k in range(self.nkpt):
            # bzkpt(k,:) + q = bzkpt(kq,:)
            for kk in range(self.nkpt):
                tmp = sum(bzkpt_kG[k] - bzkpt_kG[kk] - q)
                if (np.abs(tmp) < 1e-8 or np.abs(tmp - 0.5) < 1e-8 or 
                    np.abs(tmp + 0.5) < 1e-8):
                   kq[k] = kk
                   found = True
                   break
            if not found: 
                raise ValueError('k+q not found')
        return kq


    def pair_C(self, C1, C2):     
        '''             *               
           Calculate   C     C       for a certain k, q, n, n'(m here)
                        nkM   n'k+qN 
           C1 = C_knM(k, n, :), C2 = C_knM(k+q, m, :)
           C1.shape = (nLCAO,)

        '''

        return np.ravel(np.outer(C1.conj(),C2)) # (nS,)


    def pair_orbital_G0(self, orb_MG):

        '''                          /        *                -iGr 
         Calculate pair_phi (G=0) = | dr ( phi (r)  phi (r) ) e
                           S       /         mu       nu
                 ----
         phi   = \    Phi   ( r - R ), with Phi_mu the LCAO orbital
            mu   /---    mu        I
                   I
        But note that, all the LCAO orbitals are real, 
        which means Phi_mu and phi_mu are all real 

        '''

        pairphiG0_MM = np.zeros((self.nLCAO, self.nLCAO))

        for mu in range(self.nLCAO):
            for nu in range(self.nLCAO):
                pairphiG0_MM[mu, nu] = np.sum(orb_MG[mu] * orb_MG[nu])
        pairphiG0_S = np.reshape(pairphiG0_MM, (1, self.nS)) 

        return pairphiG0_S

    def pair_orbital_Rspace(self, orb_MG, h_c):

        ''' h_c = calc.gd.h_c 
                      /
             Phi   =  | dr  r  phi (r) phi (r)
                S    /           mu      nu
           
             orb(nband, Nx, Ny, Nz)
        '''

        N_gd = orb_MG.shape[1:4] # number of grid points
        r = np.zeros((N_gd))        
        Phi_MM = np.zeros((self.nLCAO, self.nLCAO))

        for i in range(N_gd[0]):
            for j in range(N_gd[1]):
                for k in range(N_gd[2]):
                    r[i,j,k] = i*h_c[0] + j*h_c[1] + k*h_c[2]

        for mu in range(self.nLCAO):
            for nu in range(self.nLCAO):  
                Phi_MM[mu,nu] = np.sum(orb_MG[mu] * orb_MG[nu] * r)
        Phi_S = np.reshape(Phi_MM, (1,self.nS))
        return Phi_S


    def delta_function(self, x0, dx, Nx, sigma):
        
        '''                                 (x-x0)**2
                              1           - ___________
        delta(x-x0) =  ---------------   e    4*sigma
                      2 sqrt(pi*sigma)
        '''        

        deltax = np.zeros(Nx)
        for i in range(Nx):
            deltax[i] = np.exp(-(i*dx-x0)**2/(4.*sigma))
        return deltax / (2.*np.sqrt(np.pi*sigma))


    def fxc(self, n_G, name, nspins):

        libxc = XCFunctional(name, nspins)
       
        N = n_G.shape
        fxc = np.zeros((N[0]*N[1]*N[2]))

        libxc.calculate_fxc_spinpaired(np.reshape(n_G,N[0]*N[1]*N[2]), fxc)
        return np.reshape(fxc, N)
        
