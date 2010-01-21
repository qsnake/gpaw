from math import pi, sqrt
from os.path import isfile

import numpy as np
from ase.units import Hartree, Bohr

from gpaw.xc_functional import XCFunctional
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.utilities.blas import gemmdot
from gpaw.utilities import unpack2
from gpaw.lfc import BasisFunctions
from gpaw import GPAW


class CHI:
    def __init__(self):
        self.xc = 'LDA'
        self.nspin = 1


    def initialize(self, calc, q, wcut, wmin, wmax, dw, eta):
        """Common stuff for all calculations (finite and extended systems).

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

        print
        print 'Start response function calculation! '
        print

        bzkpt_kG = calc.get_bz_k_points()
        self.nband = calc.get_number_of_bands()
        self.nkpt = bzkpt_kG.shape[0]
        self.acell = calc.atoms.cell / Bohr
        self.nvalence = calc.wfs.setups.nvalence
        print self.nvalence

        # obtain eigenvalues, occupations, LCAO coefficients and wavefunctions
        e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(self.nkpt)])
        f_kn = np.array([calc.get_occupation_numbers(kpt=k)
                          for k in range(self.nkpt)])
        
        if calc.wfs.kpt_u[0].C_nM is not None:
            C_knM = np.array([kpt.C_nM.copy() for kpt in calc.wfs.kpt_u])
            if self.nkpt == 1: # only store for the finite sys # for periodic sys stored elsewhere
                np.savez('C_knM.npz',C=C_knM)
        elif isfile('C_knM.npz'): 
            foo = np.load('C_knM.npz')
            C_knM = foo['C']
            assert C_knM.shape[0] == self.nkpt and (
                   C_knM.shape[1] == self.nband)
        else:
            raise ValueError('C_knM not exist!')


        spos_ac = calc.atoms.get_scaled_positions()
        nt_G = calc.density.nt_sG[0]
        print 'Memory usage of nt_G:', nt_G.nbytes / 1024**2, ' Mb'
        # Unit conversion
        e_kn = e_kn / Hartree
        wcut = wcut / Hartree
        wmin = wmin / Hartree
        wmax = wmax / Hartree
        self.dw = dw / Hartree
        eta = eta / Hartree
        if calc.atoms.pbc.any():
            if self.OpticalLimit:
                self.q = np.array([0.001, 0, 0])
            else:
                self.q = q

        self.Nw = int((wmax - wmin) / self.dw) + 1
        self.NwS = int(wcut/self.dw) + 1
        self.nLCAO = C_knM.shape[2]
        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        self.h_c = calc.wfs.gd.h_c

        # get LCAO orbitals 
        # sum_I Phi(r-R_I) 
        print 'Get LCAO orbitals'
        orb_MG = self.get_orbitals(calc, spos_ac)
        print 'Memory usage of orb_MG:', orb_MG.nbytes / (1024.**2), ' Mb'
        P_aMi  = self.get_P_aMi(calc)
        self.Sindex = self.get_reduced_pair_orbital_index(orb_MG, calc.wfs.gd)
        self.nS = len(self.Sindex)

        print 
        print 'Parameters used:'
        print
        print 'Number of bands:', self.nband
        print 'Number of kpoints:', self.nkpt
        print 'Unit cell (a.u.):'
        print self.acell
        print
        print 'Number of frequency points:', self.Nw
        print 'Number of frequency points for spectral function:', self.NwS
        print 'Number of LCAO orbitals:', self.nLCAO
        print 'Number of pair orbitals:', self.nLCAO**2
        print 'Number of reduced pair orbitals:', self.nS
        print 'Number of Grid points / G-vectors, and in total:', self.nG, self.nG0
        print 'Grid spacing (a.u.):', self.h_c
        print 

        print 'Eigenvalues (eV):'
        for k in range(self.nkpt):
            print e_kn[k] * Hartree
        print

        print 'Occupations:'
        for k in range(self.nkpt):
            print f_kn[k]
        print

        
        if calc.atoms.pbc.any():
            self.get_primitive_cell()

            print 'Periodic system calculations.'
            print 'Reciprocal primitive cell (1 / a.u.)'
            print self.bcell
            print 'Cell volume (a.u.**3):', self.vol
            print 'BZ volume (1/a.u.**3):', self.BZvol

            # if C_knM is not read from file, then we should chnage C_knM
            # C_knM *= e{i k. R_a}
#            if not isfile('C_knM.npz'):
                #print 'calculating renormalized C_knM'
                #bzkpt_kG = calc.get_bz_k_points()
                #pos_a = calc.get_atoms().positions / Bohr
                #m_a = calc.wfs.basis_functions.M_a
                #for a in calc.wfs.basis_functions.my_atom_indices:
                #    m1 = m_a[a]
                #    m2 = m1+ calc.wfs.setups[a].niAO
                #    for ik in range(self.nkpt):
                #        kk =  np.array([np.inner(bzkpt_kG[ik], self.bcell[:,i]) for i in range(3)])
                #        C_knM[ik,:,m1:m2] *= np.exp(-1j * np.dot(kk, pos_a[a]))
#                np.savez('C_knM.npz',C=C_knM)


        # whether to use hilbert tranform
        self.HilbertTrans = True
        try:
            chi0_wSS = np.zeros((self.Nw, self.nS, self.nS), dtype=complex) 
        except:
            self.HilbertTrans = False
            print 'Not using spectral function and hilbert transform'

        if self.HilbertTrans:
            # Get spectral function
            print 'Calculating spectral function'
            specfunc_wSS = self.calculate_spectral_function(bzkpt_kG, e_kn,
                              f_kn, C_knM, q, wcut, self.dw, sigma=2*1e-5)

            # Get chi0_SS' by hilbert transform
            print 'Performing hilbert transform'
            chi0_wSS = self.hilbert_transform(specfunc_wSS, wmin, wmax, self.dw, eta)
            print 'Memory usage of chi0_wSS:', chi0_wSS.nbytes / 1024.**2, ' Mb'
            return e_kn, f_kn, C_knM, orb_MG, P_aMi, spos_ac, nt_G, chi0_wSS 
        else:
            return e_kn, f_kn, C_knM, orb_MG, P_aMi, spos_ac, nt_G, bzkpt_kG


    def calculate_chi0(self, bzkpt_kG, e_kn, f_kn, C_knM, q, omega, eta):
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

        kq = np.zeros(self.nkpt)
        if self.nkpt > 1:  # periodic system
            assert self.OpticalLimit is not None
            if not self.OpticalLimit :
                kq = self.find_kq(bzkpt_kG, q)
            else:
                for k in range(self.nkpt):
                    kq[k] = k
        else: # finite system or Gamma-point calculation
            kq[0] = np.zeros(1)

        tmp_S = np.zeros(self.nS, dtype=C_knM.dtype)
        for k in range(self.nkpt):
            for n in range(self.nband):
                for m in range(self.nband):
                    focc = f_kn[k,n] - f_kn[kq[k],m]
                    if focc > 1e-5:
                        # pair C
                        for iS, (mu, nu) in enumerate(self.Sindex):
                            tmp_S[iS] = C_knM[k,n,mu].conj() * C_knM[kq[k],m,nu]
                        # transpose and conjugate, C*C*C*C
                        tmp_SS = np.outer(tmp_S, tmp_S.conj()) 
                        chi0_SS += tmp_SS * focc / (omega + e_kn[k,n] - e_kn[kq[k],m] + 1j*eta)

        return chi0_SS 


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

        specfunc_wSS = np.zeros((self.NwS, self.nS, self.nS), dtype=C_knM.dtype)
        kq = np.zeros(self.nkpt)

        if self.nkpt > 1:  # periodic system
            assert self.OpticalLimit is not None
            if not self.OpticalLimit : 
                kq = self.find_kq(bzkpt_kG, q)
            else:
                for k in range(self.nkpt):
                    kq[k] = k
        else: # finite system or Gamma-point calculation
            kq[0] = np.zeros(1)

        tmp_S = np.zeros(self.nS, dtype=C_knM.dtype)
        for k in range(self.nkpt):
            for n in range(self.nband):
                for m in range(self.nband):
                    focc = f_kn[k,n] - f_kn[kq[k],m]
                    if focc > 1e-5:
                        w0 = e_kn[kq[k],m] - e_kn[k,n]
                        # pair C
                        for iS, (mu, nu) in enumerate(self.Sindex):
                            tmp_S[iS] = C_knM[k,n,mu].conj() * C_knM[kq[k],m,nu]
                        # C C C C
                        tmp_SS = focc * np.outer(tmp_S, tmp_S.conj()) # tmp[nS,nS]
                        # calculate delta function
                        deltaw = self.delta_function(w0, dw,self.NwS, sigma)
                        for wi in range(self.NwS):
                            if deltaw[wi] > 1e-5:
                                specfunc_wSS[wi] += tmp_SS * deltaw[wi]
        return specfunc_wSS *dw 


    def hilbert_transform(self, specfunc_wSS, wmin, wmax, dw, eta):
        """Obtain chi0_SS' by hilbert transform with the spectral function A_SS'.

        The hilbert tranform is performed as::

                            inf
               0            /   0                1             1
            chi (q, w) =   |  A (q, w') * (___________ - ___________)  dw'
               SS'         /   SS'          w-w'+ieta      w+w'+ieta
                           0

        Note, The dw' is reduced above in the delta function
        """

        tmp_ww = np.zeros((self.Nw, self.NwS), dtype=complex)

        for iw in range(self.Nw):
            w = wmin + iw * dw
            for jw in range(self.NwS):
                ww = jw * dw # = w' above 
                tmp_ww[iw, jw] = 1. / (w - ww + 1j*eta) - 1. / (w + ww + 1j*eta)

        chi0_wSS = gemmdot(tmp_ww, np.complex128(specfunc_wSS), beta = 0.)

        return chi0_wSS

    
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


    def get_Kxc(self, nt_G, D_asp, orb_MG, P_aMi, gd, setups):
        """Calculate xc kernel in real space. Apply to isolate/periodic sys.

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

        Refer to kernel_finite_sys for the definition of n_S.
        The second term of the XC kernel can be further evaluated by::

            ---- ----           ~ a  *  ~ a         *           ~ a     ~ a
            \    \     < phi  | p   > < p   | phi  >   < phi  | p   > < p   | phi  >
            /___ /___       mu   i1      i2      nu         mu   i3      i4      nu
              a  i1,i2        1                    1          2                    2
                 i3,i4

                    (  /       a     *a        a     *a      a  
                  * | | dr phi (r) phi (r)  f [n ] phi (r) phi (r) 
                    ( /       i1      i2     xc       i3      i4

                       /    ~ a     ~ *a       ~a   ~ *a    ~ a     )
                    - | dr phi (r) phi (r)  f [n ] phi (r) phi (r)  |
                      /       i1      i2     xc       i3      i4    )

        The method four_phi_integrals calculate the () term in the above equation
        """

        # XC Kernel is evaluated in real space

        Kxc_SS = np.zeros((self.nS, self.nS), dtype=orb_MG.dtype)
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

        for i, (n, m) in enumerate(self.Sindex):
            nt1_G = orb_MG[n].conj() * orb_MG[m]
            for j, (p, q) in enumerate(self.Sindex):
                nt2_G = orb_MG[p].conj() * orb_MG[q]
                Kxc_SS[i, j] = gd.integrate(nt1_G.conj() * fxc_G * nt2_G)
                for a, P_Mi in enumerate(P_aMi):
                    P1_I = np.outer(P_Mi[n].conj(), P_Mi[m]).ravel()
                    P2_I = np.outer(P_Mi[p].conj(), P_Mi[q]).ravel()
                    Kxc_SS[i, j] += np.inner(np.inner(P1_I.conj(), J_II[a]), P2_I)

        return Kxc_SS


    def find_kq(self, bzkpt_kG, q):
        """Find the index of k+q for all kpoints in BZ."""

        found = False
        kq = np.zeros(self.nkpt, dtype=int)

        for k in range(self.nkpt):
            # bzkpt(k,:) + q = bzkpt(kq,:)
            tmp1 = bzkpt_kG[k] + q
            for dim in range(3):
                if tmp1[dim] > 0.5: # 
                    tmp1[dim] -= 1.
                elif tmp1[dim] < -0.5:
                    tmp1[dim] += 1.

            for kk in range(self.nkpt):
                tmp2 = bzkpt_kG[kk]
                if (abs(tmp1-tmp2)).sum() < 1e-8:
                    kq[k] = kk
                    found = True
                    break
            if not found:
                raise ValueError('k+q not found')
        return kq


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


    def get_orbitals(self, calc, spos_ac):
        """Obtain LCAO orbital in 3d grid.

        The LCAO orbital is calculated by::
            
                     ----
            phi   =  \     Phi (r-R)
               mu    /___     mu   I
                      I

        Written by Ask.
        """

        if self.nkpt > 1:
            bfs_a = [setup.phit_j for setup in calc.wfs.setups]
            gd = calc.wfs.gd
            bfs = BasisFunctions(gd, bfs_a, calc.wfs.kpt_comm, cut=True)
            bfs.set_positions(spos_ac)
    
            orb_MG = gd.zeros(self.nLCAO)
            C_M = np.identity(self.nLCAO)
            bfs.lcao_to_grid(C_M, orb_MG, q=-1)
        else:  # for Gamma point calculation, wrapper is correct
            from gpaw.lcao.pwf2 import LCAOwrap
            wrapper = LCAOwrap(calc)
            orb_MG = wrapper.get_orbitals()

#        for mu in range(self.nLCAO):
#            for nu in range(self.nLCAO):
#                print calc.wfs.gd.integrate(orb_MG[mu].conj() * orb_MG[nu])
        return orb_MG

    
    def get_P_aMi(self, calc):
        """Calculate P_aMi without k-dependence. """

        if self.nkpt > 1 or calc.wfs.P_aqMi is None:
            a = calc.get_atoms()
            calc = GPAW(mode='lcao', basis='dzp')
            a.set_calculator(calc)
            calc.initialize(a)
            calc.set_positions(a)

        assert calc.wfs.P_aqMi is not None
        P_aMi = []
        for a, P_qMi in calc.wfs.P_aqMi.items():
            P_aMi.append(P_qMi[0])
        
        return P_aMi


    def get_reduced_pair_orbital_index(self, orb_MG, gd, threshold=1e-5):
        """Reduce the dimension of pair orbitals and get the index for new pair orbitals.

        The overlap matrix is calculated by::

            O  = < phi   phi  |  phi   phi  >
             SS       mu   nu       mu    nu

        while phi_mu and phi_nu are normalized before calculating the overlap matrix.
        """

        try:
            self.threshold
        except AttributeError:
            self.threshold = threshold

        Sindex = [] # list of pair-orbital index
        if self.threshold > 0:
            print 'Calculate reduced pair-orbital index with threshold', self.threshold
            for mu in range(self.nLCAO):
                orb1_g = orb_MG[mu] * orb_MG[mu].conj()
    #            A1 = gd.integrate(orb1_g)
    #            orb1_g /= A1
                for nu in range(self.nLCAO):
                    orb2_g = orb_MG[nu] * orb_MG[nu].conj()
    #                A2 = gd.integrate(orb2_g)
    #                orb2_g /= A2
                    overlap = gd.integrate(orb1_g * orb2_g)
                    if overlap > self.threshold:
                        Sindex.append((mu, nu))
                print '   finished', mu, 'cycle, total: ', self.nLCAO
        else:
            print 'Calculate with all pair-orbitals.'
            for mu in range(self.nLCAO):
                for nu in range(self.nLCAO):
                    Sindex.append((mu, nu))
                
        return Sindex      
