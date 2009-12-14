from math import pi, sqrt
from os.path import isfile

import numpy as np
from ase.units import Hartree, Bohr

from gpaw.xc_functional import XCFunctional
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.utilities.blas import gemmdot

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
        bzkpt_kG = calc.get_bz_k_points()
        self.nband = calc.get_number_of_bands()
        self.nkpt = bzkpt_kG.shape[0]
        self.acell = calc.atoms.cell / Bohr

        # obtain eigenvalues, occupations, LCAO coefficients and wavefunctions
        e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(self.nkpt)])
        f_kn = np.array([calc.get_occupation_numbers(kpt=k)
                          for k in range(self.nkpt)])
        
        if calc.wfs.kpt_u[0].C_nM is not None:
            C_knM = np.array([kpt.C_nM.copy() for kpt in calc.wfs.kpt_u])
        elif isfile('C_knM.npz'): 
            foo = np.load('C_knM.npz')
            C_knM = foo['C']
            assert C_knM.shape[0] == self.nkpt and (
                   C_knM.shape[1] == self.nband)
        else:
            raise ValueError('C_knM not exist!')            

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
        self.q = q

        self.Nw = int((wmax - wmin) / self.dw) + 1
        self.NwS = int(wcut/self.dw) + 1
        self.nLCAO = C_knM.shape[2]
        self.nS = self.nLCAO **2
        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        self.h_c = calc.wfs.gd.h_c

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
        print 'Number of pair orbitals:', self.nS
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

            return e_kn, f_kn, C_knM, orb_MG, spos_ac, nt_G, chi0_wSS 
        else:
            return e_kn, f_kn, C_knM, orb_MG, spos_ac, nt_G, bzkpt_kG


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

        print 'Calculating chi0 directly'
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
                        # C C C C
                        tmp = focc * np.outer(tmp, tmp.conj()) # tmp[nS,nS]
                        # calculate delta function
                        deltaw = self.delta_function(w0, dw,self.NwS, sigma)
                        for wi in range(self.NwS):
                            if deltaw[wi] > 1e-5:
                                specfunc_wSS[wi] += tmp * deltaw[wi]
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

