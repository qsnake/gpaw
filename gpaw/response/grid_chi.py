from math import pi, sqrt
from os.path import isfile
from scipy.special import sph_jn
import numpy as np
from ase.units import Hartree, Bohr
from ase.data import chemical_symbols

from gpaw.xc_functional import XCFunctional
from gpaw.lcao.pwf2 import LCAOwrap
from gpaw.utilities.blas import gemmdot
from gpaw.utilities import unpack
from gpaw.lfc import BasisFunctions
from gpaw import GPAW

from gpaw.gaunt import gaunt as G_LLL
from gpaw.spherical_harmonics import Y
from gpaw.setup_data import SetupData
from gpaw.setup import Setup
from gpaw.fd_operators import Gradient

class CHI:
    def __init__(self):
        self.xc = 'LDA'
        self.nspin = 1

    def initialize(self, c, q, wmax, dw, eta=0.2):
        try:
            self.ncalc = len(c)
        except:
            self.ncalc = 1
            c = (c,)
            
        self.calc = calc = c[0]
        self.c = c

        bzkpt_kG = calc.get_ibz_k_points()
        self.nkpt = bzkpt_kG.shape[0]
        kweight = calc.get_k_point_weights()
    
        try:
            self.nband
        except:
            self.nband = calc.wfs.nbands
        self.nvalence = calc.wfs.nvalence

        assert calc.wfs.nspins == 1
        assert calc.get_bz_k_points().shape == calc.get_ibz_k_points().shape
    
        self.acell = calc.atoms.cell / Bohr
        self.get_primitive_cell()

        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]

        self.h_c = calc.wfs.gd.h_cv.diagonal()

        if self.ncalc == 1:

            # obtain eigenvalues, occupations
            self.e_kn = np.array([calc.get_eigenvalues(kpt=k)
                        for k in range(self.nkpt)]) / Hartree
            self.f_kn = np.array([calc.get_occupation_numbers(kpt=k)
                        for k in range(self.nkpt)])    

        else:
            
            assert self.ncalc == 2
            
            # obtain eigenvalues, occupations
            self.e1_kn = np.array([c[0].get_eigenvalues(kpt=k)
                         for k in range(self.nkpt)]) / Hartree
            self.f1_kn = np.array([c[0].get_occupation_numbers(kpt=k)
                         for k in range(self.nkpt)])
    
            self.e2_kn = np.array([c[1].get_eigenvalues(kpt=k)
                         for k in range(self.nkpt)]) / Hartree
            self.f2_kn = np.array([c[1].get_occupation_numbers(kpt=k)
                         for k in range(self.nkpt)])
    
        self.qr = np.zeros(self.nG)

        # construct q.r
        h_c = self.h_c
        self.q = q
        self.qq = qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])

        for i in range(self.nG[0]):
            for j in range(self.nG[1]):
                for k in range(self.nG[2]):
                    tmp = np.array([i*h_c[0], j*h_c[1], k*h_c[2]])
                    self.qr[i,j,k] = np.inner(qq, tmp)

        # unit conversion
        self.wmin = 0
        self.wmax  = wmax / Hartree
        self.dw = dw / Hartree
        self.Nw = int((self.wmax - self.wmin) / self.dw) + 1
        self.eta = eta / Hartree

        self.epsilonRPA = np.zeros(self.Nw, dtype = complex)
        self.epsilonM = 0.

        self.print_stuff()

        return


    def periodic(self):
        
        if self.ncalc == 1:
            self.OpticalLimit()
            print 'Optical limit calculation !'
        else:
            self.ShiftKpoint()
            print 'Numerically shift kpoint calculation !'

    def OpticalLimit(self):

        calc = self.calc

        setups = calc.wfs.setups
        gd = calc.wfs.gd
        
        f_kn = self.f_kn
        e_kn = self.e_kn
        qq = self.qq
        eta = self.eta

        chi0_w = np.zeros(self.Nw, dtype = complex)

        phi_ii = {}
        for a, id in enumerate(setups.id_a):
            Z, type, basis = id
            if not phi_ii.has_key(Z):
                phi_ii[Z] = self.two_phi_derivative(Z)

        d_c = [Gradient(gd, i, dtype=complex).apply for i in range(3)]
        dpsit_G = gd.empty(dtype=complex)
        tmp = np.zeros((3), dtype=complex)


        for k in range(self.nkpt):
            kpt = calc.wfs.kpt_u[k]
            P_ani = kpt.P_ani
            psit_nG = kpt.psit_nG    

            rho_nn = np.zeros((self.nband, self.nband), dtype=complex)            

            for n in range(self.nband):
                for m in range(self.nband):
                    # G = G' = 0 <psi_nk | e**(-iqr) | psi_n'k+q>
                    
                    if np.abs(e_kn[k, m] - e_kn[k, n]) > 1e-10:
                        for ix in range(3):
                            d_c[ix](psit_nG[m], dpsit_G, kpt.phase_cd)
                            tmp[ix] = gd.integrate( psit_nG[n].conj() * dpsit_G)
                        rho_nn[n, m] = -1j * np.inner(qq, tmp) 

                        # PAW correction
                        for a, id in enumerate(setups.id_a):
                            Z, type, basis = id
                            P_ii = np.outer(P_ani[a][n].conj(), P_ani[a][m])
                            rho_nn[n, m] += (P_ii * phi_ii[Z]).sum() 
                        rho_nn[n, m] /= e_kn[k, m] - e_kn[k, n]

            # construct (f_nk - f_n'k+q) / (w + e_nk - e_n'k+q + ieta )
            C_nn = np.zeros((self.nband, self.nband), dtype=complex)
            for iw in range(self.Nw):
                w = iw * self.dw
                for n in range(self.nband):
                    for m in range(self.nband):
                        if  np.abs(f_kn[k, n] - f_kn[k, m]) > 1e-10:
                            C_nn[n, m] = (f_kn[k, n] - f_kn[k, m]) / (
                             w + e_kn[k, n] - e_kn[k, m] + 1j * eta)

                # get chi0(G=0,G'=0,w)                
                chi0_w[iw] += (rho_nn * C_nn * rho_nn.conj()).sum()

#            chi0_w *= kweight[k] * calc.get_ibz_k_points().shape[0]
            # Obtain Macroscopic Dielectric Constant
            for n in range(self.nband):
                for m in range(self.nband):
                    C_nn[n, m] = 0.
                    if np.abs(f_kn[k, n] - f_kn[k, m]) > 1e-10:
                        C_nn[n, m] = (f_kn[k, n] - f_kn[k, m]) / (
                                  e_kn[k, n] - e_kn[k, m] )
            self.epsilonM += (rho_nn * C_nn * rho_nn.conj()).sum()

            print 'finished kpoint', k
            
        for iw in range(self.Nw):
            self.epsilonRPA[iw] =  1 - 4 * pi / np.inner(qq, qq) * chi0_w[iw] / self.vol


    def ShiftKpoint(self):

        f1_kn = self.f1_kn
        f2_kn = self.f2_kn
        e1_kn = self.e1_kn
        e2_kn = self.e2_kn
        eta = self.eta
        qr = self.qr
        qq = self.qq
        c = self.c
        setups = c[0].wfs.setups
        gd = c[0].wfs.gd

        chi0_w = np.zeros(self.Nw, dtype = complex)
        rho_nn = np.zeros((self.nband, self.nband), dtype=complex)        

        # calculate <phi_i | e**(-iq.r) | phi_j>
        phi_ii = {}
        R_a = c[0].atoms.positions / Bohr
        
        for a, id in enumerate(setups.id_a):
            Z, type, basis = id
            if not phi_ii.has_key(Z):
                phi_ii[Z] = ( self.two_phi_planewave_integrals(Z)
                                  * np.exp(-1j * np.inner(qq, R_a[a])) )

        # calculate chi0
        for k in range(10):#self.nkpt):
            kpt0 = c[0].wfs.kpt_u[k]
            kpt1 = c[1].wfs.kpt_u[k]
            P1_ani = kpt0.P_ani
            P2_ani = kpt1.P_ani
            psit1_nG = kpt0.psit_nG
            psit2_nG = kpt1.psit_nG
            
            for n in range(self.nband):
                for m in range(self.nband):
                    # G = G' = 0 <psi_nk | e**(-iqr) | psi_n'k+q>
                    rho_nn[n, m] = gd.integrate( psit1_nG[n].conj()
                                         * psit2_nG[m]
                                         * np.exp(-1j * qr) )
                    # PAW correction 
                    for a, id in enumerate(setups.id_a):
                        Z, type, basis = id
                        P_ii = np.outer(P1_ani[a][n].conj(), P2_ani[a][m])
                        rho_nn[n, m] += (P_ii * phi_ii[Z]).sum()

            # construct (f_nk - f_n'k+q) / (w + e_nk - e_n'k+q + ieta )
            C_nn = np.zeros((self.nband, self.nband), dtype=complex)
            for iw in range(self.Nw):
                w = iw * self.dw
                for n in range(self.nband):
                    for m in range(self.nband):
                        if  np.abs(f1_kn[k, n] - f2_kn[k, m]) > 1e-10:
                            C_nn[n, m] = (f1_kn[k, n] - f2_kn[k, m]) / (
                             w + e1_kn[k, n] - e2_kn[k, m] + 1j * eta)

                # get chi0(G=0,G'=0,w)                
                chi0_w[iw] += (rho_nn * C_nn * rho_nn.conj()).sum()

            # Obtain Macroscopic Dielectric Constant
            for n in range(self.nband):
                for m in range(self.nband):
                    C_nn[n, m] = 0.
                    if np.abs(f1_kn[k, n] - f2_kn[k, m]) > 1e-6:
                        C_nn[n, m] = (f1_kn[k, n] - f2_kn[k, m]) / (
                                  e1_kn[k, n] - e2_kn[k, m] )
            self.epsilonM += (rho_nn * C_nn * rho_nn.conj()).sum()

            print 'finished k', k


        for iw in range(self.Nw):
            self.epsilonRPA[iw] =  1 - 4 * pi / np.inner(qq, qq) * chi0_w[iw] / self.vol

    def check_ortho(self, calc, psit_knG):
        # Check the orthonormalization of wfs
        gd = calc.wfs.gd
        setups = calc.wfs.setups
        rho_nn = np.zeros((self.nband, self.nband))

        phi_ii = {}
        
        for a in range(len(setups)):
            phi_p = setups[a].Delta_pL[:,0].copy()
            phi_ii[a] = unpack(phi_p) * sqrt(4*pi)

        for k in range(self.nkpt):
            P_ani = calc.wfs.kpt_u[k].P_ani
            for n in range(self.nband):
                for m in range(self.nband):
                    rho_nn[n, m] = gd.integrate(psit_knG[k,n].conj() * psit_knG[k,m])

                    for a in range(len(setups)):
                        P_ii = np.outer(P_ani[a][n].conj(), P_ani[a][m])
                        rho_nn[n, m] += (P_ii * phi_ii[a]).sum()
                    #print 'after PAW', (n, m), rho_nn[n, m]
                    if n == m and np.abs(rho_nn[n, m] -1) > 1e-10:
                        print 'after PAW', (n, m), rho_nn[n, m]
                    if n != m and np.abs(rho_nn[n, m]) > 1e-10:
                        print 'after PAW', (n, m), rho_nn[n, m]

        return


    def get_primitive_cell(self):

        a = self.acell

        self.vol = np.abs(np.dot(a[0],np.cross(a[1],a[2])))
        self.BZvol = (2. * pi)**3 / self.vol

        b = np.linalg.inv(a)
    
        self.bcell = 2 * pi * b

        assert np.abs((np.dot(a, self.bcell) - 2.*pi*np.eye(3)).sum()) < 1e-10

        return


    def two_phi_planewave_integrals(self, Z):

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
        phi_ii = np.zeros((ni, ni), dtype=complex)
        j_lg = np.zeros((lmax, ng))
   
        # Store (phi_j1 * phi_j2 - phit_j1 * phit_j2 ) for further use
        tmp_jjg = np.zeros((s.nj, s.nj, ng))
        for j1 in range(s.nj):
            for j2 in range(s.nj): 
                tmp_jjg[j1, j2] = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]

        qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
        q = np.sqrt(np.inner(qq, qq)) # calculate length of q+G
        
        # Calculating spherical bessel function
        for ri in range(ng):
            j_lg[:,ri] = sph_jn(lmax - 1,  q*r_g[ri])[0]

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

                phi_ii += R_ii * Y(li**2 + mi, qq[0], qq[1], qq[2]) * (-1j)**li

        phi_ii *= 4 * pi

        return phi_ii


    def two_phi_derivative(self, Z):

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
        for j, l in enumerate(s.l_j):
            for m in range(2 * l + 1):
                L_i.append(l**2 + m)
                j_i.append(j)

        Lmax = s.Lmax
        nj = s.nj
        ni = len(L_i)
        nii = ni * (ni + 1) // 2
        dphidr_jg = np.zeros(np.shape(phi_jg))
        dphitdr_jg = np.zeros(np.shape(phit_jg))
        phi_ii = np.zeros((ni, ni))

        from gpaw.xc_correction import A_Liy
        from gpaw.sphere import Y_nL, points, weights
        from gpaw.grid_descriptor import RadialGridDescriptor

        rgd = RadialGridDescriptor(r_g, dr_g)
        ny = len(points)
        
        for j in range(nj):
            rgd.derivative(phi_jg[j], dphidr_jg[j])
            rgd.derivative(phit_jg[j], dphitdr_jg[j])
        ##second term
        for y in range(ny):
            Y_L = Y_nL[y]
            weight = weights[y]
            for i1 in range(ni):
                L1 = L_i[i1]
                j1 = j_i[i1]
                for i2 in range(ni):
                    L2 = L_i[i2]
                    j2 = j_i[i2]

                    c = Y_L[L1]*Y_L[L2] # c is a number
                    temp  = c * ( phi_jg[j1] *  dphidr_jg[j2]
                                  - phit_jg[j1] *  dphitdr_jg[j2] )
                    phi_g = ( temp * self.qq[0] + temp * self.qq[1]
                                                + temp * self.qq[2] )
                    
                    A_Li = A_Liy[:Lmax, :, y]
                    temp = ( A_Li[L2, 0] * self.qq[0] + A_Li[L2, 1] * self.qq[1]
                             + A_Li[L2, 2] * self.qq[2] ) * Y_L[L1]
                    temp *= phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2] 
                    temp[1:] /= r_g[1:]
                    temp[0] = temp[1]
                    phi_g += temp

                    phi_ii[i1, i2] += rgd.integrate(phi_g) * weight
        
        return phi_ii * (-1j)


    def print_stuff(self):

        print 
        print 'Parameters used:'
        print
        print 'Number of bands:', self.nband
        print 'Number of kpoints:', self.nkpt
        print 'Unit cell (a.u.):'
        print self.acell
        print 'Reciprocal cell (1/a.u.)'
        print self.bcell
        print 'Volome of cell (a.u.**3):', self.vol
        print 'BZ volume (1/a.u.**3):', self.BZvol
        print
        print 'Number of frequency points:', self.Nw
        print 'Number of Grid points / G-vectors, and in total:', self.nG, self.nG0
        print 'Grid spacing (a.u.):', self.h_c
        print
        print 'q in reduced coordinate:', self.q
        print 'q in cartesian coordinate:', self.qq


    def check_sum_rule(self):

        N = 0
        for iw in range(self.Nw):
            w = iw * self.dw
            N += np.imag(self.epsilonRPA[iw]) * w 
        N *= self.dw * self.vol / (2 * pi**2)
        
        print 'sum rule:'
        print 'N = ', N, (N - self.nvalence) / self.nvalence * 100, '% error'


    def get_microscopic_dielectric_constant(self):

        qq = self.qq
        epsilonM = self.epsilonM
        
        epsilonM *=  - 4 * pi / np.inner(qq, qq) / self.vol
        epsilonM += 1.

        return epsilonM
        

    def get_dielectric_function(self):
        return self.epsilonRPA

    def get_absorption_spectrum(self, filename='Absorption'):
        epsilonRPA = self.epsilonRPA
        f = open(filename,'w')
        for iw in range(self.Nw):
            print >> f, iw * self.dw * Hartree, np.real(epsilonRPA[iw]), np.imag(epsilonRPA[iw])

