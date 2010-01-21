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

    def periodic(self, calc, q, wcut, wmin, wmax, dw, eta):

        bzkpt_kG = calc.get_bz_k_points()
        self.nkpt = bzkpt_kG.shape[0]

        self.nband = calc.wfs.nbands
        self.nvalence = calc.wfs.nvalence
        assert calc.wfs.nspins == 1

        self.acell = calc.atoms.cell / Bohr
        self.get_primitive_cell()
        
        # obtain eigenvalues, occupations
        e_kn = np.array([calc.get_eigenvalues(kpt=k) for k in range(self.nkpt)])
        f_kn = np.array([calc.get_occupation_numbers(kpt=k) for k in range(self.nkpt)])

        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        print 'grid size', self.nG

        # obtain pseudo wfs
        if calc.wfs.kpt_u[0].psit_nG is not None:
            psit_knG = np.zeros((self.nkpt, self.nband, self.nG[0],
                                 self.nG[1], self.nG[2]),dtype=complex)
            for k in range(self.nkpt):
                for n in range(self.nband):
                    psit_knG[k, n]= calc.wfs.gd.zero_pad(calc.wfs.get_wave_function_array(n,k,0))
            np.savez('psit_knG', psit = psit_knG)
        else:
            foo= np.load('psit_knG.npz')
            psit_knG = foo['psit']
            
    
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

        
        # Construte k and k + q index
        self.q = q
        if self.OpticalLimit:
            kq = np.zeros(self.nkpt, dtype=int)
            for k in range(self.nkpt):
                kq[k] = k
        else:
            kq = self.find_kq(bzkpt_kG, q)
        print 'q:', self.q

        self.h_c = h_c = calc.wfs.gd.h_c
        Li = np.array([3, 1, 2])
        d_nn = np.zeros((self.nband, self.nband, 3))
        qr = np.zeros(self.nG)

        # construct q.r
        self.qq = qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
        print qq
        for i in range(self.nG[0]):
            for j in range(self.nG[1]):
                for k in range(self.nG[2]):
                    tmp = np.array([i*h_c[0], j*h_c[1], k*h_c[2]])
                    qr[i,j,k] = np.inner(qq, tmp)

        # unit conversion
        e_kn = e_kn / Hartree
        wcut = wcut / Hartree
        wmin = 0 #wmin / Hartree
        wmax = wmax / Hartree
        self.dw = dw / Hartree
        self.Nw = int((wmax - wmin) / self.dw) + 1
        eta = eta / Hartree

        setups = calc.wfs.setups
        chi0_w = np.zeros(self.Nw, dtype = complex)
        rho_nn = np.zeros((self.nband, self.nband), dtype=complex)

        # calculate <phi_i | e**(-iq.r) | phi_j>
        phi_ii = {}
        R_a = calc.atoms.positions / Bohr

        for a, id in enumerate(setups.id_a):
                Z, type, basis = id
                if self.OpticalLimit:
                    if not phi_ii.has_key(Z):
                        phi_ii[Z] = self.two_phi_derivative(Z)
                else:
                    if not phi_ii.has_key(Z):
                        phi_ii[Z] = ( self.two_phi_planewave_integrals(Z)
                                  * np.exp(-1j * np.inner(qq, R_a[a])) )
                

        # calculate chi0_GG
        # Not possible at the moment. chi0_wGG requires too much memory.
#        chi0_wGG = np.zeros((self.Nw, self.nG0, self.nG0), dtype = complex)
#
#        for k in range(self.nkpt):
#            for n in range(self.nband):
#                for m in range(self.nband):
#                    if ( f_kn[k, n] - f_kn[kq[k], m] > 1e-30
#                        and wmax + e_kn[k, n] - e_kn[kq[k], m] > 10.):
#                        rho = psit_knG[k, n].conj() * psit_knG[kq[k], m] * np.exp(-1j * qr)
#                        rho_G = np.fft.fftn(rho).ravel() * self.vol / self.nG0
#                        # PAW correction to only G = 0 component
#                        for a, id in enumerate(setups.id_a):
#                            Z, type, basis = id
#                            P_ii = np.outer(P1_ani[a][n].conj(), P2_ani[a][m])
#                            rho_G[0] += (P_ii * phi_ii[Z]).sum()
#
#                        for iw in range(self.Nw):
#                            w = iw * self.dw
#                            
#                            C =  (f_kn[k, n] - f_kn[kq[k], m]) / (
#                             w + e_kn[k, n] - e_kn[kq[k], m] + 1j * eta)
#
#                            chi0_wGG[iw] += C * np.outer(rho_G, rho_G.conj())
#                            
        # calculate chi0
        if self.OpticalLimit:
            d_c = [Gradient(gd, c, dtype=psit_knG.dtype).apply for c in range(3)]
            dpsit_G = gd.empty(dtype=psit_knG.dtype)
            tmp = np.zeros((3), dtype=psit_knG.dtype)
            
        for k in range(self.nkpt):
            P1_ani = calc.wfs.kpt_u[k].P_ani
            P2_ani = calc.wfs.kpt_u[kq[k]].P_ani
            for n in range(self.nband):
                for m in range(self.nband):
                    # G = G' = 0 <psi_nk | e**(-iqr) | psi_n'k+q>
                    
                    if self.OpticalLimit:
                        if e_kn[k, m] - e_kn[k, n] != 0:
                            for ix in range(3):
                                d_c[ix](psit_knG[k, m], dpsit_G, calc.wfs.kpt_u[k].phase_cd)
                                tmp[ix] = gd.integrate( psit_knG[k, n].conj() * dpsit_G)
                            rho_nn[n, m] = -1j * np.inner(qq, tmp) / (e_kn[k, m] - e_kn[k, n])

                            # Check optical limit length scale
                            # Result wrong, can not use this !!!!
#                            rho_nn[n, m] = -1j * gd.integrate(
#                                                    psit_knG[k, n].conj()
#                                                  * qr * psit_knG[k, m] )
                            
                            # PAW correction
                            for a, id in enumerate(setups.id_a):
                                Z, type, basis = id
                                P_ii = np.outer(P1_ani[a][n].conj(), P2_ani[a][m])
                                rho_nn[n, m] += (P_ii * phi_ii[Z]).sum()
                    else:
                        rho_nn[n, m] = gd.integrate( psit_knG[k, n].conj()
                                             * psit_knG[kq[k], m]
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
                        C_nn[n, m] = (f_kn[k, n] - f_kn[kq[k], m]) / (
                             w + e_kn[k, n] - e_kn[kq[k], m] + 1j * eta)
                # get chi0(G=0,G'=0,w)                
                chi0_w[iw] += (rho_nn * C_nn * rho_nn.conj()).sum()
                
        epsilonRPA = np.zeros(self.Nw)
        for iw in range(self.Nw):
            epsilonRPA[iw] =  - 4 * pi / np.inner(qq, qq) * np.imag(chi0_w[iw])

        # Check sum rule
        N = 0
        for iw in range(self.Nw):
            w = iw * self.dw
            N += epsilonRPA[iw] * w 
        N *= self.dw / (2 * pi**2)
        
        print 'sum rule:'
        print 'N = ', N, (N - self.nvalence) / self.nvalence * 100, '% error'

        f = open('Absorption','w')
        for iw in range(self.Nw):
            print >> f, iw * self.dw * Hartree, epsilonRPA[iw]
#        import pylab as pl
#        pl.plot(epsilonRPA)
#        pl.show()
        
        return

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

    def sph_jn(self, n, z):

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

        a = self.acell

        self.vol = np.dot(a[0],np.cross(a[1],a[2]))
        self.BZvol = (2. * pi)**3 / self.vol

        b = np.zeros_like(a)
        b[0] = np.cross(a[1], a[2])
        b[1] = np.cross(a[2], a[0])
        b[2] = np.cross(a[0], a[1])
        self.bcell = 2. * pi * b / self.vol

        return

    def kernel_extended_sys(self):

        Gvec = self.get_Gvectors()
        
        Kcoul_G = np.zeros(self.nG0)

        assert (self.q).any() != 0

        for i in range(self.nG0):
            # get q+G vector 
            xx = np.array([np.inner(np.float64((Gvec[i]) + self.q), self.bcell[:,j]) for j in range(3)])
            Kcoul_G[i] = 1. /  np.inner(xx, xx)
        Kcoul_G *= 4. * pi 
        

        return Kcoul_G

    def get_Gvectors(self):
        
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


    def finite(self, calc, q, wcut, wmin, wmax, dw, eta):

        self.nband = calc.wfs.nbands
        self.nkpt = 1 
        self.nvalence = calc.wfs.nvalence
                
        # obtain eigenvalues, occupations
        e_n = calc.get_eigenvalues(kpt=0)
        f_n = calc.get_occupation_numbers(kpt=0)

        # obtain pseudo wfs
        assert calc.wfs.nspins == 1
        psit_nG= np.array([calc.wfs.gd.zero_pad(calc.wfs.get_wave_function_array(n,0,0))
                                   for n in range(self.nband)])

        self.nG = psit_nG.shape[1:]
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        print 'grid size', self.nG
        # obtain the paw term for wfs
        #calc.wfs.kpt_u[k].P_ani
        
        gd = calc.wfs.gd
        setups = calc.wfs.setups
        rho_nn = np.zeros((self.nband, self.nband))
        P_ani = calc.wfs.kpt_u[0].P_ani

        # check orthonormalization of wfs
        phi_ii = {}
        for a in range(len(setups)):
            phi_p = setups[a].Delta_pL[:,0].copy()
            phi_ii[a] = unpack(phi_p) * sqrt(4*pi)
                    
        for n in range(self.nband):
            for m in range(self.nband):
                rho_nn[n, m] = gd.integrate(psit_nG[n].conj() * psit_nG[m])
#                print 'before PAW', (n, m), rho_nn[n, m]
                for a in range(len(setups)):
                    P_ii = np.outer(P_ani[a][n].conj(), P_ani[a][m])
                    rho_nn[n, m] += (P_ii * phi_ii[a]).sum()
                if n == m and np.abs(rho_nn[n, m] -1) > 1e-10:
                    print 'after PAW', (n, m), rho_nn[n, m]
                if n != m and np.abs(rho_nn[n, m]) > 1e-10:
                    print 'after PAW', (n, m), rho_nn[n, m]
                        
        # get dipole
        N_gd = self.nG
        self.h_c = calc.wfs.gd.h_c

        Li = np.array([3, 1, 2])
        d_nn = np.zeros((self.nband, self.nband, 3))
        r = np.zeros(self.nG)


        for ix in range(3):
            for i in range(N_gd[0]):
                for j in range(N_gd[1]):
                    for k in range(N_gd[2]):
                        if ix == 0:
                            r[i,j,k] = i*self.h_c[0]
                        elif ix == 1:
                            r[i,j,k] = j*self.h_c[1] 
                        else:
                            r[i,j,k] = k*self.h_c[2]

            for n in range(self.nband):
                for m in range(self.nband):
                    d_nn[n, m, ix] = gd.integrate(psit_nG[n].conj() * psit_nG[m] * r)
                    # print 'before PAW', (n, m), d_nn[n, m]
                    for a in range(len(setups)):
                        phi_p = setups[a].Delta_pL[:,Li[ix]].copy()
                        phi_ii[a] = unpack(phi_p) * sqrt(4*pi/3)
                        P_ii = np.outer(P_ani[a][n].conj(), P_ani[a][m])
                        d_nn[n, m, ix] += (P_ii * phi_ii[a]).sum()
                    # print 'after PAW', (n, m), d_nn[n, m]
                    

        e_n = e_n / Hartree
        wcut = wcut / Hartree
        wmin = 0 #wmin / Hartree
        wmax = wmax / Hartree
        self.dw = dw / Hartree
        self.Nw = int((wmax - wmin) / self.dw) + 1
        eta = eta / Hartree

        C_nn = np.zeros((self.nband, self.nband), dtype=complex)
        S = np.zeros((self.Nw, 3))
        for iw in range(self.Nw):
            w = iw * self.dw
            for n in range(self.nband):
                for m in range(self.nband):
                    C_nn[n, m] = (f_n[n] - f_n[m]) / (w + e_n[n] - e_n[m] + 1j * eta)
                    
            for ix in range(3):
                S[iw, ix] = - 2 * w / pi * np.imag((C_nn * d_nn[:,:,ix] * d_nn[:,:,ix].conj()).sum())

        print 'sum rule'
        N = S.sum() * self.dw / 3
        print 'N = ', N, (N - self.nvalence) / self.nvalence * 100 ,'% error'

        return

                


